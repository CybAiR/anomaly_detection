import os
import cv2
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.nn.functional import interpolate
from sklearn.metrics import roc_curve, auc

# Disable Rich formatting for Anomalib
os.environ["RICH"] = "0"

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

def main():

    # MAIN PATHS
    # ==========================================
    BASE_DIR = Path(r"C:\Users\patryk\Desktop\projekt_cnn")
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # TRAINING AND TESTING
    # ==========================================

    datamodule = Folder(
        name="mars_dataset",
        root=str(BASE_DIR),
        normal_dir="ZdjMaciek/Normalne",
        abnormal_dir="ZdjMaciek/Anomalie",
        normal_test_dir="ZdjPatryk/Normalne",
        train_batch_size=4,        
        eval_batch_size=8,         
        num_workers=0,             # Disables Dataloader multithreading, saving RAM
    )
    
    model = Patchcore(
    coreset_sampling_ratio=0.01, 
    backbone="resnet18", 
    pre_trained=True
)
    
    engine = Engine(
        default_root_dir=str(results_dir),
        enable_progress_bar=True,
    )
    
    print("\nStart fit (Training)...")
    engine.fit(datamodule=datamodule, model=model)
    print("Fit finished. Start test...")
    test_results = engine.test(datamodule=datamodule, model=model)
    print("Test finished.")
    
    # ROC CURVE
    # ==========================================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    device = model.device
    model.eval()
    
    y_true = []
    y_score = []
    
    print("\nGenerating ROC curve...")
    
    # add your path to files
    # Class 0
    for img_path in (BASE_DIR / "ZdjPatryk/Normalne").glob("*.*"):
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image).to(device)
        img_norm = (img_tensor - mean.to(device)) / std.to(device)
        with torch.no_grad():
            out = model.model(img_norm.unsqueeze(0))
        y_true.append(0)
        y_score.append(out.pred_score.item())
    
    # Class 1
    for img_path in (BASE_DIR / "ZdjMaciek/Anomalie").glob("*.*"):
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image).to(device)
        img_norm = (img_tensor - mean.to(device)) / std.to(device)
        with torch.no_grad():
            out = model.model(img_norm.unsqueeze(0))
        y_true.append(1)
        y_score.append(out.pred_score.item())
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('PatchCore model ROC curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(BASE_DIR / "ROC_curve.png")
    
    # VIDEO PROCESSING
    # ==========================================
    video_path = BASE_DIR / "ZdjAparat" / "Filmy" / "MVI_7440.AVI"  # add your path
    
    if not video_path.exists():
        print(f"\nMissing video file at path: {video_path}")
    else:
        output_dir = BASE_DIR / "anomaly_frames"
        output_dir.mkdir(exist_ok=True)
        alert_log = BASE_DIR / "alert_video.log"
        output_video_path = str(BASE_DIR / 'anomaly_output.mp4')
    
        threshold_score = 20
        analyze_every_n = 3
    
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"\nProcessing video: {total_frames} frames, {fps:.2f} FPS, {w}x{h}")
    
        out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
        model.eval()
        frame_idx = 0
        anomaly_frames = 0
    
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
    
            if frame_idx % analyze_every_n != 0:
                out_video.write(frame)
                continue
    
            if frame_idx % 30 == 0:
                print(f"Frame {frame_idx}/{total_frames}")
    
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            img_tensor = transform(pil_img).to(device)
            img_norm = (img_tensor - mean.to(device)) / std.to(device)
    
            with torch.no_grad():
                output = model.model(img_norm.unsqueeze(0))
    
            pred_score = output.pred_score.item()
            is_anomaly = pred_score > threshold_score
    
            if is_anomaly:
                anomaly_frames += 1
                amap_resized = interpolate(output.anomaly_map, size=(h, w), mode='bilinear', align_corners=False)
                amap = amap_resized.squeeze().cpu().numpy()
    
                amap = cv2.GaussianBlur(amap, (5, 5), 0)
                p_val = np.percentile(amap, 99)
                adaptive_thresh = p_val + (amap.max() - p_val) * 0.02
                mask = (amap > adaptive_thresh).astype(np.uint8) * 255
    
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                frame_area = h * w
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 20 or area > 0.3 * frame_area:
                        continue
                    mask_cnt = np.zeros_like(mask)
                    cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                    mean_val = cv2.mean(amap, mask=mask_cnt)[0]
                    
                    if mean_val < p_val:
                        continue
                        
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    if bw > 10 and bh > 10:
                        cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
    
                cv2.putText(frame, f"Anomaly! {pred_score:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                out_path = output_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
    
                with open(alert_log, "a") as f:
                    f.write(f"{datetime.datetime.now()}: frame {frame_idx} - anomaly {pred_score:.3f}\n")
    
            out_video.write(frame)
    
        cap.release()
        out_video.release()
        print(f"End. Anomalies in {anomaly_frames} frames. Video saved to desktop!")

if __name__ == '__main__':
    main()