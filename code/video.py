import anomalib
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.nn.functional import interpolate
from anomalib.models import Patchcore

# Disable Rich formatting
os.environ["RICH"] = "0"

def main():
    
    # LOAD TRAINED MODEL
    # ==========================================
    BASE_DIR = Path(r"C:\Users\patryk\Desktop\projekt_cnn")
    results_dir = BASE_DIR / "results"
    
    # The script automatically searches for the latest saved "brain" on the disk (.ckpt)
    ckpt_files = list(results_dir.rglob("*.ckpt"))
    if not ckpt_files:
        print("Saved model file (.ckpt) not found in the results folder!")
        return
        
    latest_ckpt = max(ckpt_files, key=os.path.getmtime)
    print(f"\nLoading trained knowledge from file:\n{latest_ckpt.name}")
    


    model = Patchcore.load_from_checkpoint(latest_ckpt, weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    

    video_path = BASE_DIR / "ZdjAparat" / "Filmy" / "MVI_7459.AVI" 
    
    # SENSITIVITY THRESHOLD
    # set to e.g. 15.0 or 20.0
    threshold_score = 17.0
    
    analyze_every_n = 3 # Every nth frame to analyze (speeds up the process)
    
    if not video_path.exists():
        print(f"\nMissing video file at path:\n{video_path}")
        return
        
    output_video_path = str(BASE_DIR / 'result_mp4.mp4')
    
    # VIDEO GENERATION
    # ==========================================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nStarting bounding box application: {total_frames} frames, {fps:.2f} FPS")
    out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    frame_idx = 0
    anomaly_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        if frame_idx % analyze_every_n != 0:
            out_video.write(frame)
            continue
            
        if frame_idx % 30 == 0:
            print(f"Processed frame: {frame_idx}/{total_frames}")
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        img_tensor = transform(pil_img).to(device)
        img_norm = (img_tensor - mean) / std
        
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
                if area < 20 or area > 0.3 * frame_area: continue
                mask_cnt = np.zeros_like(mask)
                cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                mean_val = cv2.mean(amap, mask=mask_cnt)[0]
                if mean_val < p_val: continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw > 10 and bh > 10:
                    cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                    
            cv2.putText(frame, f"Anomaly! {pred_score:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        out_video.write(frame)
        
    cap.release()
    out_video.release()
    print(f"\nDone! Successfully applied bounding boxes. Detected anomalies in {anomaly_frames} frames.")
    print("The video 'result.mp4' is waiting in your main folder on the desktop.")

if __name__ == '__main__':
    main()