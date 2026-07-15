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
    
    ckpt_files = list(results_dir.rglob("*.ckpt"))
    if not ckpt_files:
        print(".ckpt file not found!")
        return
        
    latest_ckpt = max(ckpt_files, key=os.path.getmtime)
    print(f"\nLoading model:\n{latest_ckpt.name}")
    
    model = Patchcore.load_from_checkpoint(latest_ckpt, weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # YOUR IMAGE SETTINGS
    # ==========================================
    
    # PROVIDE THE PATH TO THE TEST IMAGE (.jpg) HERE
    image_path = BASE_DIR / "ZdjAparat" / "Anomalie" / "IMG_7510.jpg" 
    
    # SENSITIVITY THRESHOLD
    threshold_score = 15.0 
    
    if not image_path.exists():
        print(f"\nMissing image at path:\n{image_path}")
        return
        
    # IMAGE ANALYSIS
    # ==========================================
    print("\nAnalyzing image...")
    frame = cv2.imread(str(image_path))
    h, w = frame.shape[:2]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    img_tensor = transform(pil_img).to(device)
    img_norm = (img_tensor - mean) / std
    
    with torch.no_grad():
        output = model.model(img_norm.unsqueeze(0))
        
    pred_score = output.pred_score.item()
    print(f"Anomaly score: {pred_score:.3f}")
    
    is_anomaly = pred_score > threshold_score
    
    if is_anomaly:
        print("Anomaly detected! Applying bounding boxes...")
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
            if area < 50 or area > 0.3 * frame_area: continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 3)
                
        cv2.putText(frame, f"Anomaly! Score: {pred_score:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    else:
        print("Image normal. No anomalies.")
        cv2.putText(frame, f"OK. Score: {pred_score:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
    output_path = str(BASE_DIR / f"wynik_{image_path.name}")
    cv2.imwrite(output_path, frame)
    print(f"\nSaved the processed image on desktop as:\n{output_path}")

if __name__ == '__main__':
    main()