import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.switch_backend('Agg')

# ============================================================================
# 1. Configuration & Mappings
# ============================================================================
value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 600: 6, 700: 7, 800: 8, 7100: 9, 10000: 10
}
n_classes = len(value_map)

# Color palette for visualization (RGB)
color_palette = {
    0: [0, 0, 0],         # Background: Black
    1: [34, 139, 34],     # Trees: Forest Green
    2: [0, 255, 0],       # Lush Bushes: Lime Green
    3: [218, 165, 32],    # Dry Grass: Goldenrod
    4: [139, 69, 19],     # Dry Bushes: Saddle Brown
    5: [169, 169, 169],   # Ground Clutter: Dark Gray
    6: [255, 20, 147],    # Flowers: Deep Pink
    7: [160, 82, 45],     # Logs: Sienna
    8: [105, 105, 105],   # Rocks: Dim Gray
    9: [244, 164, 96],    # Landscape: Sandy Brown
    10: [135, 206, 235]   # Sky: Sky Blue
}

def decode_segmap(image, nc=11):
    """Converts a 2D segmentation mask (0-10) into an RGB image."""
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = color_palette[l][0]
        g[idx] = color_palette[l][1]
        b[idx] = color_palette[l][2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

# ============================================================================
# 2. Dataset Loader
# ============================================================================
class DesertTestDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Directory not found: {self.image_dir}")
            
        self.transform = transform
        self.mask_transform = mask_transform
        
        self.data_ids = [f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg'))]

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # Convert raw labels
        arr = np.array(mask)
        new_arr = np.zeros_like(arr, dtype=np.uint8)
        for raw_value, new_value in value_map.items():
            new_arr[arr == raw_value] = new_value
        mask = Image.fromarray(new_arr)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = mask.squeeze(0).long()

        return image, mask, data_id

# ============================================================================
# 3. DuneSegM2 Architecture 
# ============================================================================
class CBR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class AggregationBlock(nn.Module):
    def __init__(self, in_c_lf, in_c_hf, out_c=64):
        super().__init__()
        self.align_lf = nn.Conv2d(in_c_lf, out_c, 1) if in_c_lf != out_c else nn.Identity()
        self.align_hf = nn.Conv2d(in_c_hf, out_c, 1) if in_c_hf != out_c else nn.Identity()
        self.cbr_lf = CBR(out_c, out_c)
        self.cbr_fused = CBR(out_c, out_c)

    def forward(self, lf, hf):
        lf_aligned = self.align_lf(lf)
        hf_aligned = self.align_hf(hf)
        lf_cbr = self.cbr_lf(lf_aligned)
        hf_up = F.interpolate(hf_aligned, size=lf.shape[2:], mode='bilinear', align_corners=False)
        fused = lf_cbr + hf_up
        return self.cbr_fused(fused) + fused

class DuneSegM2(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=None).features # No weights needed for testing
        self.branch4_net = mobilenet[:2]   
        self.branch3_net = mobilenet[2:4]  
        self.branch2_net = mobilenet[4:7]  
        self.branch1_net = mobilenet[7:14] 
        
        self.fusion1 = nn.Conv2d(96, 64, kernel_size=1) 
        self.agg2 = AggregationBlock(32, 64, 64)
        self.agg3 = AggregationBlock(24, 64, 64)
        self.agg4 = AggregationBlock(16, 64, 64)

        self.cls1 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.cls2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.cls3 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.cls4 = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self.decoder_conv = CBR(64, 64)
        self.final_cls = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]
        feat4 = self.branch4_net(x)      
        feat3 = self.branch3_net(feat4)  
        feat2 = self.branch2_net(feat3)  
        feat1 = self.branch1_net(feat2)  
        
        out1 = self.fusion1(feat1)
        out2 = self.agg2(feat2, out1)
        out3 = self.agg3(feat3, out2)
        out4 = self.agg4(feat4, out3)
        
        decoder_fused = self.decoder_conv(out4) 
        final_pred = F.interpolate(self.final_cls(decoder_fused), size=input_size, mode='bilinear', align_corners=False)
        return final_pred

# ============================================================================
# 4. Evaluation Metrics
# ============================================================================
def compute_iou(pred_mask, target, num_classes):
    iou_per_class = []
    for cls in range(num_classes):
        pred_inds = pred_mask == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union > 0:
            iou_per_class.append((intersection / union).item())
        else:
            iou_per_class.append(float('nan'))
    return np.nanmean(iou_per_class), iou_per_class

# ============================================================================
# 5. Main Testing Logic
# ============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on hardware: {device}")

    # Paths
    test_data_dir = "./data/desert_objects/test"
    model_path = "./outputs/duneseg_best_val.pth"
    output_dir = "./test_results"
    
    os.makedirs(output_dir, exist_ok=True)
    visuals_dir = os.path.join(output_dir, "visual_comparisons")
    os.makedirs(visuals_dir, exist_ok=True)

    h, w = 544, 960

    # Transformations
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])

    print("Loading test dataset...")
    try:
        testset = DesertTestDataset(test_data_dir, transform=transform, mask_transform=mask_transform)
        test_loader = DataLoader(testset, batch_size=1, shuffle=False)
    except FileNotFoundError as e:
        print(f"Warning: {e}. Please ensure test dataset is in {test_data_dir}")
        return

    print(f"Loading trained model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = DuneSegM2(num_classes=n_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_ious = []
    all_inf_times = []
    class_ious_list = []

    print("\n--- Starting Evaluation ---")
    with torch.no_grad():
        for i, (img, label, img_name) in enumerate(tqdm(test_loader, desc="Testing Images")):
            img, label = img.to(device), label.to(device)
            
            start_time = time.time()
            with torch.amp.autocast('cuda'):
                pred = model(img)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inf_time = (time.time() - start_time) * 1000 # in milliseconds
            all_inf_times.append(inf_time)

            # 2. Get Predictions
            pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu()
            true_mask = label.squeeze(0).cpu()

            # 3. Calculate Metrics
            m_iou, c_iou = compute_iou(pred_mask, true_mask, n_classes)
            all_ious.append(m_iou)
            class_ious_list.append(c_iou)

            if i < 20: # Save first 20 visual comparisons
                orig_img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                orig_img = std * orig_img + mean
                orig_img = np.clip(orig_img, 0, 1)

                rgb_true = decode_segmap(true_mask.numpy(), n_classes)
                rgb_pred = decode_segmap(pred_mask.numpy(), n_classes)

                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                axes[0].imshow(orig_img); axes[0].set_title("Original Image"); axes[0].axis('off')
                axes[1].imshow(rgb_true); axes[1].set_title("Ground Truth Mask"); axes[1].axis('off')
                axes[2].imshow(rgb_pred); axes[2].set_title(f"Prediction (IoU: {m_iou:.3f})"); axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(visuals_dir, f"comp_{img_name[0]}"))
                plt.close(fig)

    # Calculate final aggregates
    final_mean_iou = np.nanmean(all_ious)
    final_avg_inf_time = np.mean(all_inf_times)
    avg_class_iou = np.nanmean(class_ious_list, axis=0)

    # Save to Text Report
    report_path = os.path.join(output_dir, "test_evaluation_metrics.txt")
    with open(report_path, "w") as f:
        f.write("DESERT OBJECTS SEGMENTATION - TEST RESULTS\n")
        f.write("=" * 45 + "\n")
        f.write(f"Total Images Processed: {len(testset)}\n")
        f.write(f"Final Mean IoU:         {final_mean_iou:.4f}\n")
        f.write(f"Average Inference Time: {final_avg_inf_time:.2f} ms per image\n")
        f.write("=" * 45 + "\n\n")
        
        f.write("Per-Class IoU Breakdown:\n")
        class_names = ["Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", 
                       "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"]
        for cls_name, iou in zip(class_names, avg_class_iou):
             f.write(f"  {cls_name:<15}: {iou:.4f}\n")

    print(f"\nEvaluation Complete!")
    print(f"Final Test mIoU: {final_mean_iou:.4f}")
    print(f"Average Inference Time: {final_avg_inf_time:.2f}ms")
    print(f"Saved numerical report to: {report_path}")
    print(f"Saved visual comparisons to: {visuals_dir}")

if __name__ == "__main__":
    main()