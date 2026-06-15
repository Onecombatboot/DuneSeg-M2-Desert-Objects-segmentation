import os
# Optional: Optimization for memory management
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

# Enable Interactive Mode for Live Plotting
plt.ion() 

# ============================================================================
# 1. Dataset & Class Mapping (11 Classes)
# ============================================================================
value_map = {
    0: 0,        # Background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    600: 6,      # Flowers
    700: 7,      # Logs
    800: 8,      # Rocks
    7100: 9,     # Landscape
    10000: 10    # Sky
}
n_classes = len(value_map)

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)

class DesertSegmentationDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, mask_transform=None):
        self.data_dir = os.path.join(data_dir, split)
        self.image_dir = os.path.join(self.data_dir, 'Color_Images')
        self.masks_dir = os.path.join(self.data_dir, 'Segmentation')
        
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
        mask = convert_mask(mask)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = mask.squeeze(0).long()

        return image, mask

# ============================================================================
# 2. DuneSegM2 Architecture
# ============================================================================
class CBR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

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
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features
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
        
        pred1 = F.interpolate(self.cls1(out1), size=input_size, mode='bilinear', align_corners=False)
        pred2 = F.interpolate(self.cls2(out2), size=input_size, mode='bilinear', align_corners=False)
        pred3 = F.interpolate(self.cls3(out3), size=input_size, mode='bilinear', align_corners=False)
        pred4 = F.interpolate(self.cls4(out4), size=input_size, mode='bilinear', align_corners=False)
        
        decoder_fused = self.decoder_conv(out4) 
        final_pred = F.interpolate(self.final_cls(decoder_fused), size=input_size, mode='bilinear', align_corners=False)
        
        if self.training:
            return final_pred, pred1, pred2, pred3, pred4
        return final_pred

# ============================================================================
# 3. Detailed Evaluation Metrics
# ============================================================================
def compute_iou(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    iou_per_class = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union > 0:
            iou_per_class.append((intersection / union).item())
        else:
            iou_per_class.append(float('nan'))
    return np.nanmean(iou_per_class), iou_per_class

def compute_dice(pred, target, num_classes, smooth=1e-6):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)
        dice_per_class.append(dice_score.item())
    return np.mean(dice_per_class), dice_per_class

def compute_pixel_accuracy(pred, target):
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().item()

# ============================================================================
# 4. Report & Graph Generation Functions
# ============================================================================
def save_training_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    current_backend = plt.get_backend()
    plt.switch_backend('Agg')

    # Plot 1: Loss curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'], label='val')
    plt.title('Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

    # Plot 2: IoU curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.title('Train IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'], label='Val IoU')
    plt.title('Validation IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curves.png'))
    plt.close()

    # Plot 3: Dice curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.title('Train Dice vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_dice'], label='Val Dice')
    plt.title('Validation Dice vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_curves.png'))
    plt.close()

    # Plot 4: Combined metrics plot
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss vs Epoch')
    plt.legend(); plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history['train_iou'], label='train')
    plt.plot(history['val_iou'], label='val')
    plt.title('IoU vs Epoch')
    plt.legend(); plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history['train_dice'], label='train')
    plt.plot(history['val_dice'], label='val')
    plt.title('Dice Score vs Epoch')
    plt.legend(); plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'], label='val')
    plt.title('Pixel Accuracy vs Epoch')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'))
    plt.close()
    
    plt.switch_backend(current_backend)

def save_history_to_file(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')

    with open(filepath, 'w') as f:
        f.write("TRAINING RESULTS (DuneSegM2)\n")
        f.write("=" * 50 + "\n\n")

        f.write("Final Metrics:\n")
        f.write(f"  Final Train Loss:     {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss:       {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Train IoU:      {history['train_iou'][-1]:.4f}\n")
        f.write(f"  Final Val IoU:        {history['val_iou'][-1]:.4f}\n")
        f.write(f"  Final Train Dice:     {history['train_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Dice:       {history['val_dice'][-1]:.4f}\n")
        f.write(f"  Final Train Accuracy: {history['train_pixel_acc'][-1]:.4f}\n")
        f.write(f"  Final Val Accuracy:   {history['val_pixel_acc'][-1]:.4f}\n")
        f.write(f"  Final Inference Time: {history['inference_time'][-1]:.2f} ms\n")
        f.write("=" * 50 + "\n\n")

        f.write("Best Results:\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Epoch History:\n")
        f.write("-" * 110 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU', 'Train Dice', 'Val Dice', 'Train Acc', 'Val Acc', 'Inf_ms']
        f.write("{:<6} {:<11} {:<11} {:<11} {:<11} {:<11} {:<11} {:<11} {:<11} {:<11}\n".format(*headers))
        f.write("-" * 110 + "\n")

        n_epochs = len(history['train_loss'])
        for i in range(n_epochs):
            f.write("{:<6} {:<11.4f} {:<11.4f} {:<11.4f} {:<11.4f} {:<11.4f} {:<11.4f} {:<11.4f} {:<11.4f} {:<11.2f}\n".format(
                i + 1,
                history['train_loss'][i], history['val_loss'][i],
                history['train_iou'][i], history['val_iou'][i],
                history['train_dice'][i], history['val_dice'][i],
                history['train_pixel_acc'][i], history['val_pixel_acc'][i],
                history['inference_time'][i]
            ))

# ============================================================================
# 5. Main Training Loop
# ============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware utilized: {device}")

    base_data_dir = "./data/desert_objects"
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)

    batch_size = 6 
    lr = 1e-3
    n_epochs = 20
    h, w = 540, 960 

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])

    print("Loading datasets...")
    # These paths are placeholders, user should point to their dataset
    try:
        trainset = DesertSegmentationDataset(base_data_dir, split='train', transform=transform, mask_transform=mask_transform)
        valset = DesertSegmentationDataset(base_data_dir, split='val', transform=transform, mask_transform=mask_transform)

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    except FileNotFoundError as e:
        print(f"Warning: {e}. Please ensure dataset is in {base_data_dir}")
        return

    model = DuneSegM2(num_classes=n_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    weights = torch.tensor([1.0, 2.0, 2.0, 1.5, 2.0, 4.0, 8.0, 5.0, 3.0, 0.5, 0.5]).to(device)
    loss_fct = nn.CrossEntropyLoss(weight=weights)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # Full Metric History Dictionary
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_iou': [], 'val_iou': [],
        'train_dice': [], 'val_dice': [],
        'train_pixel_acc': [], 'val_pixel_acc': [],
        'inference_time': []
    }

    k_b_prev = [None] * 4 
    best_train_loss = float('inf')
    best_val_loss = float('inf')

    # --- Setup Live Plot Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plt.show(block=False)

    print("\n--- Starting DuneSegM2 Training Phase ---")
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        train_ious, train_dices, train_accs = [], [], []

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]")
        for imgs, labels in train_pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                final_pred, p1, p2, p3, p4 = model(imgs)
                l_final = loss_fct(final_pred, labels)
                L_b = [loss_fct(p1, labels), loss_fct(p2, labels), loss_fct(p3, labels), loss_fct(p4, labels)]

                # Multi-Loss Balancing Algorithm
                if k_b_prev[0] is None:
                    k_b_prev = [l.item() for l in L_b]
                    lambda_b = [0.25] * 4
                else:
                    alpha = [L_b[i].item() / (L_b[i].item() + k_b_prev[i] + 1e-8) for i in range(4)]
                    k_b_curr = [(1 - alpha[i]) * k_b_prev[i] + alpha[i] * L_b[i].item() for i in range(4)]
                    r_b = [L_b[i].item() / (k_b_prev[i] + 1e-8) for i in range(4)]
                    sum_r = sum(r_b)
                    lambda_b = [(sum_r - r) / (sum_r + 1e-8) for r in r_b]
                    k_b_prev = k_b_curr

                balanced_loss = sum(lambda_b[i] * L_b[i] for i in range(4))
                total_loss = l_final + balanced_loss 

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += l_final.item()

            # Record Train Metrics on the fly
            with torch.no_grad():
                t_iou, _ = compute_iou(final_pred, labels, n_classes)
                t_dice, _ = compute_dice(final_pred, labels, n_classes)
                t_acc = compute_pixel_accuracy(final_pred, labels)
                train_ious.append(t_iou); train_dices.append(t_dice); train_accs.append(t_acc)

            train_pbar.set_postfix(loss=f"{total_loss.item():.4f}")

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_ious, val_dices, val_accs = [], [], []
        inf_times = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                start_time = time.time()
                with torch.amp.autocast('cuda'):
                    final_pred = model(imgs) 
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                inf_times.append((time.time() - start_time) / imgs.shape[0])

                loss = loss_fct(final_pred, labels)
                val_loss += loss.item()

                # Detailed Validation Metrics
                v_iou, _ = compute_iou(final_pred, labels, n_classes)
                v_dice, _ = compute_dice(final_pred, labels, n_classes)
                v_acc = compute_pixel_accuracy(final_pred, labels)

                val_ious.append(v_iou)
                val_dices.append(v_dice)
                val_accs.append(v_acc)

        # --- Update Full History ---
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_iou'].append(np.mean(train_ious))
        history['val_iou'].append(np.mean(val_ious))
        history['train_dice'].append(np.mean(train_dices))
        history['val_dice'].append(np.mean(val_dices))
        history['train_pixel_acc'].append(np.mean(train_accs))
        history['val_pixel_acc'].append(np.mean(val_accs))
        history['inference_time'].append(np.mean(inf_times) * 1000)

        print(f"Val Loss: {history['val_loss'][-1]:.4f} | Val IoU: {history['val_iou'][-1]:.4f} | Val Acc: {history['val_pixel_acc'][-1]:.4f}")
        scheduler.step(history['val_iou'][-1])

        # --- Live Plot Update ---
        axes[0].clear()
        axes[0].plot(history['train_loss'], label='Train Loss', color='blue', marker='o')
        axes[0].plot(history['val_loss'], label='Val Loss', color='red', marker='o')
        axes[0].set_title('Training & Validation Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].clear()
        axes[1].plot(history['train_iou'], label='Train IoU', color='orange', marker='o', alpha=0.5)
        axes[1].plot(history['val_iou'], label='Val IoU', color='green', marker='o')
        axes[1].set_title('Mean IoU Progression')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('mIoU Score')
        axes[1].legend()
        axes[1].grid(True)

        plt.draw()
        plt.pause(0.1)

        current_train_loss = history['train_loss'][-1]
        current_val_loss = history['val_loss'][-1]

        if current_train_loss < best_train_loss:
            best_train_loss = current_train_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'duneseg_best_train.pth'))
            print(f"--> [Model Checkpoint] Saved new Best Training Model (Loss: {best_train_loss:.4f})")
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'duneseg_best_val.pth'))
            print(f"--> [Model Checkpoint] Saved new Best Validation Model (Loss: {best_val_loss:.4f})")

    # ============================================================================
    #                       6. Generate Static Files
    # ============================================================================
    print("\nTraining Complete! Generating detailed static graphs and text summaries...")
    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir)
    print(f"All logs, static .png graphs, and the .txt summary are saved in: {output_dir}/")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()