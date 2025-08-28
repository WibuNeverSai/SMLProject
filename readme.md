### 1. Install dependencies
!pip install ultralytics opencv-python pillow torch torchvision numpy matplotlib seaborn pandas PyYAML tqdm psutil requests


### 2. Import thư viện
from ultralytics import YOLO
import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import cv2
import numpy as np

print("All libraries imported successfully!")


### 3. Check Dataset
def check_dataset():
    """Kiểm tra cấu trúc dataset"""
    print("Dataset Structure:")
    print("=" * 50)
    
    # Check if data.yaml exists
    if os.path.exists('/kaggle/input/objectdetectingdatasets/datasets/data.yaml'):
        with open('/kaggle/input/objectdetectingdatasets/datasets/data.yaml', 'r') as f:
            data_config = yaml.safe_load(f)
        
        print(f"Data config found:")
        print(f"   - Train: {data_config.get('train', 'Not found')}")
        print(f"   - Val: {data_config.get('val', 'Not found')}")
        print(f"   - Test: {data_config.get('test', 'Not found')}")
        print(f"   - Classes: {data_config.get('nc', 'Not found')}")
        print(f"   - Names: {data_config.get('names', 'Not found')}")
    else:
        print("data.yaml not found!")
    
    print()
    
    # Check image counts
    splits = ['train', 'valid', 'test']
    for split in splits:
        img_path = f'/kaggle/input/objectdetectingdatasets/datasets/{split}/images'
        label_path = f'/kaggle/input/objectdetectingdatasets/datasets/{split}/labels'
        
        if os.path.exists(img_path):
            img_count = len([f for f in os.listdir(img_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"{split.capitalize()} images: {img_count}")
        else:
            print(f"{split.capitalize()} images folder not found")
            
        if os.path.exists(label_path):
            label_count = len([f for f in os.listdir(label_path) if f.endswith('.txt')])
            print(f"{split.capitalize()} labels: {label_count}")
        else:
            print(f"{split.capitalize()} labels folder not found")
        print()

# Chạy function kiểm tra
check_dataset()


### 4. Check samples
def visualize_samples():
    """Hiển thị một số ảnh mẫu từ dataset"""
    splits = ['train', 'valid', 'test']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Sample Images from Dataset', fontsize=16)
    
    for i, split in enumerate(splits):
        img_path = f'/kaggle/input/objectdetectingdatasets/datasets/{split}/images'
        if os.path.exists(img_path):
            images = [f for f in os.listdir(img_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                # Show 3 sample images
                for j in range(3):
                    if j < len(images):
                        img_file = os.path.join(img_path, images[j])
                        img = cv2.imread(img_file)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        axes[i, j].imshow(img)
                        axes[i, j].set_title(f'{split.capitalize()} - {images[j][:20]}...')
                        axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

# Chạy function hiển thị ảnh
visualize_samples()


### 5. Train model
model = YOLO('yolov8n.pt')  # Load YOLOv8 nano model

print("Starting YOLO training...")
print("Model loaded: YOLOv8n")
print(f"Current working directory: {os.getcwd()}")

# Start training
results = model.train(
        # === DATA CONFIG ===
        data='/kaggle/input/objectdetectingdatasets/datasets/data.yaml',                   # Path to optimized data.yaml
        epochs=150,                                        # Tăng epochs để train kỹ hơn
        imgsz=832,                                        # Tăng image size để detect chi tiết hơn
        batch=8,                                          # Giảm batch size để tránh OOM
        
        # === DEVICE & PERFORMANCE ===
        device=0,                                         # GPU
        workers=2,                                        # Giảm workers để tránh lỗi Kaggle
        cache=False,                                      # Tắt cache để tiết kiệm RAM
        
        # === SAVING & LOGGING ===
        project='/kaggle/working/runs/train',             # Save results
        name='detailed_potato_detection',                 # Experiment name
        exist_ok=True,                                    # Overwrite existing
        save_period=15,                                   # Save mỗi 15 epochs
        
        # === OPTIMIZATION ===
        pretrained=True,                                  # Use pretrained weights
        optimizer='AdamW',                                # Dùng AdamW thay vì SGD
        amp=True,                                         # Mixed precision
        
        # === LEARNING RATE ===
        lr0=0.001,                                       # Giảm learning rate ban đầu
        lrf=0.01,                                        # Final learning rate
        momentum=0.95,                                    # Tăng momentum
        
        # === REGULARIZATION ===
        weight_decay=0.001,                               # Tăng weight decay
        warmup_epochs=5.0,                                # Tăng warmup epochs
        
        # === LOSS WEIGHTS ===
        box=8.0,                                         # Tăng box loss để detect chính xác hơn
        cls=0.3,                                         # Giảm class loss
        dfl=1.5,                                         # Giữ nguyên DFL loss
        
        # === AUGMENTATION MẠNH ===
        hsv_h=0.015,                                     # Hue augmentation
        hsv_s=0.8,                                       # Saturation augmentation  
        hsv_v=0.5,                                       # Value augmentation
        degrees=15.0,                                     # Xoay ảnh ±15 độ
        translate=0.2,                                    # Dịch chuyển
        scale=0.6,                                        # Thay đổi kích thước
        shear=5.0,                                        # Shear transformation
        perspective=0.001,                                # Perspective transformation
        flipud=0.0,                                       # Không lật dọc (potato nằm ngang)
        fliplr=0.5,                                       # Lật ngang
        mosaic=1.0,                                       # Mosaic augmentation
        mixup=0.1,                                        # Mixup augmentation
        copy_paste=0.1,                                   # Copy-paste augmentation
        
        # === ADVANCED AUGMENTATION ===
        auto_augment='randaugment',                       # Auto augmentation
        erasing=0.2,                                      # Random erasing
        crop_fraction=0.9,                                # Crop augmentation
        
        # === TRAINING CONTROL ===
        patience=25,                                      # Tăng patience
        close_mosaic=15,                                  # Tắt mosaic cuối 15 epochs
        deterministic=True,                               # Deterministic training
        seed=42,                                          # Random seed
        
        # === VERBOSE ===
        verbose=True,                                     # Hiển thị chi tiết
        fraction=1.0,                                     # Dùng toàn bộ data
)

print("Training completed successfully!")


