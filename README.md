# DuneSeg-M2: Desert Objects Segmentation

DuneSeg-M2 is a high-intensity semantic segmentation model designed for identifying and isolating 11 distinct desert objects. The architecture integrates two heavy models with a custom aggregation block to achieve high accuracy in challenging off-road and desert environments.

## Features

- **Multi-Scale Architecture**: Utilizes a MobileNetV2 backbone with custom fusion and aggregation layers.
- **11-Class Segmentation**: Specifically tuned for desert environments:
  - Background, Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Flowers, Logs, Rocks, Landscape, and Sky.
- **Adaptive Loss Balancing**: Implements a multi-loss balancing algorithm to optimize training across multiple prediction heads.
- **Inference Ready**: Optimized for real-time performance on modern GPU hardware.

## Model Architecture

DuneSeg-M2 uses a specialized branching strategy:
1. **Backbone**: MobileNetV2 features for efficient feature extraction.
2. **Aggregation Blocks**: Custom modules that align and fuse low-frequency and high-frequency features.
3. **Multi-Head Prediction**: Four auxiliary classification heads in addition to the final decoder output for stable training.

## Installation

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (recommended)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Onecombatboot/DuneSeg-M2-Desert-Objects-segmentation.git
   cd DuneSeg-M2-Desert-Objects-segmentation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
To start training, place your dataset in the `./data/desert_objects` directory and run:
```bash
python train.py
```

### Testing
To evaluate the model and generate visual comparisons:
```bash
python test.py
```

## Dataset Structure
The project expects the following directory structure:
```
data/
└── desert_objects/
    ├── train/
    │   ├── Color_Images/
    │   └── Segmentation/
    └── val/
        ├── Color_Images/
        └── Segmentation/
```

## Metrics
The project includes tools for generating:
- Mean IoU (mIoU)
- Dice Score
- Pixel Accuracy
- Inference Time Analysis
- Visual comparison plots

## License
MIT License
