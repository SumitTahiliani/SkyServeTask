# EuroSat Land Use Land Cover Classification

**Original Repository**: [https://github.com/Rumeysakeskin/EuroSat-Satellite-CNN-and-ResNet/tree/main](https://github.com/Rumeysakeskin/EuroSat-Satellite-CNN-and-ResNet/tree/main)

**Dataset**: [https://github.com/phelber/eurosat](https://github.com/phelber/eurosat)

**Note**: The train_cnn.ipynb as well train_resnet.ipynb are from the original repo (retained here for ease of comparison). The work done by me can be found in training.ipynb

## Dataset Information

- **Total Images**: 27,000 satellite images
- **Classes**: 10 land use categories
- **Image Size**: 64x64 pixels
- **Spectral Bands**: 13 bands from Sentinel-2 satellite (B01-B12 + B8A)
- **Classes**: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake

The dataset is well-balanced with around 2,700 images per class (no major imbalance), which makes it good for training without worrying about class imbalance issues.

## Baseline Implementation Analysis

I started by looking at what the baseline code was doing to understand where improvements could be made.

### Current Baseline Models

The baseline had two models:

1. **Custom CNN (80.6% accuracy)**
   - Simple 2-layer network: Conv2d(3→4) → BatchNorm → ReLU → MaxPool → Conv2d(4→8) → BatchNorm → ReLU → MaxPool → FC layers
   - Pretty basic architecture that doesn't capture complex patterns well

2. **Custom ResNet (93.1% accuracy)**
   - ResNet-8 with 2 residual blocks per layer
   - Better than the CNN but still a simplified version
   - Has residual connections but the architecture is too small for complex multi-spectral data

### Baseline Drawbacks and Limitations

I found several issues with the baseline that were limiting performance:

1. **Only using RGB channels**
   - The baseline only uses 3 RGB channels out of 13 available spectral bands
   - This misses important information like NIR (B08) which is crucial for vegetation detection
   - SWIR bands (B11, B12) are also important for soil and water analysis

2. **Poor data preprocessing**
   - Uses ImageNet normalization values which don't work well for satellite data
   - No data augmentation, so the model doesn't see enough variations
   - Satellite data has different intensity ranges than natural images

3. **Basic training approach**
   - Uses SGD with fixed learning rate
   - No learning rate scheduling or early stopping
   - No gradient clipping or other modern techniques

4. **No transfer learning**
   - Both models start from scratch instead of using pre-trained weights
   - This means they have to learn basic features that could be transferred from ImageNet
   - Takes longer to train and achieves lower performance

5. **Limited evaluation**
   - Only tracks accuracy, missing F1-score and other important metrics
   - No per-class analysis to see which land cover types are hard to classify

## Improvement Plan Overview

Based on the baseline issues I found, I planned improvements in these areas:

1. **Exploratory Data Analysis (EDA)**
   - Analyze all 13 spectral bands to understand their characteristics
   - Figure out which bands are most important for classification
   - Understand the data distribution for proper preprocessing

2. **Better Data Preprocessing**
   - Calculate proper normalization values from the actual data
   - Select the most useful 4-5 spectral bands
   - Add data augmentation to improve generalization

3. **Modern Architectures**
   - Use EfficientNet-B0 and ResNet-18 with pre-trained weights
   - Modify the input layers to handle multi-spectral data
   - Leverage transfer learning from ImageNet

4. **Advanced Training**
   - Use AdamW optimizer with weight decay
   - Add learning rate scheduling and early stopping
   - Include gradient clipping for stability

5. **Better Evaluation**
   - Track F1-score in addition to accuracy
   - Analyze per-class performance
   - Add comprehensive logging

## Exploratory Data Analysis (EDA)

I performed EDA to understand the dataset characteristics and select optimal band combinations for the classification task.

### Key Findings

- **Dataset Structure**: 27,000 images, almost 2,700 per class (no major class imbalance), 64x64 pixel multi-spectral TIFF files
- **Band Selection**: Chose B02 (Blue), B03 (Green), B04 (Red), B08 (NIR). RGB gives us regular images, NIR helps distinguish vegetation.
- **Data Distribution**: Pixel values range 0-4000, mostly clustered around 1400-1600, used for proper normalization
- **Visualization**: Created RGB and false-color composites to validate band selection and understand spectral signatures

## Training Pipeline Improvements

Based on my EDA findings, I implemented several improvements to the training pipeline to address the baseline limitations.

### 1. Data Preprocessing Enhancements

**Custom Dataset Class**
The baseline used ImageFolder which is designed for RGB images. I created a custom `EuroSATDataset` class that:
- Handles multi-spectral TIFF files properly
- Implements band selection
- Provides flexible normalization options 
- Works well with the augmentation pipeline

**Better Normalization**
Instead of using ImageNet normalization values, I:
- Calculated actual mean and standard deviation from the training data
- Applied [0, 4000] clipping to handle outliers
- Made sure each band is properly normalized

I did not use normalization during training since taking means of the dataset was leading to error accumulation over 20k images

**Data Augmentation**
The baseline had no augmentation, so I added:

```python
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),           # Different viewing angles
    A.VerticalFlip(p=0.3),             # Satellite orientation variations
    A.RandomRotate90(p=0.5),           # Different satellite passes
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),  # Geometric variations
    A.ElasticTransform(alpha=10, sigma=5, p=0.2),  # Atmospheric distortion
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),   # Sensor noise
    ToTensorV2()
])
```

**Why this matters**: Proper preprocessing and augmentation help the model generalize better and handle real-world variations in satellite data.

My background in robotics and working with object detection using YOLO helped me in this step since I could figure out what augmentations to apply based on past experience

### 2. Architecture Improvements

**Model Selection**
I chose EfficientNet-B0 and ResNet-18 for their proven performance and transfer learning capabilities.

**Key Modifications**
The main change was adapting the input layer for multi-spectral data:

```python
def get_efficientnet_model(num_classes, in_channels=4):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Modify first layer for multi-spectral input
    original_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=(original_conv.bias is not None)
    )
    
    # Initialize properly
    nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
    if new_conv.bias is not None:
        nn.init.constant_(new_conv.bias, 0)
    
    model.features[0][0] = new_conv
    return model
```

**Why kaiming_normal:**
`kaiming_normal_` is designed specifically for layers that use ReLU-like non-linearities. It helps preserve the variance of activations across layers, which:
- Avoids vanishing/exploding gradients
- Encourages healthy weight scaling early in training
- Scales Weight Variance Appropriately
- `kaiming_normal_` samples weights from a normal distribution with variance scaled by the number of input units (fan-in), so each output feature gets a stable gradient flow regardless of input size.
**Benefits**
- Preserves transfer learning benefits from ImageNet pre-training
- Proper initialization ensures good gradient flow
- Maintains architecture consistency while adapting to multi-spectral input

### 3. Training Strategy Enhancements

**Optimizer**: AdamW with weight decay (1e-5) for better convergence and regularization

**Learning Rate Scheduling**: ReduceLROnPlateau with factor 0.5, patience 5 epochs

**Early Stopping**: Patience of 8 epochs, saves best model based on validation F1-score

**Gradient Clipping**: Maximum gradient norm of 1.0 for training stability

### 4. Evaluation Metrics

**Metrics**: Track both accuracy and F1-score for comprehensive performance assessment

**Logging**: Comprehensive training logs with detailed metrics for analysis

## Results and Performance

### Baseline vs Improved Performance

| Model | Test Accuracy | F1-Score | Characteristics |
|-------|---------------|----------|----------------|
| Baseline CNN | 80.6% | N/A | Custom architecture, RGB only |
| Baseline ResNet | 93.1% | N/A | Custom ResNet, RGB only |
| **Improved EfficientNet (4-ch)** | **97.58%** | **97.47%** | Pre-trained, multi-spectral |
| **Improved EfficientNet (5-ch)** | **97.14%** | **97.04%** | Pre-trained, multi-spectral |

### Key Improvements

- **Accuracy**: +4.48% improvement over baseline ResNet
- **Multi-spectral**: Effective use of 4-5 spectral bands instead of RGB only
- **Transfer Learning**: Leveraged pre-trained weights from ImageNet
- **Advanced Training**: AdamW optimizer, learning rate scheduling, early stopping
- **Data Augmentation**: Comprehensive augmentation pipeline for better generalization

## Technical Implementation Details

### Training Configuration

**Hyperparameters**:
- Batch Size: 32
- Learning Rate: 1e-4
- Weight Decay: 1e-5
- Epochs: 50 (with early stopping)
- Optimizer: AdamW
- Loss Function: CrossEntropyLoss

**Data Split**:
- Training: 70% (18,900 images)
- Validation: 15% (4,050 images)
- Test: 15% (4,050 images)

## Files and Structure

### Core Files
- `training.ipynb`: Main implementation with EDA and training

### Model Checkpoints
- `efficientnet_in4_20-54.pth`: 4-channel EfficientNet model(RGB+NIR)
- `efficientnet_in5_20-10.pth`: 5-channel EfficientNet model(RGB+NIR+SWIR)
- `resnet_best_model.pth`: Improved ResNet model (initial attempt to configure pipeline and get everything working)

### Logs and Outputs
- `training_log_efficientnet_in4_20-54.txt`: Training logs for 4-channel model
- `training_log_efficientnet_in5_20-10.txt`: Training logs for 5-channel model

### Dataset
- `EuroSAT_MS/`: Multi-spectral satellite imagery dataset 