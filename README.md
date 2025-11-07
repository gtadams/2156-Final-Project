# 2156-Final-Project
2.156 Final Project, Grayson/Ryan/Becca

## STP File Classification using Deep Learning

This project implements a machine learning pipeline for classifying images based on 3D CAD models stored in .stp (STEP) files. It uses PyTorch and pre-trained deep learning models to learn visual features from rendered 3D models and classify new images.

### Features

- **STP File Loading**: Import and manage .stp (STEP) CAD files
- **3D Rendering**: Generate photorealistic renderings using CadQuery
- **Data Augmentation**: Create training data with multiple view angles, backgrounds, and filters
- **Pre-trained Models**: Leverage pre-trained ResNet/MobileNet models from torchvision
- **Unsupervised Learning**: Train classifiers using PyTorch
- **Classification Tool**: Classify new images against trained models

### Project Structure

```
2156-Final-Project/
├── src/
│   ├── stp_loader.py       # Load and manage .stp files
│   ├── renderer.py         # Render 3D models with augmentations
│   ├── trainer.py          # PyTorch training pipeline
│   ├── classifier.py       # Image classification tool
│   └── train_pipeline.py   # Complete training pipeline
├── examples/
│   └── demo.py            # Demo with synthetic data
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/gtadams/2156-Final-Project.git
cd 2156-Final-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Option 1: Full Pipeline with Real STP Files

If you have .stp files to work with:

```bash
python src/train_pipeline.py \
    --stp-dir /path/to/stp/files \
    --output-dir ./output \
    --model-name resnet18 \
    --epochs 10
```

This will:
1. Load all .stp files from the specified directory
2. Generate multiple rendered images with different angles, backgrounds, and filters
3. Train a classifier on the generated images
4. Save the trained model for later use

#### Option 2: Demo with Synthetic Data

To test the system without .stp files:

```bash
python examples/demo.py
```

This creates synthetic training data and demonstrates the complete pipeline.

#### Classifying New Images

Once you have a trained model, classify images:

```bash
# Classify a single image
python src/classifier.py \
    --model ./output/models/resnet18_stp_classifier.pth \
    --image /path/to/image.png

# Classify all images in a directory
python src/classifier.py \
    --model ./output/models/resnet18_stp_classifier.pth \
    --directory /path/to/images \
    --output results.json
```

### Components

#### 1. STP Loader (`stp_loader.py`)
Handles importing .stp (STEP) CAD files using CadQuery:
- Scans directories for .stp/.step files
- Loads CAD models into memory
- Manages multiple models

#### 2. Renderer (`renderer.py`)
Generates photorealistic renderings with augmentation:
- Multiple view angles (orthogonal, isometric, custom)
- Background image composition
- Image filters (blur, sharpen, brightness, contrast, saturation)
- Batch rendering with all permutations

#### 3. Trainer (`trainer.py`)
PyTorch-based training pipeline:
- Loads pre-trained models (ResNet18, ResNet50, MobileNet V2)
- Fine-tunes on STP-rendered images
- Implements data augmentation
- Tracks training metrics
- Saves best model checkpoints

#### 4. Classifier (`classifier.py`)
Command-line tool for inference:
- Loads trained models
- Classifies single images or batches
- Outputs predictions with confidence scores
- Saves results to JSON

#### 5. Training Pipeline (`train_pipeline.py`)
Complete end-to-end pipeline:
- Orchestrates all components
- Configurable via command-line arguments
- Generates training reports

### Technical Details

**Machine Learning Model:**
- Pre-trained on ImageNet
- Fine-tuned for STP file classification
- Options: ResNet18, ResNet50, MobileNet V2

**Data Augmentation:**
- 12+ view angles per model
- 3-5 background variations
- 5+ image filters
- Results in 180-300 images per STP file

**Training:**
- Uses Adam optimizer
- Learning rate scheduling
- Cross-entropy loss
- 80/20 train/validation split

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CadQuery 2.3+
- Pillow, NumPy, OpenCV
- See `requirements.txt` for complete list

### Example Workflow

1. **Prepare Data**: Place .stp files in a directory
2. **Train Model**: Run training pipeline
3. **Classify Images**: Use trained model to classify new images
4. **Evaluate**: Review classification results and confidence scores

### Limitations & Future Work

- CadQuery rendering capabilities are limited; integration with VTK or Blender would improve photorealism
- Currently implements supervised learning; true unsupervised learning features could be added
- Real-time rendering could be optimized
- Support for additional CAD formats (IGES, BREP)

### License

See LICENSE file for details.

### Contributors

Grayson, Ryan, Becca
