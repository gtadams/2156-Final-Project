"""
Demo Script
Demonstrates the STP classifier with synthetic data.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from renderer import Renderer
from trainer import STAPClassifier


def create_synthetic_stp_data(output_dir: str, num_classes: int = 3):
    """
    Create synthetic images representing different STP file classes.
    This simulates what would be rendered from actual .stp files.
    
    Args:
        output_dir: Directory to save synthetic images
        num_classes: Number of different "STP file" classes
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    class_names = [f"part_{i}" for i in range(num_classes)]
    
    print(f"Creating synthetic data for {num_classes} classes...")
    
    # Create images for each class
    for class_idx, class_name in enumerate(class_names):
        # Each class will have different visual characteristics
        base_color = (
            (class_idx * 80 + 50) % 255,
            (class_idx * 120 + 100) % 255,
            (class_idx * 60 + 150) % 255
        )
        
        # Generate multiple images per class (simulating different angles/augmentations)
        for img_idx in range(30):
            img = Image.new('RGB', (224, 224), color=base_color)
            pixels = img.load()
            
            # Add some patterns to make classes distinguishable
            for y in range(224):
                for x in range(224):
                    # Add noise
                    noise = np.random.randint(-20, 20)
                    
                    # Add class-specific pattern
                    if class_idx == 0:
                        # Circles
                        dist = np.sqrt((x - 112)**2 + (y - 112)**2)
                        if 50 < dist < 70 or 80 < dist < 100:
                            noise += 50
                    elif class_idx == 1:
                        # Horizontal stripes
                        if y % 20 < 10:
                            noise += 40
                    else:
                        # Diagonal pattern
                        if (x + y) % 30 < 15:
                            noise += 40
                    
                    r = max(0, min(255, base_color[0] + noise))
                    g = max(0, min(255, base_color[1] + noise))
                    b = max(0, min(255, base_color[2] + noise))
                    
                    pixels[x, y] = (r, g, b)
            
            # Save image
            filename = f"{class_name}_synthetic_{img_idx}.png"
            img.save(output_path / filename)
    
    print(f"Created {num_classes * 30} synthetic images in {output_dir}")
    return class_names


def run_demo():
    """Run a complete demo of the STP classifier."""
    print("="*80)
    print("STP Classifier Demo with Synthetic Data")
    print("="*80)
    
    # Create demo directories
    demo_dir = Path("demo_output")
    data_dir = demo_dir / "data"
    models_dir = demo_dir / "models"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create synthetic data
    print("\n[Step 1/3] Creating synthetic training data...")
    class_names = create_synthetic_stp_data(str(data_dir), num_classes=3)
    
    # Step 2: Train classifier
    print("\n[Step 2/3] Training classifier...")
    
    classifier = STAPClassifier(
        num_classes=len(class_names),
        model_name='resnet18',
        pretrained=True
    )
    
    print("Preparing data loaders...")
    train_loader, val_loader = classifier.prepare_data(
        image_dir=str(data_dir),
        class_names=class_names,
        train_split=0.8
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    model_save_path = models_dir / "demo_model.pth"
    
    print("\nTraining (5 epochs for demo)...")
    history = classifier.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,
        learning_rate=0.001,
        save_path=str(model_save_path)
    )
    
    # Step 3: Test classifier
    print("\n[Step 3/3] Testing classifier...")
    
    # Get a few test images
    test_images = list(data_dir.glob("*.png"))[:5]
    
    print("\nTest predictions:")
    for img_path in test_images:
        pred_idx, confidence, class_name = classifier.predict(str(img_path))
        actual_class = img_path.stem.split('_')[0] + "_" + img_path.stem.split('_')[1]
        
        print(f"  {img_path.name}")
        print(f"    Predicted: {class_name} (confidence: {confidence:.4f})")
        print(f"    Actual: {actual_class}")
        print()
    
    # Summary
    print("="*80)
    print("\nDemo Complete!")
    print(f"\nFinal validation accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"Model saved to: {model_save_path}")
    print("\nNote: This demo uses synthetic data. With real .stp files,")
    print("the images would be actual 3D renderings of CAD models.")
    print("="*80)


if __name__ == '__main__':
    run_demo()
