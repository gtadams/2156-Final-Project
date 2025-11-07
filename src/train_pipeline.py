"""
Main Training Pipeline
Complete pipeline for training STP file classifier from CAD models.
"""

import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import json

from stp_loader import STPLoader
from renderer import Renderer
from trainer import STAPClassifier


def generate_sample_backgrounds(num_backgrounds: int = 5, size: tuple = (224, 224)):
    """
    Generate sample background images for augmentation.
    
    Args:
        num_backgrounds: Number of backgrounds to generate
        size: Size of background images
        
    Returns:
        List of PIL Image objects
    """
    backgrounds = []
    
    # Generate varied backgrounds
    for i in range(num_backgrounds):
        # Create gradient backgrounds with different colors
        img = Image.new('RGB', size)
        pixels = img.load()
        
        # Random color scheme
        base_color = np.random.randint(0, 255, 3)
        
        for y in range(size[1]):
            for x in range(size[0]):
                # Create gradient effect
                factor = (x + y) / (size[0] + size[1])
                color = tuple(int(base_color[c] * (0.5 + 0.5 * factor)) for c in range(3))
                pixels[x, y] = color
        
        backgrounds.append(img)
    
    return backgrounds


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description='Train STP file classifier from CAD models'
    )
    
    parser.add_argument(
        '--stp-dir',
        type=str,
        required=True,
        help='Directory containing .stp files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Directory for output files (rendered images, models, etc.)'
    )
    
    parser.add_argument(
        '--num-backgrounds',
        type=int,
        default=3,
        help='Number of background variations to use'
    )
    
    parser.add_argument(
        '--num-angles',
        type=int,
        default=None,
        help='Number of view angles to use (default: all)'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='resnet18',
        choices=['resnet18', 'resnet50', 'mobilenet_v2'],
        help='Pre-trained model to use'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for training'
    )
    
    args = parser.parse_args()
    
    # Create output directories
    output_path = Path(args.output_dir)
    rendered_dir = output_path / 'rendered_images'
    models_dir = output_path / 'models'
    
    rendered_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("STP File Classifier Training Pipeline")
    print("="*80)
    
    # Step 1: Load STP files
    print("\n[Step 1/4] Loading STP files...")
    loader = STPLoader(args.stp_dir)
    
    try:
        model_names = loader.scan_directory()
        print(f"Found {len(model_names)} STP files: {model_names}")
        
        if len(model_names) == 0:
            print("Error: No STP files found!")
            print("Please ensure your directory contains .stp or .step files")
            return
        
        # Load all models
        loader.load_all()
        print(f"Successfully loaded {len(loader.loaded_models)} models")
        
    except Exception as e:
        print(f"Error loading STP files: {e}")
        print("\nNote: If you don't have STP files yet, you can create sample data")
        print("or use the pipeline in demo mode with synthetic images.")
        return
    
    # Step 2: Generate rendered images
    print("\n[Step 2/4] Generating rendered images...")
    renderer = Renderer(image_size=(224, 224))
    
    # Generate sample backgrounds
    backgrounds = generate_sample_backgrounds(args.num_backgrounds)
    print(f"Generated {len(backgrounds)} background variations")
    
    filters = ['none', 'blur', 'sharpen', 'brightness_up', 'contrast_up']
    print(f"Using filters: {filters}")
    
    # Render all models
    all_generated_files = []
    for model_name in model_names:
        print(f"\nRendering {model_name}...")
        assembly = loader.get_model(model_name)
        
        generated_files = renderer.generate_augmented_images(
            assembly=assembly,
            model_name=model_name,
            output_dir=str(rendered_dir),
            backgrounds=backgrounds,
            filters=filters,
            num_angles=args.num_angles
        )
        
        all_generated_files.extend(generated_files)
        print(f"  Generated {len(generated_files)} images")
    
    print(f"\nTotal images generated: {len(all_generated_files)}")
    
    # Step 3: Train classifier
    print("\n[Step 3/4] Training classifier...")
    
    classifier = STAPClassifier(
        num_classes=len(model_names),
        model_name=args.model_name,
        pretrained=True
    )
    
    # Prepare data
    print("Preparing data loaders...")
    train_loader, val_loader = classifier.prepare_data(
        image_dir=str(rendered_dir),
        class_names=model_names,
        train_split=0.8
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Train model
    model_save_path = models_dir / f"{args.model_name}_stp_classifier.pth"
    
    print(f"\nStarting training for {args.epochs} epochs...")
    history = classifier.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_path=str(model_save_path)
    )
    
    # Save training history
    history_path = models_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining history saved to {history_path}")
    
    # Step 4: Summary
    print("\n[Step 4/4] Training complete!")
    print("="*80)
    print("\nSummary:")
    print(f"  STP files processed: {len(model_names)}")
    print(f"  Images generated: {len(all_generated_files)}")
    print(f"  Model saved to: {model_save_path}")
    print(f"  Final validation accuracy: {history['val_acc'][-1]:.2f}%")
    
    print("\nTo classify new images, run:")
    print(f"  python src/classifier.py --model {model_save_path} --image <path_to_image>")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
