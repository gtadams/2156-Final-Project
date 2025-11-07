"""
Classification Tool
Main interface for classifying images against trained STP models.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict
import torch
from PIL import Image

from trainer import STAPClassifier


class ImageClassifier:
    """Tool for classifying images against STP file models."""
    
    def __init__(self, model_path: str):
        """
        Initialize classifier with a trained model.
        
        Args:
            model_path: Path to saved model weights
        """
        self.model_path = model_path
        
        # Load model metadata
        checkpoint = torch.load(model_path, map_location='cpu')
        num_classes = checkpoint['num_classes']
        model_name = checkpoint.get('model_name', 'resnet50')
        
        # Initialize classifier
        self.classifier = STAPClassifier(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=False
        )
        
        # Load trained weights
        self.classifier.load_model(model_path)
        
        print(f"Loaded model from {model_path}")
        print(f"Model type: {model_name}")
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {self.classifier.class_names}")
    
    def classify_image(self, image_path: str, top_k: int = 3) -> List[Dict]:
        """
        Classify a single image.
        
        Args:
            image_path: Path to image to classify
            top_k: Number of top predictions to return
            
        Returns:
            List of dictionaries with predictions
        """
        # Get prediction
        predicted_idx, confidence, class_name = self.classifier.predict(image_path)
        
        # For top-k, we'd need to modify the predict method
        # For now, return single prediction
        results = [{
            'class_index': predicted_idx,
            'class_name': class_name,
            'confidence': confidence
        }]
        
        return results
    
    def classify_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Classify multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of classification results
        """
        results = []
        
        for image_path in image_paths:
            try:
                prediction = self.classify_image(image_path)
                results.append({
                    'image_path': image_path,
                    'predictions': prediction
                })
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def classify_directory(self, directory: str, output_file: Optional[str] = None) -> List[Dict]:
        """
        Classify all images in a directory.
        
        Args:
            directory: Directory containing images
            output_file: Optional JSON file to save results
            
        Returns:
            List of classification results
        """
        dir_path = Path(directory)
        
        # Find all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(dir_path.glob(ext))
        
        image_paths = [str(p) for p in image_paths]
        
        print(f"Found {len(image_paths)} images in {directory}")
        
        # Classify all images
        results = self.classify_batch(image_paths)
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return results


def main():
    """Command-line interface for the classifier."""
    parser = argparse.ArgumentParser(
        description='Classify images against trained STP file models'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to single image to classify'
    )
    
    parser.add_argument(
        '--directory',
        type=str,
        help='Path to directory of images to classify'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to output JSON file for results'
    )
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = ImageClassifier(args.model)
    
    # Classify based on input
    if args.image:
        print(f"\nClassifying image: {args.image}")
        results = classifier.classify_image(args.image)
        
        print("\nResults:")
        for result in results:
            print(f"  Class: {result['class_name']}")
            print(f"  Confidence: {result['confidence']:.4f}")
    
    elif args.directory:
        print(f"\nClassifying images in directory: {args.directory}")
        results = classifier.classify_directory(args.directory, args.output)
        
        print(f"\nClassified {len(results)} images")
        
        # Print summary
        if results:
            print("\nSample results:")
            for result in results[:5]:
                if 'predictions' in result:
                    pred = result['predictions'][0]
                    print(f"  {Path(result['image_path']).name}: {pred['class_name']} ({pred['confidence']:.4f})")
    
    else:
        print("Error: Please provide either --image or --directory")
        parser.print_help()


if __name__ == '__main__':
    main()
