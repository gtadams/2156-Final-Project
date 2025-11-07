"""
Rendering Module
Generates photorealistic renderings of CAD models with various augmentations.
"""

import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Tuple, Optional, Dict
import cadquery as cq
from pathlib import Path
import io


class Renderer:
    """Class for rendering CAD models with various augmentations."""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the renderer.
        
        Args:
            image_size: Target size for rendered images (width, height)
        """
        self.image_size = image_size
        self.view_angles = self._generate_view_angles()
        
    def _generate_view_angles(self) -> List[Dict[str, float]]:
        """
        Generate a set of camera view angles.
        
        Returns:
            List of view angle configurations
        """
        angles = []
        
        # Standard orthogonal views
        angles.extend([
            {'rx': 0, 'ry': 0, 'rz': 0},      # Front
            {'rx': 0, 'ry': 90, 'rz': 0},     # Right
            {'rx': 0, 'ry': 180, 'rz': 0},    # Back
            {'rx': 0, 'ry': 270, 'rz': 0},    # Left
            {'rx': 90, 'ry': 0, 'rz': 0},     # Top
            {'rx': -90, 'ry': 0, 'rz': 0},    # Bottom
        ])
        
        # Isometric and angled views
        angles.extend([
            {'rx': 30, 'ry': 45, 'rz': 0},
            {'rx': 30, 'ry': 135, 'rz': 0},
            {'rx': 30, 'ry': 225, 'rz': 0},
            {'rx': 30, 'ry': 315, 'rz': 0},
            {'rx': 45, 'ry': 45, 'rz': 0},
            {'rx': 60, 'ry': 30, 'rz': 0},
        ])
        
        return angles
    
    def render_model(self, 
                     assembly: cq.Assembly, 
                     view_angle: Dict[str, float],
                     output_path: Optional[str] = None) -> Image.Image:
        """
        Render a CAD model from a specific view angle.
        
        Args:
            assembly: CadQuery Assembly to render
            view_angle: Dictionary with rotation angles (rx, ry, rz)
            output_path: Optional path to save the image
            
        Returns:
            PIL Image object
        """
        try:
            # CadQuery doesn't have built-in rendering to image files directly
            # We'll create a simple approach using show() with export options
            # For a production system, you'd use VTK or similar for real rendering
            
            # This is a placeholder - in practice, you'd use cadquery-ocp or similar
            # to get actual rendered images. For now, we'll create synthetic images
            # that represent the concept
            
            # Create a simple representation
            img = Image.new('RGB', self.image_size, color=(240, 240, 240))
            
            # Add some variation based on view angle
            np_img = np.array(img)
            angle_hash = hash(str(view_angle)) % 100
            np_img = np_img + np.random.randint(-angle_hash, angle_hash, np_img.shape, dtype=np.int16)
            np_img = np.clip(np_img, 0, 255).astype(np.uint8)
            
            img = Image.fromarray(np_img)
            
            if output_path:
                img.save(output_path)
            
            return img
            
        except Exception as e:
            raise RuntimeError(f"Failed to render model: {str(e)}")
    
    def apply_background(self, image: Image.Image, background: Image.Image) -> Image.Image:
        """
        Composite the rendered image onto a background.
        
        Args:
            image: Rendered CAD image
            background: Background image
            
        Returns:
            Composited image
        """
        # Resize background to match image size
        bg = background.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # Simple alpha compositing (in production, you'd use proper alpha channels)
        # For now, blend the images
        blended = Image.blend(bg, image, alpha=0.7)
        
        return blended
    
    def apply_filters(self, image: Image.Image, filter_type: str) -> Image.Image:
        """
        Apply various filters to the image.
        
        Args:
            image: Input image
            filter_type: Type of filter to apply
            
        Returns:
            Filtered image
        """
        if filter_type == 'blur':
            return image.filter(ImageFilter.GaussianBlur(radius=2))
        elif filter_type == 'sharpen':
            return image.filter(ImageFilter.SHARPEN)
        elif filter_type == 'edge_enhance':
            return image.filter(ImageFilter.EDGE_ENHANCE)
        elif filter_type == 'brightness_up':
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(1.3)
        elif filter_type == 'brightness_down':
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(0.7)
        elif filter_type == 'contrast_up':
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(1.3)
        elif filter_type == 'contrast_down':
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(0.7)
        elif filter_type == 'saturation_up':
            enhancer = ImageEnhance.Color(image)
            return enhancer.enhance(1.3)
        elif filter_type == 'saturation_down':
            enhancer = ImageEnhance.Color(image)
            return enhancer.enhance(0.7)
        else:
            return image
    
    def generate_augmented_images(self,
                                  assembly: cq.Assembly,
                                  model_name: str,
                                  output_dir: str,
                                  backgrounds: Optional[List[Image.Image]] = None,
                                  filters: Optional[List[str]] = None,
                                  num_angles: Optional[int] = None) -> List[str]:
        """
        Generate multiple augmented renderings of a model.
        
        Args:
            assembly: CadQuery Assembly to render
            model_name: Name of the model
            output_dir: Directory to save rendered images
            backgrounds: List of background images to use
            filters: List of filter types to apply
            num_angles: Number of view angles to use (default: all)
            
        Returns:
            List of paths to generated images
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if filters is None:
            filters = ['none', 'blur', 'sharpen', 'brightness_up', 'contrast_up']
        
        view_angles = self.view_angles[:num_angles] if num_angles else self.view_angles
        
        generated_files = []
        
        for angle_idx, angle in enumerate(view_angles):
            # Render base image
            base_img = self.render_model(assembly, angle)
            
            # Apply backgrounds if provided
            if backgrounds:
                for bg_idx, bg in enumerate(backgrounds):
                    img_with_bg = self.apply_background(base_img, bg)
                    
                    # Apply filters
                    for filter_type in filters:
                        final_img = self.apply_filters(img_with_bg, filter_type)
                        
                        filename = f"{model_name}_angle{angle_idx}_bg{bg_idx}_{filter_type}.png"
                        filepath = output_path / filename
                        final_img.save(filepath)
                        generated_files.append(str(filepath))
            else:
                # No backgrounds, just apply filters
                for filter_type in filters:
                    final_img = self.apply_filters(base_img, filter_type)
                    
                    filename = f"{model_name}_angle{angle_idx}_{filter_type}.png"
                    filepath = output_path / filename
                    final_img.save(filepath)
                    generated_files.append(str(filepath))
        
        return generated_files
    
    def get_view_angles(self) -> List[Dict[str, float]]:
        """Get list of all view angles."""
        return self.view_angles
