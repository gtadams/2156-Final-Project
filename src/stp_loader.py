"""
STP File Loader Module
Handles importing and managing .stp (STEP) CAD files.
"""

import os
from pathlib import Path
from typing import List, Dict
import cadquery as cq


class STPLoader:
    """Class for loading and managing .stp files."""
    
    def __init__(self, stp_directory: str):
        """
        Initialize the STP loader.
        
        Args:
            stp_directory: Path to directory containing .stp files
        """
        self.stp_directory = Path(stp_directory)
        self.stp_files: Dict[str, Path] = {}
        self.loaded_models: Dict[str, cq.Assembly] = {}
        
    def scan_directory(self) -> List[str]:
        """
        Scan directory for .stp and .step files.
        
        Returns:
            List of found .stp/.step file names
        """
        if not self.stp_directory.exists():
            raise ValueError(f"Directory {self.stp_directory} does not exist")
        
        # Find all .stp and .step files
        stp_patterns = ['*.stp', '*.step', '*.STP', '*.STEP']
        for pattern in stp_patterns:
            for file_path in self.stp_directory.glob(pattern):
                if file_path.is_file():
                    self.stp_files[file_path.stem] = file_path
        
        return list(self.stp_files.keys())
    
    def load_stp_file(self, file_key: str) -> cq.Assembly:
        """
        Load a specific .stp file.
        
        Args:
            file_key: Name of the file (without extension)
            
        Returns:
            CadQuery Assembly object
        """
        if file_key not in self.stp_files:
            raise ValueError(f"File {file_key} not found. Available files: {list(self.stp_files.keys())}")
        
        file_path = self.stp_files[file_key]
        
        try:
            # Import STEP file using CadQuery
            imported_shape = cq.importers.importStep(str(file_path))
            
            # Create assembly from the shape
            assembly = cq.Assembly()
            assembly.add(imported_shape, name=file_key)
            
            self.loaded_models[file_key] = assembly
            return assembly
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {file_path}: {str(e)}")
    
    def load_all(self) -> Dict[str, cq.Assembly]:
        """
        Load all .stp files found in the directory.
        
        Returns:
            Dictionary mapping file names to Assembly objects
        """
        file_keys = self.scan_directory()
        
        for file_key in file_keys:
            try:
                self.load_stp_file(file_key)
            except Exception as e:
                print(f"Warning: Could not load {file_key}: {str(e)}")
        
        return self.loaded_models
    
    def get_model(self, file_key: str) -> cq.Assembly:
        """
        Get a loaded model by name.
        
        Args:
            file_key: Name of the file (without extension)
            
        Returns:
            CadQuery Assembly object
        """
        if file_key not in self.loaded_models:
            return self.load_stp_file(file_key)
        return self.loaded_models[file_key]
    
    def list_models(self) -> List[str]:
        """
        List all available model names.
        
        Returns:
            List of model names
        """
        return list(self.stp_files.keys())
