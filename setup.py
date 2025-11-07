"""
Setup script for STP Classifier package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text() if readme_file.exists() else ''

# Read requirements
requirements_file = Path(__file__).parent / 'requirements.txt'
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')

setup(
    name='stp-classifier',
    version='1.0.0',
    description='Machine learning pipeline for classifying images based on 3D CAD models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Grayson, Ryan, Becca',
    python_requires='>=3.8',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'stp-train=src.train_pipeline:main',
            'stp-classify=src.classifier:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
