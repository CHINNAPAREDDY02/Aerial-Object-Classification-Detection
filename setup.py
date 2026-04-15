from setuptools import setup, find_packages

setup(
    name="aerial_object_classifier",
    version="1.0.0",
    description="Aerial Bird vs Drone Classification & Detection",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "tensorflow>=2.12.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.7.0",
        "Pillow>=9.5.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "streamlit>=1.28.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0",
    ],
)
