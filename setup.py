from setuptools import setup, find_packages

setup(
    name="dg_reid",
    version="1.0.0",
    description="Visibility-Aware Visual Domain Erasure for Generalizable Person Re-Identification",
    author="Diya Bangera, Kapish Verma, Rishit Rana",
    author_email="kapishverma@nsut.ac.in",
    url="https://github.com/kapish19/btp-",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "pillow>=9.0.0",
        "pyyaml>=6.0",
        "ftfy>=6.0",
        "regex",
        "numpy>=1.22.0",
    ],
)
