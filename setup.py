from setuptools import setup, find_packages

setup(
    name="dg_reid",
    version="1.0.0",
    description="Visibility-Aware Visual Domain Erasure for Generalizable Person Re-Identification",
    author="Diya Bangera, Kapish Verma, Rishit Rana",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "Pillow>=9.3.0",
        "PyYAML>=6.0",
        "scikit-learn>=1.2.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12"
    ]
)
