"""Setup script for sklearn-meta library."""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="sklearn-meta",
    version="0.1.0",
    description="A meta-learning library for sklearn-compatible ML models",
    author="sklearn-meta Contributors",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "xgboost": ["xgboost>=1.6.0"],
        "lightgbm": ["lightgbm>=3.3.0"],
        "catboost": ["catboost>=1.0.0"],
        "all": ["xgboost>=1.6.0", "lightgbm>=3.3.0", "catboost>=1.0.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=3.0.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
