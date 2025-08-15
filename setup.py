
from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="aspectfl",
    version="1.0.0",
    author="Anas ALSobeh",
    author_email="anas.alsobeh@siu.edu",
    description="Aspect-Oriented Programming for Trustworthy and Compliant Federated Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aspectfl/aspectfl",
    project_urls={
        "Bug Tracker": "https://github.com/aspectfl/aspectfl/issues",
        "Documentation": "https://aspectfl.readthedocs.io/",
        "Source Code": "https://github.com/aspectfl/aspectfl",
        "Research Paper": "https://doi.org/10.3390/info16010xxx",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "experiments": [
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
            "plotly>=5.0.0",
            "dash>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aspectfl=aspectfl.cli:main",
            "aspectfl-experiment=experiments.comprehensive_experiments:main",
            "aspectfl-analyze=experiments.data_analysis:main",
        ],
    },
    include_package_data=True,
    package_data={
        "aspectfl": [
            "data/*.json",
            "templates/*.yaml",
            "schemas/*.json",
        ],
    },
    keywords=[
        "federated learning",
        "aspect-oriented programming",
        "trustworthy AI",
        "FAIR principles",
        "security",
        "compliance",
        "privacy",
        "machine learning",
        "distributed computing",
    ],
    license="MIT",
    zip_safe=False,
)

