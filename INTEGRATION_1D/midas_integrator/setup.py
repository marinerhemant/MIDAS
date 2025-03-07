from setuptools import setup, find_packages

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="midas_integrator",
    version="1.0.0",
    author="Hemant Sharma",
    author_email="hsharma@anl.gov",  # Replace with actual email
    description="A package for processing diffraction images and fitting Voigt profiles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marinerhemant/MIDAS",  # Replace with actual repo URL
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Open Source License",  # Adjust license as needed
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.4.0",
        "matplotlib>=3.1.0",
        "pillow>=7.0.0",
        "numba>=0.48.0",
    ],
    extras_require={
        "gpu": ["numba[cuda]>=0.48.0"],
        "dev": ["pytest", "pytest-cov", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "midas_integrator=midas_integrator.cli:main",
        ],
    },
)
