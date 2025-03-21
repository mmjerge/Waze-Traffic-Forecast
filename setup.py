from setuptools import setup, find_packages
setup(
    name="waze_traffic_forecast",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "torch>=1.10.0",
        "pyarrow>=6.0.0",
        "matplotlib>=3.4.0",
        "networkx>=2.6.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "pytz>=2021.1",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "black>=21.6b0",
            "flake8>=3.9.2",
            "isort>=5.9.2",
            "jupyter>=1.0.0",
            "pytest-cov>=2.12.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "waze-inspect=scripts.inspect_waze_schema:main",
            "waze-build-graph=scripts.build_waze_graph:main",
            "waze-train=scripts.train_model:main",
        ],
    },
    author="Michael Jerge",
    author_email="mj6ux@virginia.edu",
    description="Traffic forecasting with graph transformers using Waze data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/waze-traffic-forecast",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",  
)