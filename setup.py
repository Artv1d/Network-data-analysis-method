from setuptools import setup, find_packages

setup(
    name="network-data-analysis-method",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "process-data=src.app:main",
        ],
    },
    python_requires=">=3.8",
)
