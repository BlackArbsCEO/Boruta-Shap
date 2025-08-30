from setuptools import setup

def readme():
    with open('README.md') as f:
        README = f.read()
    return README

setup(
    name="BorutaShap",
    version="1.0.16",
    description="A feature selection algorithm.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Ekeany/Boruta-Shap",
    author="Eoghan Keany",
    author_email="egnkeany@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    py_modules = ["BorutaShap"],
    package_dir = {"" : "src"},
    python_requires=">=3.7",
    install_requires=[
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "statsmodels>=0.13.0",
        "matplotlib>=3.3.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "shap>=0.40.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0"
    ],
)