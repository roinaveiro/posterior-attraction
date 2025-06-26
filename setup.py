from setuptools import setup, find_packages

setup(
    name="posterior-attraction",
    version="0.1.0",
    packages=find_packages(include=["posterior-attraction", "posterior-attraction.*"]),
    install_requires=[
        #"torch>=1.13.0",
        #"torchvision>=0.14.0",
        #"torchaudio>=0.13.0",
        #"pyyaml>=6.0",
        #"matplotlib>=3.5.0",
        #"notebook>=6.4.0",
        #"pytest>=7.0.0",
    ],
    entry_points={
        "console_scripts": [
            "run_experiment=posterior_attraction.experiments.experiment_runner:run_experiment",
        ]
    },
    description="A package for posterior attraction experiments using KL divergence minimization.",
    author="Roi Naveiro",
    author_email="roi.naveiro@cunef.edu",
    url="https://github.com/roinaveiro/posterior-attraction",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
