from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='torchswarm',
    version='0.0.1',    
    description='A fast implementation of Particle Swarm Optimization using PyTorch',
    url='https://github.com/rohanmohapatra/torchswarm',
    author='Rohan Mohapatra',
    license='MIT',
    install_requires=['torch'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.6",
)