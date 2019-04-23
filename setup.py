from setuptools import setup, find_packages

setup(
    name="ImageCluster",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'Click==7.0',
        'numpy==1.16.2'
    ],
    author="Krzysztof Kowalczyk",
    author_email="kk385830@students.mimuw.edu.pl",
    license="BSD 2-Clause"
)
