from setuptools import setup, find_packages

setup(
    name="DeepLearningDevelopment",
    version="0.1.3",
    description="Personal Deep Learning Module",
    author="Sisung Kim",
    packages=find_packages(),
    install_requires=[
        "numpy>=1,<2",
        "torch>=2,<3",
    ],
)
