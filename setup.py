from setuptools import setup, find_packages

setup(
    name="DeepLearningDevelopment",
    version="0.1.3",
    description="Personal Deep Learning Module",
    author="Sisung Kim",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1,<2",
        "torch>=2,<3",
    ],
)
