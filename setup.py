from setuptools import setup, find_packages

setup(
    name="DeepLearningDevelopment",
    version="0.1.11",
    description="Personal Deep Learning Module",
    author="Sisung Kim",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "torch",
    ],
)
