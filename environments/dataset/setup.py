from setuptools import setup, find_packages

setup(
    name="environments.dataset",
    version="0.2",
    description="Dataset for D3IL",
    license="MIT",
    package_data={"dataset": ["*"]},
    packages=find_packages(),
)
