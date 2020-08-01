from setuptools import find_packages
from setuptools import setup

# Package meta-data.
setup(
    name="thesis_project_heid",
    version="1.0",
    description="The package in which all my thesis replication is located.",
    author="Pascal Heid",
    author_email="s6plheid@uni-bonn.de",
    url=None,
    packages=find_packages(exclude=("tests",)),
    license="MIT",
    include_package_data=True,
)
