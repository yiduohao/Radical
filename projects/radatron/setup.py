from setuptools import find_packages, setup

setup(
    name="radatron",
    version="1.0",
    author="Waleed Ahmed",
    description="Radatron: Accurate Detection using Multi-Resolution Cascaded MIMO Radar, ECCV 2022",
    packages=find_packages(exclude=("configs", "tests")),
    #install_requires=["detectron2"],
)