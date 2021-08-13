from setuptools import setup

with open("aml/VERSION") as f:
    version = f.read().strip()


def list_reqs(fname='requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()


setup(
    name='aml',
    version=version,
    description='xtreme gridsearchCV',
    author='Jerzy Słomkowski',
    install_requires=list_reqs()
)
