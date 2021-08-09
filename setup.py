from setuptools import setup


def list_reqs(fname='requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()


setup(
    name='aml',
    version='0.1',
    description='xtreme gridsearchCV',
    author='Jerzy Słomkowski',
    install_requires=list_reqs()
)
