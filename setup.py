from setuptools import setup, find_packages


def list_reqs(fname='requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()


setup(
    name='aml',
    version='0.1',
    description='xtreme gridsearchCV',
    author='Jerzy Słomkowski',
    packages=find_packages(exclude=['tests']),
    install_requires=list_reqs()

)
