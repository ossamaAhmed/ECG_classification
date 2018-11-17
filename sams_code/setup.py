from setuptools import find_packages
from setuptools import setup
REQUIRED_PACKAGES = ['numpy==1.14.5', 'matplotlib', 'pandas==0.23.4', 'tensorflow==1.8.0']
setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='CNN'
)