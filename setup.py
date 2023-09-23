from pathlib import Path
from setuptools import find_packages, setup

# Package meta-data
NAME = 'aisg-regression-model'
EMAIL = 'joannakhek@gmail.com'
AUTHOR = 'Joanna Khek Cuina'

    
setup(
    name=NAME,
    author=AUTHOR,
    author_email=EMAIL,
    packages=find_packages(exclude=('tests')),
    package_data={"regression_model": ["VERSION"]},
    version='0.0.1',
    include_package_data=True,
    license="BSD-3",
)
