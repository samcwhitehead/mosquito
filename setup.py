from setuptools import setup
from setuptools import find_packages
import os

setup(
    name='mosquito',
    version='0.0.1',
    description='Code for analyzing high speed video and electrophysiology data from mosquitoes.',
    long_description=__doc__,
    author='Sam Whitehead',
    author_email='swhitehe@caltech',
    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
    ],

    packages=find_packages(exclude=['examples', 'scratch']),
)
