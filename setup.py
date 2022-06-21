#!/usr/bin/env python

from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(name='cyclonysus',
      version='0.0.1',
      description="Representative Cycles doesn't have to be hard.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Nathaniel Saul',
      author_email='nathaniel.saul@wsu.edu',
      url='https://github.com/sauln/cyclonysus',
      license='MIT',
      packages=['cyclonysus'],
      include_package_data=True,
      install_requires=[
        # For the library
        'dionysus @ git+https://github.com/mrzv/dionysus.git#egg=dionysus',
        'numpy',

        # For the examples
        'scipy',
        'scikit-learn',
        'matplotlib'
      ],
      python_requires='>=3.6',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
      ],
      keywords='persistent homology, representative cycles'
     )
