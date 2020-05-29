#! /usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

ver_dic = {}
version_file_name = "pytato/version.py"
with open(version_file_name) as version_file:
    version_file_contents = version_file.read()

exec(compile(version_file_contents, version_file_name, 'exec'), ver_dic)

setup(name="pytato",
      version=ver_dic["VERSION_TEXT"],
      description="Get Descriptions of Array Computations via Lazy Evaluation",
      long_description=open("README.rst", "r").read(),
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Other Audience',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Visualization',
          'Topic :: Software Development :: Libraries',
          ],

      install_requires=[
          "loo.py",
          ],

      author="Andreas Kloeckner, Matt Wala, Xiaoyu Wei",
      url="http://github.com/inducer/pytato",
      author_email="inform@tiker.net",
      license="MIT",
      packages=find_packages())
