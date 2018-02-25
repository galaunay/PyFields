# -*- coding: utf-8 -*-
#!/bin/env python3

# Copyright (C) 2018 Gaby Launay

# Author: Gaby Launay  <gaby.launay@tutanota.com>
# URL: https://github.com/galaunay/pyfields
# Version: 0.1

# This file is part of PyFields

# PyFields is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# PyFields is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup, find_packages

setup(
    name='PyFields',
    version='1.0',
    description='Tools to work with 2D and 3D fields',
    author='Gaby Launay',
    author_email='gaby.launay@tutanota.com',
    license='GPLv3',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GPLv3 License',
        'Programming Language :: Python :: 3.5',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
    ],
    keywords='fields velocity concentration sets',
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'samples']),
    install_requires=['numpy', 'matplotlib', 'scipy', 'unum'],
    extras_require={},
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov'],
)
