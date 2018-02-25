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

import os
import sys
try:
    dirname = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dirname)
    os.chdir('..')
    sys.path.append(os.getcwd())
except:
    pass

import numpy as np
import pytest
import unum
import unum.units as units

from helper import sane_parameters
from PyFields.utils import make_unit


class TestUnits(object):
    """ Done """

    def setup(self):
        sane_parameters()
        self.u1 = units.m/units.s
        self.u2 = units.Hz/units.kg

    def test_make_unit(self):
        # should create units
        assert self.u1 == make_unit('m/s')
        assert self.u2 == make_unit('Hz/kg')
        # should fail for badly formed units
        with pytest.raises(Exception):
            make_unit('import antigravity')
        with pytest.raises(Exception):
            make_unit('kill_everything()')
        with pytest.raises(Exception):
            make_unit('metres par heures')
