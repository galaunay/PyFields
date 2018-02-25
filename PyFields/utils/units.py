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

import re
import unum
import unum.units as units
units.counts = unum.Unum.unit('counts')
units.px = unum.Unum.unit('px')


def make_unit(string):
    """
    Create an Unum unit from a string.
    For more details, see the Unum module documentation.

    Parameters
    ----------
    string : string
        String representing some units (e.g. 'kg.m^-2.s^-1', or 'kg/m^2/s').

    Returns
    -------
    unit : unum.Unum object
        unum object representing the given units.

    Examples
    --------
    >>> make_unit("m/s")
    1 [m/s]
    >>> make_unit("N/m/s**3")
    1 [kg/s4]
    """
    if string == "":
        return unum.Unum({})
    # Safe check
    forbidden = ['import', '=', '\n', ';', ':', '"', "'", "open"]
    forbidden += [key for key in globals().keys()]
    for f in forbidden:
        if f in string:
            raise Exception("Unauthorized string: {}".format(f))
    if re.match("[a-zA-Z]+\(.*\)", string):
        raise Exception("No call allowed here")
    # exec
    env = {}
    exec("from unum.units import *;res = {}".format(string), env)
    return env['res']
