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
from PyFields import Field


class TestField(object):
    """ Done """

    def setup(self):
        sane_parameters()
        self.x = np.linspace(0, 13.5, 98)
        self.y = np.linspace(0, 8.19892, 125)
        self.F = Field(axis_x=self.x, axis_y=self.y, unit_x='m', unit_y='s')
        self.F_nounits = Field(axis_x=self.x, axis_y=self.y,
                               unit_x='', unit_y='')
        self.F_reversed_axis = Field(axis_x=self.x[::-1],
                                     axis_y=self.y[::-1],
                                     unit_x='m', unit_y='s')
        self.x_irreg = [xi**2 for xi in self.x]
        self.y_irreg = [yi**2 for yi in self.y]
        self.F_irregular_axis = Field(axis_x=self.x_irreg,
                                      axis_y=self.y_irreg,
                                      unit_x='m', unit_y='s')
        self.Fs = [self.F, self.F_nounits, self.F_reversed_axis,
                   self.F_irregular_axis]
        self.Fs_reg = [self.F, self.F_nounits, self.F_reversed_axis]

    def test_init(self):
        # normal init
        F = Field(self.x, self.y, 'm', 'mm')
        assert np.all(F.axis_x == self.x)
        assert np.all(F.axis_y == self.y)
        assert F.unit_x.strUnit() == '[m]'
        assert F.unit_y.strUnit() == '[mm]'
        assert F.dx == self.x[1] - self.x[0]
        assert F.dy == self.y[1] - self.y[0]
        assert F.shape[0] == len(self.x)
        assert F.shape[1] == len(self.y)
        assert F._Field__is_axis_x_regular
        assert F._Field__is_axis_y_regular
        # init without units
        F = Field(self.x, self.y)
        assert np.all(F.axis_x == self.x)
        assert np.all(F.axis_y == self.y)
        assert F.unit_x.strUnit() == '[]'
        assert F.unit_y.strUnit() == '[]'
        assert F.dx == self.x[1] - self.x[0]
        assert F.dy == self.y[1] - self.y[0]
        assert F.shape[0] == len(self.x)
        assert F.shape[1] == len(self.y)
        assert F._Field__is_axis_x_regular
        assert F._Field__is_axis_y_regular
        # init with reversed axis
        F = Field(self.x[::-1], self.y[::-1])
        assert np.all(F.axis_x == self.x)
        assert np.all(F.axis_y == self.y)
        assert F.unit_x.strUnit() == '[]'
        assert F.unit_y.strUnit() == '[]'
        assert F.dx == self.x[1] - self.x[0]
        assert F.dy == self.y[1] - self.y[0]
        assert F.shape[0] == len(self.x)
        assert F.shape[1] == len(self.y)
        assert F._Field__is_axis_x_regular
        assert F._Field__is_axis_y_regular
        # init with irregular axis
        F = Field(self.x_irreg, self.y_irreg, 's', 'us')
        assert np.all(F.axis_x == self.x_irreg)
        assert np.all(F.axis_y == self.y_irreg)
        assert F.unit_x.strUnit() == '[s]'
        assert F.unit_y.strUnit() == '[us]'
        with pytest.raises(Exception):
            F.dx
        with pytest.raises(Exception):
            F.dy
        assert F.shape[0] == len(self.x)
        assert F.shape[1] == len(self.y)
        assert not F._Field__is_axis_x_regular
        assert not F._Field__is_axis_y_regular

    def test_axis(self):
        for F in self.Fs:
            # axis should raise error if set with different size
            with pytest.raises(ValueError):
                F.axis_x = [1, 2, 3]
            with pytest.raises(ValueError):
                F.axis_y = [1, 2, 3]
            # axis should raise error if set with something else than an array
            with pytest.raises(ValueError):
                F.axis_x = 'not an array'
            with pytest.raises(ValueError):
                F.axis_y = 'not an array'
            with pytest.raises(TypeError):
                F.axis_x = F
            with pytest.raises(TypeError):
                F.axis_y = F
        for F in [self.F, self.F_nounits, self.F_reversed_axis]:
            # axis should update dx and dy
            dx = F.dx
            F.axis_x = self.x/10
            assert dx/10 == F.dx
            dy = F.dy
            F.axis_y = self.y/10
            assert dy/10 == F.dy

    def test_units(self):
        for F in self.Fs:
            # units should be settable by string or unum
            F.unit_x = 'Hz'
            assert F.unit_x.strUnit() == '[Hz]'
            F.unit_y = 'kg'
            assert F.unit_y.strUnit() == '[kg]'
            F.unit_x = units.Hz
            assert F.unit_x.strUnit() == '[Hz]'
            F.unit_y = 1*units.kg
            assert F.unit_y.strUnit() == '[kg]'
            # units should not be settable by someting else
            with pytest.raises(TypeError):
                F.unit_x = 45
            with pytest.raises(TypeError):
                F.unit_y = 35.12
            with pytest.raises(TypeError):
                F.unit_x = F
            with pytest.raises(TypeError):
                F.unit_y = F
            # units should be normalized
            with pytest.raises(ValueError):
                F.unit_x = 10*units.m
            with pytest.raises(ValueError):
                F.unit_y = 8.3*units.s/units.kg

    def test_get_indice_on_axis(self):
        # Should return bounds indice on axis
        bds = self.F.get_indice_on_axis('x', 7.4, kind='bounds')
        assert self.F.axis_x[bds[0]] < 7.4
        assert self.F.axis_x[bds[1]] > 7.4
        assert bds[1] == bds[0] + 1
        bds = self.F.get_indice_on_axis('y', 4.45, kind='bounds')
        assert self.F.axis_y[bds[0]] < 4.45
        assert self.F.axis_y[bds[1]] > 4.45
        assert bds[1] == bds[0] + 1
        # Should return nearest indice
        nst = self.F.get_indice_on_axis('x', 3.78, kind='nearest')
        val = self.F.axis_x[nst]
        valp = self.F.axis_x[nst + 1]
        valm = self.F.axis_x[nst - 1]
        assert abs(val - 3.78) < abs(valp - 3.78)
        assert abs(val - 3.78) < abs(valm - 3.78)
        nst = self.F.get_indice_on_axis('y', 5.78, kind='nearest')
        val = self.F.axis_y[nst]
        valp = self.F.axis_y[nst + 1]
        valm = self.F.axis_y[nst - 1]
        assert abs(val - 5.78) < abs(valp - 5.78)
        assert abs(val - 5.78) < abs(valm - 5.78)
        # Should return decimal indice
        dec = self.F.get_indice_on_axis('x', 11.91, kind='decimal')
        assert dec == 85.57555555555555
        dec = self.F.get_indice_on_axis('y', 1.91, kind='decimal')
        assert dec == 28.886731423163056
        # should raise an error for wrong direction
        with pytest.raises(ValueError):
            self.F.get_indice_on_axis('truc', 8.18)
        with pytest.raises(ValueError):
            self.F.get_indice_on_axis(9.1, 8.18)
        # should raise an error for out of bound value
        with pytest.raises(ValueError):
            self.F.get_indice_on_axis('x', -7)
        with pytest.raises(ValueError):
            self.F.get_indice_on_axis('x', 89)
        with pytest.raises(ValueError):
            self.F.get_indice_on_axis('y', -4)
        with pytest.raises(ValueError):
            self.F.get_indice_on_axis('y', 14)
        # should raise an error for wrong kind
        with pytest.raises(ValueError):
            self.F.get_indice_on_axis('x', 4, 'something_else')
        with pytest.raises(ValueError):
            self.F.get_indice_on_axis('y', 4, 'something_else')

    def test_scale(self):
        # should be scalable by numbers
        for F in self.Fs_reg:
            tmp_F = F.scale(scalex=10)
            assert tmp_F.dx == 10*F.dx
            assert np.all(tmp_F.axis_x == 10*F.axis_x)
            tmp_F = F.scale(scaley=1.43)
            assert tmp_F.dy == 1.43*F.dy
            assert np.all(tmp_F.axis_y == 1.43*F.axis_y)
            tmp_F = F.scale(scalex=10, scaley=1.43)
            assert tmp_F.dx == 10*F.dx
            assert np.all(tmp_F.axis_x == 10*F.axis_x)
            assert tmp_F.dy == 1.43*F.dy
            assert np.all(tmp_F.axis_y == 1.43*F.axis_y)
        # should be scalable by units
        u1 = 10*units.m/units.s
        tmp_F = self.F.scale(scalex=u1)
        assert tmp_F.unit_x.strUnit() == '[m2/s]'
        assert tmp_F.dx == 10*self.F.dx
        assert np.all(tmp_F.axis_x == 10*self.F.axis_x)
        u2 = units.ms
        tmp_F = self.F.scale(scaley=u2)
        assert tmp_F.unit_y.strUnit() == '[ms.s]'
        assert tmp_F.dy == self.F.dy/1000
        assert np.allclose(tmp_F.axis_y, self.F.axis_y/1000)
        # Should not modified the source
        save_F = self.F.copy()
        self.F.scale(10, 2)
        assert np.all(save_F.axis_x == self.F.axis_x)
        assert np.all(save_F.unit_x == self.F.unit_x)
        assert np.all(save_F.axis_y == self.F.axis_y)
        assert np.all(save_F.unit_y == self.F.unit_y)
        # should modify in place when inplace is true
        save_F = self.F.copy()
        tmp_F = self.F.copy()
        tmp_F.scale(scalex=10, scaley=1.43, inplace=True)
        assert 10*save_F.dx == tmp_F.dx
        assert np.all(10*save_F.axis_x == tmp_F.axis_x)
        assert 1.43*save_F.dy == tmp_F.dy
        assert np.all(1.43*save_F.axis_y == tmp_F.axis_y)
        # should raise an error when scale is inadequate
        with pytest.raises(TypeError):
            self.F.scale('test')
        with pytest.raises(TypeError):
            self.F.scale(self.F)

    def test_rotate(self):
        # should rotate
        tmp_F = self.F.rotate(90)
        assert np.all(tmp_F.axis_x == -self.F.axis_y[::-1])
        assert np.all(tmp_F.axis_y == self.F.axis_x)
        tmp_F = self.F.rotate(-90)
        assert np.all(tmp_F.axis_x == self.F.axis_y)
        assert np.all(tmp_F.axis_y == -self.F.axis_x[::-1])
        # should not modify source
        save_F = self.F.copy()
        self.F.rotate(90)
        assert save_F == self.F
        # should rotate inplace whan asked
        tmp_F = self.F.copy()
        tmp_F.rotate(270, inplace=True)
        assert np.all(tmp_F.axis_x == self.F.axis_y)
        assert np.all(tmp_F.axis_y == -self.F.axis_x[::-1])

    def test_change_unit(self):
        # should change unit
        tmp_F = self.F.copy()
        tmp_F.change_unit('x', 'mm')
        assert tmp_F.unit_x.strUnit() == '[mm]'
        assert np.allclose(tmp_F.axis_x, self.F.axis_x*1000)
        tmp_F = self.F.copy()
        tmp_F.change_unit('y', 'us')
        assert tmp_F.unit_y.strUnit() == '[us]'
        assert np.allclose(tmp_F.axis_y, self.F.axis_y*1e6)
        # should not change if unit is not coherent
        with pytest.raises(unum.IncompatibleUnitsError):
            self.F.change_unit('x', 'Hz')
        # should raise an error if the arg are not strings
        with pytest.raises(TypeError):
            self.F.change_unit(45, 'Hz')
        with pytest.raises(TypeError):
            self.F.change_unit('x', 43)

    def test_change_origin(self):
        # should change origin
        tmp_F = self.F.copy()
        tmp_F.set_origin(x=4, y=3)
        assert np.allclose(tmp_F.axis_x + 4, self.F.axis_x)
        assert np.allclose(tmp_F.axis_y + 3, self.F.axis_y)
        # should raise an error if args are not numbers
        with pytest.raises(TypeError):
            tmp_F.set_origin('test', 4)
        with pytest.raises(TypeError):
            tmp_F.set_origin(3, 'test')

    def test_crop(self):
        # should crop
        tmp_F = self.F.crop(intervx=[3, 10],
                            intervy=[2, 6])
        assert tmp_F.axis_x[0] == 3.061855670103093
        assert tmp_F.axis_x[-1] == 9.881443298969073
        assert len(tmp_F.axis_x) == 50
        assert tmp_F.shape[0] == 50
        #  should crop with indice
        tmp_F = self.F.crop(intervx=[3, 30],
                            intervy=[2, 60], ind=True)
        assert len(tmp_F.axis_x) == 28
        assert len(tmp_F.axis_y) == 59
        assert np.allclose(tmp_F.axis_x, self.F.axis_x[3:31])
        assert np.allclose(tmp_F.axis_y, self.F.axis_y[2:61])
        # should modify inplace if asked
        tmp_F = self.F.copy()
        tmp_F.crop(intervx=[3, 10], intervy=[2, 6], inplace=True)
        assert tmp_F.axis_x[0] == 3.061855670103093
        assert tmp_F.axis_x[-1] == 9.881443298969073
        assert len(tmp_F.axis_x) == 50
        assert tmp_F.shape[0] == 50
        # should return indices when asked
        r, l, u, d, tmp_F = self.F.crop(intervx=[3, 10],
                                        intervy=[2, 6],
                                        full_output=True)
        assert r == 22
        assert l == 71
        assert u == 31
        assert d == 90

    def test_extend(self):
        # should extend
        tmp_F = self.F.extend(5, 8, 1, 3)
        assert len(self.F.axis_x) + 13 == len(tmp_F.axis_x)
        assert len(self.F.axis_y) + 4 == len(tmp_F.axis_y)
        assert np.allclose(self.F.axis_x, tmp_F.axis_x[5:-8])
        assert np.allclose(self.F.axis_y, tmp_F.axis_y[3:-1])
        # should extend in place if asked
        tmp_F = self.F.copy()
        tmp_F.extend(5, 8, 1, 3, inplace=True)
        assert len(self.F.axis_x) + 13 == len(tmp_F.axis_x)
        assert len(self.F.axis_y) + 4 == len(tmp_F.axis_y)
        assert np.allclose(self.F.axis_x, tmp_F.axis_x[5:-8])
        assert np.allclose(self.F.axis_y, tmp_F.axis_y[3:-1])
