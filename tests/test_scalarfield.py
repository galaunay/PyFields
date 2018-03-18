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
import matplotlib.pyplot as plt
import numpy as np
import pytest
import unum
import unum.units as units

from helper import sane_parameters
from PyFields import ScalarField, Field



class TestScalarField(object):
    """ Done """

    def setup(self):
        sane_parameters()
        # data
        self.x = np.linspace(0, 13.5, 98)
        self.y = np.linspace(0, 8.19892, 125)
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        self.values = 3*np.cos(X) - 5*np.sin(0.43*Y)
        self.values_int = np.array(3*np.cos(X) - 5*np.sin(0.43*Y), dtype=int)
        self.values_nans = self.values.copy()
        self.values_nans[self.values >5] = np.nan
        self.mask = self.values > 4
        # create scalar fields
        self.SF = ScalarField(axis_x=self.x,
                              axis_y=self.y,
                              mask=None,
                              values=self.values,
                              unit_x='m',
                              unit_y='mm',
                              unit_values='m/s')
        self.SF_int = ScalarField(axis_x=self.x,
                                  axis_y=self.y,
                                  mask=None,
                                  values=self.values_int,
                                  unit_x='m',
                                  unit_y='mm',
                                  unit_values='m/s')
        self.SF_mask = ScalarField(axis_x=self.x,
                                   axis_y=self.y,
                                   mask=self.mask,
                                   values=self.values,
                                   unit_x='m',
                                   unit_y='mm',
                                   unit_values='m/s')
        self.SF_nans = ScalarField(axis_x=self.x,
                                   axis_y=self.y,
                                   mask=None,
                                   values=self.values_nans,
                                   unit_x='m',
                                   unit_y='mm',
                                   unit_values='m/s')
        self.SFs = [self.SF, self.SF_mask, self.SF_nans, self.SF_int]

    def test_init(self):
        # normal init
        SF = ScalarField(self.x, self.y, self.values,
                         None, 'm', 'mm', 'm/s')
        assert np.all(SF.axis_x == self.x)
        assert np.all(SF.axis_y == self.y)
        assert np.all(SF.values == self.values)
        assert not np.all(SF.mask)
        assert SF.unit_x.strUnit() == '[m]'
        assert SF.unit_y.strUnit() == '[mm]'
        assert SF.unit_values.strUnit() == '[m/s]'
        assert SF.dtype == np.float
        assert SF.dx == self.x[1] - self.x[0]
        assert SF.dy == self.y[1] - self.y[0]
        assert SF.shape[0] == len(self.x)
        assert SF.shape[1] == len(self.y)
        assert SF.shape[0] == self.values.shape[0]
        assert SF.shape[1] == self.values.shape[1]
        assert SF._Field__is_axis_x_regular
        assert SF._Field__is_axis_y_regular
        # with mask
        SF = ScalarField(self.x, self.y, self.values,
                         self.mask, 'm', 'mm', 'm/s')
        assert np.all(SF.axis_x == self.x)
        assert np.all(SF.axis_y == self.y)
        assert np.all(SF.values[~SF.mask] == self.values[~self.mask])
        assert np.all(SF.mask == self.mask)
        assert SF.unit_x.strUnit() == '[m]'
        assert SF.unit_y.strUnit() == '[mm]'
        assert SF.unit_values.strUnit() == '[m/s]'
        assert SF.dtype == np.float
        assert SF.dx == self.x[1] - self.x[0]
        assert SF.dy == self.y[1] - self.y[0]
        assert SF.shape[0] == len(self.x)
        assert SF.shape[1] == len(self.y)
        assert SF.shape[0] == self.values.shape[0]
        assert SF.shape[1] == self.values.shape[1]
        assert SF._Field__is_axis_x_regular
        assert SF._Field__is_axis_y_regular
        # with nans
        SF = ScalarField(self.x, self.y, self.values_nans,
                         None, 'm', 'mm', 'm/s')
        assert np.all(SF.axis_x == self.x)
        assert np.all(SF.axis_y == self.y)
        assert np.all(SF.values == self.values)
        assert np.all(SF.mask == np.isnan(self.values_nans))
        assert SF.unit_x.strUnit() == '[m]'
        assert SF.unit_y.strUnit() == '[mm]'
        assert SF.unit_values.strUnit() == '[m/s]'
        assert SF.dtype == np.float
        assert SF.dx == self.x[1] - self.x[0]
        assert SF.dy == self.y[1] - self.y[0]
        assert SF.shape[0] == len(self.x)
        assert SF.shape[1] == len(self.y)
        assert SF.shape[0] == self.values.shape[0]
        assert SF.shape[1] == self.values.shape[1]
        assert SF._Field__is_axis_x_regular
        assert SF._Field__is_axis_y_regular
        # with integers
        SF = ScalarField(self.x, self.y, self.values_int,
                         None, 'm', 'mm', 'm/s')
        assert np.all(SF.axis_x == self.x)
        assert np.all(SF.axis_y == self.y)
        assert np.all(SF.values == self.values_int)
        assert np.all(SF.mask == np.isnan(self.values_nans))
        assert SF.unit_x.strUnit() == '[m]'
        assert SF.unit_y.strUnit() == '[mm]'
        assert SF.unit_values.strUnit() == '[m/s]'
        assert SF.dtype == np.int
        assert SF.dx == self.x[1] - self.x[0]
        assert SF.dy == self.y[1] - self.y[0]
        assert SF.shape[0] == len(self.x)
        assert SF.shape[1] == len(self.y)
        assert SF.shape[0] == self.values.shape[0]
        assert SF.shape[1] == self.values.shape[1]
        assert SF._Field__is_axis_x_regular
        assert SF._Field__is_axis_y_regular
        # should raise error if length are incoherent
        with pytest.raises(ValueError):
            SF = ScalarField(self.x[1:], self.y, self.values_int,
                             None, 'm', 'mm', 'm/s')
        with pytest.raises(ValueError):
            SF = ScalarField(self.x, self.y[0:-12], self.values_int,
                             None, 'm', 'mm', 'm/s')
        # should raise an error if mask is not a boolean array
        with pytest.raises(ValueError):
            SF = ScalarField(self.x, self.y, self.values_int,
                             [1, 2, 'not everything'], 'm', 'mm', 'm/s')

    def test_values_property(self):
        #should update mask
        tmp_SF = self.SF.copy()
        val = tmp_SF.values
        mask = val > 2
        val[mask] = np.nan
        tmp_SF.values = val
        assert np.all(tmp_SF.mask == mask)
        # should raise error for incoherent axis sizes
        with pytest.raises(ValueError):
            tmp_SF.values = val[1::, :]
        with pytest.raises(ValueError):
            tmp_SF.values = val[:, :-3]

    def test_mask_property(self):
        # should raise error for incoherent axis sizes
        tmp_SF = self.SF.copy()
        val = tmp_SF.values
        mask = val > 2
        with pytest.raises(ValueError):
            tmp_SF.mask = mask[1::, :]
        with pytest.raises(ValueError):
            tmp_SF.mask = mask[:, :-12]

    def test_mask_as_sf(self):
        mask = self.SF.mask_as_sf
        assert np.all(mask.values == self.SF.mask)
        assert not np.any(mask.mask)
        assert np.all(mask.axis_x == self.SF.axis_x)
        assert np.all(mask.axis_y == self.SF.axis_y)
        assert np.all(mask.unit_x == self.SF.unit_x)
        assert np.all(mask.unit_y == self.SF.unit_y)
        assert np.all(mask.unit_values.strUnit() == "[]")

    def test_equality(self):
        tmp_SF = self.SF.copy()
        # should return True if equal
        assert tmp_SF == self.SF
        assert not tmp_SF != self.SF
        # should return False if different
        tmp_SF = self.SF.copy()
        values = tmp_SF.values
        values[0, 0] = 43
        tmp_SF.values = values
        assert not tmp_SF == self.SF
        assert tmp_SF != self.SF
        tmp_SF = self.SF.copy()
        tmp_SF.unit_values = "Hz"
        assert not tmp_SF == self.SF
        assert tmp_SF != self.SF
        tmp_SF = self.SF.copy()
        mask = self.mask.copy()
        mask[2, 2] = True
        tmp_SF.mask = mask
        assert tmp_SF != self.SF

    def test_addition(self):
        # should add other scalarfield if coherent axis system
        tmp_SF = self.SF.copy()
        tmp_SF.values *= 3
        res_SF = tmp_SF + self.SF
        assert np.all(res_SF.values == 4*self.SF.values)
        assert np.all(res_SF.mask == self.SF.mask)
        assert Field.__eq__(res_SF, self.SF)
        # should add with cropped scalarfield if coherent axis system
        tmp_SF = self.SF.copy().crop(intervx=[10, 50], ind=True)
        tmp_SF.values *= 3
        res_SF = tmp_SF + self.SF
        assert res_SF.shape == tmp_SF.shape
        assert res_SF.shape[0] == 41
        assert np.all(res_SF.values == 4*self.SF.values[10:51])
        assert np.all(res_SF.mask == self.SF.mask[10:51])
        assert not Field.__eq__(res_SF, self.SF)
        # should add with numbers
        tmp_SF = self.SF + 5
        assert np.all(tmp_SF.values == 5 + self.SF.values)
        assert np.all(tmp_SF.mask == self.SF.mask)
        assert Field.__eq__(tmp_SF, self.SF)
        # should add with units if coherent
        tmp_SF = self.SF + 5.2*units.m/units.s
        assert np.all(tmp_SF.values == 5.2 + self.SF.values)
        assert np.all(tmp_SF.mask == self.SF.mask)
        assert Field.__eq__(tmp_SF, self.SF)
        # shoudl raise error if incoherent axis
        tmp_SF = self.SF.copy()
        tmp_SF.axis_x += 1
        with pytest.raises(ValueError):
            tmp_SF + self.SF
        tmp_SF = self.SF.copy()
        tmp_SF.axis_y += 3.4
        with pytest.raises(ValueError):
            tmp_SF + self.SF
        # should raise an error if incoherent units
        tmp_SF = self.SF.copy()
        tmp_SF.unit_x = 'Hz'
        with pytest.raises(unum.IncompatibleUnitsError):
            tmp_SF + self.SF

    def test_substitutaion(self):
        # should add other scalarfield if coherent axis system
        tmp_SF = self.SF.copy()
        tmp_SF.values *= 3
        res_SF = tmp_SF - self.SF
        assert np.allclose(res_SF.values, 2*self.SF.values)
        assert np.all(res_SF.mask == 2*self.SF.mask)
        assert Field.__eq__(res_SF, self.SF)

    def test_division(self):
        # should divide with another scalarfield if coherent axis system
        tmp_SF = self.SF.copy()
        tmp_SF.values *= 3
        res_SF = tmp_SF / self.SF
        assert np.allclose(res_SF.values, 3)
        assert np.allclose(res_SF.mask, self.SF.mask)
        assert Field.__eq__(res_SF, self.SF)
        # should divide with another array if coherent size
        values = self.SF.values.copy()
        values *= 3
        res_SF = self.SF / values
        assert np.allclose(res_SF.values, 1/3)
        assert np.allclose(res_SF.mask, self.SF.mask)
        assert Field.__eq__(res_SF, self.SF)
        values = self.SF.values.copy()
        values *= 3
        res_SF = self.SF.__rtruediv__(values)
        assert np.allclose(res_SF.values, 3)
        assert np.allclose(res_SF.mask, self.SF.mask)
        assert Field.__eq__(res_SF, self.SF)
        # # should add with cropped scalarfield if coherent axis system
        # tmp_SF = self.SF.copy().crop(intervx=[10, 50], ind=True)
        # tmp_SF.values *= 3
        # res_SF = tmp_SF / self.SF
        # assert res_SF.shape == tmp_SF.shape
        # assert res_SF.shape[0] == 41
        # assert np.allclose(res_SF.values, 3)
        # assert np.allclose(res_SF.mask, self.SF.mask)
        # assert not Field.__eq__(res_SF, self.SF)
        # should add with numbers
        tmp_SF = self.SF / 5
        assert np.allclose(tmp_SF.values, self.SF.values/5)
        assert np.allclose(tmp_SF.mask, self.SF.mask)
        assert Field.__eq__(tmp_SF, self.SF)
        tmp_SF = 5 / self.SF
        assert np.allclose(tmp_SF.values, 5/self.SF.values)
        assert np.allclose(tmp_SF.mask, self.SF.mask)
        assert Field.__eq__(tmp_SF, self.SF)
        # should add with units if coherent
        tmp_SF = self.SF / (5.2*units.m/units.s)
        assert np.allclose(tmp_SF.values, self.SF.values/5.2)
        assert tmp_SF.unit_values.strUnit() == '[]'
        assert np.allclose(tmp_SF.mask, self.SF.mask)
        assert Field.__eq__(tmp_SF, self.SF)
        tmp_SF = self.SF.__rtruediv__(5.2*units.m/units.s)
        assert np.allclose(tmp_SF.values, 5.2/self.SF.values)
        assert tmp_SF.unit_values.strUnit() == '[]'
        assert np.allclose(tmp_SF.mask, self.SF.mask)
        assert Field.__eq__(tmp_SF, self.SF)
        # shoudl raise error if incoherent axis
        tmp_SF = self.SF.copy()
        tmp_SF.axis_x += 1
        with pytest.raises(ValueError):
            tmp_SF / self.SF
        tmp_SF = self.SF.copy()
        tmp_SF.axis_y += 3.4
        with pytest.raises(ValueError):
            tmp_SF / self.SF
        # should raise an error if incoherent units
        tmp_SF = self.SF.copy()
        tmp_SF.unit_x = 'Hz'
        with pytest.raises(unum.IncompatibleUnitsError):
            tmp_SF / self.SF

    def test_multiplication(self):
        # should divide with another scalarfield if coherent axis system
        tmp_SF = self.SF.copy()
        tmp_SF.values *= 3
        res_SF = tmp_SF * self.SF
        assert np.allclose(res_SF.values, self.SF.values*3*self.SF.values)
        assert np.allclose(res_SF.mask, self.SF.mask)
        assert Field.__eq__(res_SF, self.SF)
        # should divide with another array if coherent size
        values = self.SF.values.copy()
        values *= 3
        res_SF = self.SF * values
        assert np.allclose(res_SF.values, self.SF.values**2*3)
        assert np.allclose(res_SF.mask, self.SF.mask)
        assert Field.__eq__(res_SF, self.SF)
        values = self.SF.values.copy()
        values *= 3
        res_SF = self.SF.__rmul__(values)
        assert np.allclose(res_SF.values, self.SF.values**2*3)
        assert np.allclose(res_SF.mask, self.SF.mask)
        assert Field.__eq__(res_SF, self.SF)
        # # should add with cropped scalarfield if coherent axis system
        # tmp_SF = self.SF.copy().crop(intervx=[10, 50], ind=True)
        # tmp_SF.values *= 3
        # res_SF = tmp_SF / self.SF
        # assert res_SF.shape == tmp_SF.shape
        # assert res_SF.shape[0] == 41
        # assert np.allclose(res_SF.values, 3)
        # assert np.allclose(res_SF.mask, self.SF.mask)
        # assert not Field.__eq__(res_SF, self.SF)
        # should add with numbers
        tmp_SF = self.SF * 5
        assert np.allclose(tmp_SF.values, self.SF.values*5)
        assert np.allclose(tmp_SF.mask, self.SF.mask)
        assert Field.__eq__(tmp_SF, self.SF)
        # should add with units if coherent
        tmp_SF = self.SF * 5.2*units.m
        assert np.allclose(tmp_SF.values, self.SF.values*5.2)
        assert tmp_SF.unit_values.strUnit() == '[m2/s]'
        assert np.allclose(tmp_SF.mask, self.SF.mask)
        assert Field.__eq__(tmp_SF, self.SF)
        # shoudl raise error if incoherent axis
        tmp_SF = self.SF.copy()
        tmp_SF.axis_x += 1
        with pytest.raises(ValueError):
            tmp_SF * self.SF
        tmp_SF = self.SF.copy()
        tmp_SF.axis_y += 3.4
        with pytest.raises(ValueError):
            tmp_SF * self.SF
        # should raise an error if incoherent units
        tmp_SF = self.SF.copy()
        tmp_SF.unit_x = 'Hz'
        with pytest.raises(unum.IncompatibleUnitsError):
            tmp_SF * self.SF

    def test_abs(self):
        # should return absolute
        tmp_SF = abs(self.SF)
        assert np.allclose(tmp_SF.values, abs(self.SF.values))
        assert np.allclose(tmp_SF.mask, self.SF.mask)
        assert Field.__eq__(tmp_SF, self.SF)

    def test_power(self):
        # should raise to the power
        tmp_SF = self.SF**3.14
        values = tmp_SF.values
        values[~tmp_SF.mask] = values[~tmp_SF.mask]**3.14
        tmp_SF.values = values
        assert np.allclose(tmp_SF.values[~tmp_SF.mask],
                           self.SF.values[~tmp_SF.mask]**3.14)
        assert Field.__eq__(tmp_SF, self.SF)
        # should rais error if not numbers
        with pytest.raises(TypeError):
            a = 'test'
            tmp_SF**a

    def test_iter(self):
        # should iterate on axis
        for (i, j), (x, y), val in self.SF:
            assert self.SF.values[i, j] == val

    def test_get_props(self):
        text = self.SF.get_props()
        assert text == """Shape: (98, 125)
Axis x: [0.0..13.5][m]
Axis y: [0.0..8.19892][mm]
Values: [-7.997578154671322..4.872896257509408][m/s]
Masked values: 0/12250"""

    def test_minmax_mean(self):
        # soul return min
        mini = self.SF.min
        assert mini == np.min(self.SF.values)
        # soul return max
        maxi = self.SF.max
        assert maxi == np.max(self.SF.values)
        # soul return mean
        mean = self.SF.mean
        assert mean == np.mean(self.SF.values)

    def test_get_value(self):
        # should return value at pos
        val = self.SF.get_value(4, 5)
        assert val == -6.1412866985105525
        # should return value at indice pos
        val = self.SF.get_value(4, 5, ind=True)
        assert val == 1.8386067890952045
        # should return value at grid point
        val = self.SF.get_value(self.SF.axis_x[4], 5)
        assert val == -1.637086192841739
        # should return value at grid point
        val = self.SF.get_value(4, self.SF.axis_y[9])
        assert val == -3.222703508063412
        # should return value at grid point
        val = self.SF.get_value(self.SF.axis_x[4], self.SF.axis_y[9])
        assert val == 1.2814969976054018
        # should return the unit if asked
        val = self.SF.get_value(self.SF.axis_x[4], self.SF.axis_y[9],
                                unit=True)
        assert val.asNumber() == 1.2814969976054018
        assert val.strUnit() == '[m/s]'
        # should raise an error if the point is outside
        with pytest.raises(ValueError):
            self.SF.get_value(10000, 20000)
        with pytest.raises(ValueError):
            self.SF.get_value(10000, 20000, ind=True)

    def test_profile(self):
        # TODO: add those when profile will be implemented
        pass

    def test_histogram(self):
        # TODO: add those when profile will be implemented
        pass

    def test_interpolator(self):
        # should return interpolator
        interp = self.SF.get_interpolator()
        val = interp(5, 7.2)
        assert np.isclose(val[0], 0.62262345)
        # should work for arrays
        val = interp([3, 4, 5], [3, 7.2, 3])
        assert np.allclose(val, [[-7.76660496, -6.76091338, -3.95322401],
                                 [-7.76660496, -6.76091338, -3.95322401],
                                 [-3.1907575, -2.18506593, 0.62262345]])

    def test_integrator(self):
        # should return integral
        integ, unit = self.SF.integrate()
        assert np.isclose(integ, -277.01594306920055)
        assert unit.strUnit() == '[m2.mm/s]'
        # should not integrate with masked values
        with pytest.raises(Exception):
            self.SF_mask.integrate()

    def test_scale(self):
        # should scale
        SF = self.SF
        tmp_SF = SF.scale(scalex=2.2)
        assert tmp_SF.dx == 2.2*SF.dx
        assert np.all(tmp_SF.axis_x == 2.2*SF.axis_x)
        tmp_SF = SF.scale(scaley=1.43)
        assert tmp_SF.dy == 1.43*SF.dy
        assert np.all(tmp_SF.axis_y == 1.43*SF.axis_y)
        tmp_SF = SF.scale(scalex=10, scaley=1.43)
        assert tmp_SF.dx == 10*SF.dx
        assert np.all(tmp_SF.axis_x == 10*SF.axis_x)
        assert tmp_SF.dy == 1.43*SF.dy
        assert np.all(tmp_SF.axis_y == 1.43*SF.axis_y)
        tmp_SF = self.SF.scale(scalev=5.4)
        assert np.allclose(SF.values*5.4, tmp_SF.values)
        assert Field.__eq__(tmp_SF, SF)
        # should scale inplace
        SF = self.SF
        tmp_SF = SF.copy()
        tmp_SF = SF.scale(scalex=10, scaley=1.43, scalev=5.4)
        assert tmp_SF.dx == 10*SF.dx
        assert np.all(tmp_SF.axis_x == 10*SF.axis_x)
        assert tmp_SF.dy == 1.43*SF.dy
        assert np.all(tmp_SF.axis_y == 1.43*SF.axis_y)
        assert np.allclose(SF.values*5.4, tmp_SF.values)
        # should scale with units
        SF = self.SF
        sx = -2.2*units.m
        sy = -1.43*units.Hz
        sv = -5.4*1/(units.m/units.s)
        tmp_SF = SF.scale(scalex=sx)
        assert np.isclose(tmp_SF.dx, 2.2*SF.dx)
        assert np.allclose(tmp_SF.axis_x, -2.2*SF.axis_x[::-1])
        assert tmp_SF.unit_x.strUnit() == '[m2]'
        tmp_SF = SF.scale(scaley=sy)
        assert np.isclose(tmp_SF.dy, 1.43*SF.dy)
        assert np.all(tmp_SF.axis_y == -1.43*SF.axis_y[::-1])
        assert tmp_SF.unit_y.strUnit() == '[Hz.mm]'
        tmp_SF = SF.scale(scalex=sx, scaley=sy)
        assert np.isclose(tmp_SF.dx, 2.2*SF.dx)
        assert np.all(tmp_SF.axis_x == -2.2*SF.axis_x[::-1])
        assert np.isclose(tmp_SF.dy, 1.43*SF.dy)
        assert tmp_SF.unit_y.strUnit() == '[Hz.mm]'
        assert tmp_SF.unit_x.strUnit() == '[m2]'
        assert np.allclose(tmp_SF.axis_y, -1.43*SF.axis_y[::-1])
        tmp_SF = self.SF.scale(scalev=sv)
        assert np.allclose(SF.values*-5.4, tmp_SF.values)
        assert tmp_SF.unit_values.strUnit() == '[]'
        assert Field.__eq__(tmp_SF, SF)

    def test_rotate(self):
        # should rotate
        tmp_SF = self.SF.rotate(90)
        assert np.all(tmp_SF.axis_x == -self.SF.axis_y[::-1])
        assert np.all(tmp_SF.axis_y == self.SF.axis_x)
        assert tmp_SF.values.shape[0] == self.SF.values.shape[1]
        assert tmp_SF.values.shape[1] == self.SF.values.shape[0]
        assert tmp_SF.shape[0] == self.SF.shape[1]
        assert tmp_SF.shape[1] == self.SF.shape[0]
        tmp_SF = self.SF.rotate(-90)
        assert np.all(tmp_SF.axis_x == self.SF.axis_y)
        assert np.all(tmp_SF.axis_y == -self.SF.axis_x[::-1])
        assert tmp_SF.values.shape[0] == self.SF.values.shape[1]
        assert tmp_SF.values.shape[1] == self.SF.values.shape[0]
        assert tmp_SF.shape[0] == self.SF.shape[1]
        assert tmp_SF.shape[1] == self.SF.shape[0]
        tmp_SF = self.SF.rotate(-180)
        assert np.all(tmp_SF.axis_x == -self.SF.axis_x[::-1])
        assert np.all(tmp_SF.axis_y == -self.SF.axis_y[::-1])
        assert tmp_SF.values.shape[0] == self.SF.values.shape[0]
        assert tmp_SF.values.shape[1] == self.SF.values.shape[1]
        assert tmp_SF.shape[0] == self.SF.shape[0]
        assert tmp_SF.shape[1] == self.SF.shape[1]
        # should not modify source
        save_SF = self.SF.copy()
        self.SF.rotate(90)
        assert save_SF == self.SF
        # should rotate inplace whan asked
        tmp_SF = self.SF.copy()
        tmp_SF.rotate(270, inplace=True)
        assert np.all(tmp_SF.axis_x == self.SF.axis_y)
        assert np.all(tmp_SF.axis_y == -self.SF.axis_x[::-1])
        # should raise an error if angle is not a multiple of 90
        with pytest.raises(ValueError):
            self.SF.rotate(43)

    def test_change_unit(self):
        # should change unit
        tmp_SF = self.SF.copy()
        tmp_SF.change_unit('x', 'mm')
        assert tmp_SF.unit_x.strUnit() == '[mm]'
        assert np.allclose(tmp_SF.axis_x, self.SF.axis_x*1000)
        tmp_SF = self.SF.copy()
        tmp_SF.change_unit('y', 'km')
        assert tmp_SF.unit_y.strUnit() == '[km]'
        assert np.allclose(tmp_SF.axis_y, self.SF.axis_y/1e6)
        tmp_SF = self.SF.copy()
        tmp_SF.change_unit('values', 'mm/us')
        assert tmp_SF.unit_values.strUnit() == '[mm/us]'
        assert np.allclose(tmp_SF.values, self.SF.values/1e3)
        # should not change if unit is not coherent
        with pytest.raises(unum.IncompatibleUnitsError):
            self.SF.change_unit('x', 'Hz')
        with pytest.raises(unum.IncompatibleUnitsError):
            self.SF.change_unit('values', 'Hz/J')
        # should raise an error if the arg are not strings
        with pytest.raises(TypeError):
            self.SF.change_unit(45, 'Hz')
        with pytest.raises(TypeError):
            self.SF.change_unit('x', 43)
        with pytest.raises(TypeError):
            self.SF.change_unit(45, 'Hz')
        with pytest.raises(TypeError):
            self.SF.change_unit('values', 45)

    def test_crop(self):
        # should crop
        tmp_SF = self.SF.crop(intervx=[3, 10],
                              intervy=[2, 6])
        assert tmp_SF.axis_x[0] == 3.061855670103093
        assert tmp_SF.axis_x[-1] == 9.881443298969073
        assert len(tmp_SF.axis_x) == 50
        assert len(tmp_SF.axis_y) == 60
        assert tmp_SF.values.shape[0] == 50
        assert tmp_SF.values.shape[1] == 60
        assert tmp_SF.shape[0] == 50
        assert tmp_SF.shape[1] == 60
        #  should crop with indice
        tmp_SF = self.SF.crop(intervx=[3, 30],
                              intervy=[2, 60], ind=True)
        assert len(tmp_SF.axis_x) == 28
        assert len(tmp_SF.axis_y) == 59
        assert tmp_SF.values.shape[0] == 28
        assert tmp_SF.values.shape[1] == 59
        assert tmp_SF.shape[0] == 28
        assert tmp_SF.shape[1] == 59
        assert np.allclose(tmp_SF.axis_x, self.SF.axis_x[3:31])
        assert np.allclose(tmp_SF.axis_y, self.SF.axis_y[2:61])
        #
        tmp_SF = self.SF.crop(intervx=[3, 30], ind=True)
        assert len(tmp_SF.axis_x) == 28
        assert len(tmp_SF.axis_y) == 125
        assert tmp_SF.values.shape[0] == 28
        assert tmp_SF.values.shape[1] == 125
        assert tmp_SF.shape[0] == 28
        assert tmp_SF.shape[1] == 125
        assert np.allclose(tmp_SF.axis_x, self.SF.axis_x[3:31])
        assert np.allclose(tmp_SF.axis_y, self.SF.axis_y)
        #
        tmp_SF = self.SF.crop(intervy=[2, 60], ind=True)
        assert len(tmp_SF.axis_x) == 98
        assert len(tmp_SF.axis_y) == 59
        assert tmp_SF.values.shape[0] == 98
        assert tmp_SF.values.shape[1] == 59
        assert tmp_SF.shape[0] == 98
        assert tmp_SF.shape[1] == 59
        assert np.allclose(tmp_SF.axis_x, self.SF.axis_x)
        assert np.allclose(tmp_SF.axis_y, self.SF.axis_y[2:61])
        # should modify inplace if asked
        tmp_SF = self.SF.copy()
        tmp_SF.crop(intervx=[3, 10], intervy=[2, 6], inplace=True)
        assert tmp_SF.axis_x[0] == 3.061855670103093
        assert tmp_SF.axis_x[-1] == 9.881443298969073
        assert tmp_SF.values.shape[0] == 50
        assert tmp_SF.values.shape[1] == 60
        assert tmp_SF.shape[0] == 50
        assert tmp_SF.shape[1] == 60
        assert len(tmp_SF.axis_x) == 50
        assert len(tmp_SF.axis_y) == 60
        assert tmp_SF.shape[0] == 50
        assert tmp_SF.shape[1] == 60
        # should raise error when wrong types are provided
        with pytest.raises(ValueError):
            self.SF.crop(intervx="test")
        with pytest.raises(ValueError):
            self.SF.crop(intervy="test")
        with pytest.raises(ValueError):
            self.SF.crop(intervx=[1])
        with pytest.raises(ValueError):
            self.SF.crop(intervy=[5])
        with pytest.raises(ValueError):
            self.SF.crop(intervx=[110, 24])
        with pytest.raises(ValueError):
            self.SF.crop(intervy=[50, 1])
        with pytest.raises(ValueError):
            self.SF.crop(intervx=[10000, 20000])
        with pytest.raises(ValueError):
            self.SF.crop(intervy=[10000, 20000])

    def test_extend(self):
        # should extend
        tmp_SF = self.SF.extend(5, 8, 1, 3)
        assert len(self.SF.axis_x) + 13 == len(tmp_SF.axis_x)
        assert len(self.SF.axis_y) + 4 == len(tmp_SF.axis_y)
        assert self.SF.values.shape[0] + 13 == tmp_SF.values.shape[0]
        assert self.SF.values.shape[1] + 4 == tmp_SF.values.shape[1]
        assert self.SF.mask.shape[0] + 13 == tmp_SF.mask.shape[0]
        assert self.SF.mask.shape[1] + 4 == tmp_SF.mask.shape[1]
        assert np.allclose(self.SF.axis_x, tmp_SF.axis_x[5:-8])
        assert np.allclose(self.SF.axis_y, tmp_SF.axis_y[3:-1])
        # should extend in place if asked
        tmp_SF = self.SF.copy()
        tmp_SF.extend(5, 8, 1, 3, inplace=True)
        assert len(self.SF.axis_x) + 13 == len(tmp_SF.axis_x)
        assert len(self.SF.axis_y) + 4 == len(tmp_SF.axis_y)
        assert self.SF.values.shape[0] + 13 == tmp_SF.values.shape[0]
        assert self.SF.values.shape[1] + 4 == tmp_SF.values.shape[1]
        assert self.SF.mask.shape[0] + 13 == tmp_SF.mask.shape[0]
        assert self.SF.mask.shape[1] + 4 == tmp_SF.mask.shape[1]
        assert np.allclose(self.SF.axis_x, tmp_SF.axis_x[5:-8])
        assert np.allclose(self.SF.axis_y, tmp_SF.axis_y[3:-1])

    def test_crop_masked_border(self):
        # should remove masked borders
        tmp_SF = self.SF.copy()
        mask = tmp_SF.mask
        mask[0:2, :] = True
        tmp_SF.mask = mask
        crop_SF = tmp_SF.crop_masked_border()
        assert crop_SF.shape[0] == tmp_SF.shape[0] - 2
        tmp_SF.crop_masked_border(inplace=True)
        assert self.SF.shape[0] - 2 == tmp_SF.shape[0]
        #
        tmp_SF = self.SF.copy()
        mask = tmp_SF.mask
        mask[-5::, :] = True
        tmp_SF.mask = mask
        crop_SF = tmp_SF.crop_masked_border()
        assert crop_SF.shape[0] == tmp_SF.shape[0] - 5
        tmp_SF.crop_masked_border(inplace=True)
        assert self.SF.shape[0] - 5 == tmp_SF.shape[0]
        tmp_SF = self.SF.copy()
        #
        mask = tmp_SF.mask
        mask[:, 0:2] = True
        tmp_SF.mask = mask
        crop_SF = tmp_SF.crop_masked_border()
        assert crop_SF.shape[1] == tmp_SF.shape[1] - 2
        tmp_SF.crop_masked_border(inplace=True)
        assert self.SF.shape[1] - 2 == tmp_SF.shape[1]
        #
        tmp_SF = self.SF.copy()
        mask = tmp_SF.mask
        mask[:, -5::] = True
        tmp_SF.mask = mask
        crop_SF = tmp_SF.crop_masked_border()
        assert crop_SF.shape[1] == tmp_SF.shape[1] - 5
        tmp_SF.crop_masked_border(inplace=True)
        assert self.SF.shape[1] - 5 == tmp_SF.shape[1]
        # should hard crop
        tmp_SF = self.SF.copy()
        mask = tmp_SF.mask
        mask[0:2, :] = True
        mask[3, 3] = True
        tmp_SF.mask = mask
        crop_SF = tmp_SF.crop_masked_border(hard=True)
        assert crop_SF.shape[0] == tmp_SF.shape[0] - 4
        tmp_SF.crop_masked_border(hard=True, inplace=True)
        assert self.SF.shape[0] - 4 == tmp_SF.shape[0]
        #
        tmp_SF = self.SF.copy()
        mask = tmp_SF.mask
        mask[-5::, :] = True
        mask[-6, 3] = True
        tmp_SF.mask = mask
        crop_SF = tmp_SF.crop_masked_border(hard=True)
        assert crop_SF.shape[0] == tmp_SF.shape[0] - 6
        tmp_SF.crop_masked_border(inplace=True, hard=True)
        assert self.SF.shape[0] - 6 == tmp_SF.shape[0]
        tmp_SF = self.SF.copy()
        #
        tmp_SF = self.SF.copy()
        mask = tmp_SF.mask
        mask[:, 0:2] = True
        mask[3, 2] = True
        tmp_SF.mask = mask
        crop_SF = tmp_SF.crop_masked_border(hard=True)
        assert crop_SF.shape[1] == tmp_SF.shape[1] - 3
        tmp_SF.crop_masked_border(inplace=True, hard=True)
        assert self.SF.shape[1] - 3 == tmp_SF.shape[1]
        #
        tmp_SF = self.SF.copy()
        mask = tmp_SF.mask
        mask[:, -5::] = True
        tmp_SF.mask = mask
        crop_SF = tmp_SF.crop_masked_border(hard=True)
        assert crop_SF.shape[1] == tmp_SF.shape[1] - 5
        tmp_SF.crop_masked_border(inplace=True)
        assert self.SF.shape[1] - 5 == tmp_SF.shape[1]

    def test_fill(self):
        value = 5.4
        for kind in ['linear', 'cubic', 'nearest', 'value']:
            for crop in [True, False]:
                # should fill holes
                tmp_SF = self.SF.copy()
                mask = tmp_SF.mask
                mask[4, 5] = True
                tmp_SF.mask = mask
                assert np.isnan(tmp_SF.values[4, 5])
                res_SF = tmp_SF.fill(kind=kind, value=value, crop=crop)
                assert not np.isnan(res_SF.values[4, 5])
                assert np.isnan(tmp_SF.values[4, 5])
                if kind == 'value':
                    assert res_SF.values[4, 5] == 5.4
                # should fill holes inplace
                tmp_SF = self.SF.copy()
                mask = tmp_SF.mask
                mask[4, 5] = True
                tmp_SF.mask = mask
                assert np.isnan(tmp_SF.values[4, 5])
                tmp_SF.fill(inplace=True, kind=kind, value=value, crop=crop)
                assert not np.isnan(tmp_SF.values[4, 5])
                if kind == 'value':
                    assert tmp_SF.values[4, 5] == 5.4

    def test_smooth(self):
        # should smooth
        tmp_SF = self.SF.smooth()
        assert self.SF.values[5, 5] == 1.594074929763259
        assert tmp_SF.values[5, 5] == 1.5794236449287167
        # should smooth
        tmp_SF = self.SF.smooth('gaussian')
        assert self.SF.values[5, 5] == 1.594074929763259
        assert tmp_SF.values[5, 5] == 1.5721711469265691
        # should smooth
        tmp_SF = self.SF.copy()
        tmp_SF.smooth(inplace=True)
        assert self.SF.values[5, 5] == 1.594074929763259
        assert tmp_SF.values[5, 5] == 1.5794236449287167
        # should smooth
        tmp_SF = self.SF.copy()
        tmp_SF.smooth('gaussian', inplace=True)
        assert self.SF.values[5, 5] == 1.594074929763259
        assert tmp_SF.values[5, 5] == 1.5721711469265691
        # should smooth
        tmp_SF = self.SF.smooth(size=5)
        assert self.SF.values[5, 5] == 1.594074929763259
        assert tmp_SF.values[5, 5] == 1.5502931977507135
        # should smooth
        tmp_SF = self.SF.smooth('gaussian', size=5)
        assert self.SF.values[5, 5] == 1.594074929763259
        assert tmp_SF.values[5, 5] == 1.0270509831619823
        # should smooth
        tmp_SF = self.SF.copy()
        tmp_SF.smooth(inplace=True, size=5)
        assert self.SF.values[5, 5] == 1.594074929763259
        assert tmp_SF.values[5, 5] == 1.5502931977507135
        # should smooth
        tmp_SF = self.SF.copy()
        tmp_SF.smooth('gaussian', inplace=True, size=5)
        assert self.SF.values[5, 5] == 1.594074929763259
        assert tmp_SF.values[5, 5] == 1.0270509831619823

    def test_make_evenly_spaced(self):
        tmp_SF = self.SF.make_evenly_spaced()
        assert tmp_SF.values[5, 5] == 1.5733117031856432
        tmp_SF = self.SF.copy()
        tmp_SF.make_evenly_spaced(inplace=True)
        assert tmp_SF.values[5, 5] == 1.5733117031856432

    def test_reduce_resolution(self):
        tmp_SF = self.SF.reduce_resolution(2)
        assert len(tmp_SF.axis_x) == int(len(self.SF.axis_x)/2)
        assert len(tmp_SF.axis_y) == int(len(self.SF.axis_y)/2)
        tmp_SF = self.SF.copy()
        tmp_SF.reduce_resolution(2, inplace=True)
        assert len(tmp_SF.axis_x) == int(len(self.SF.axis_x)/2)
        assert len(tmp_SF.axis_y) == int(len(self.SF.axis_y)/2)
        tmp_SF = self.SF.reduce_resolution(5)
        assert len(tmp_SF.axis_x) == int(len(self.SF.axis_x)/5)
        assert len(tmp_SF.axis_y) == int(len(self.SF.axis_y)/5)
        tmp_SF = self.SF.copy()
        tmp_SF.reduce_resolution(5, inplace=True)
        assert len(tmp_SF.axis_x) == int(len(self.SF.axis_x)/5)
        assert len(tmp_SF.axis_y) == int(len(self.SF.axis_y)/5)

    @pytest.mark.mpl_image_compare
    def test_SF_display_imshow(self):
        fig = plt.figure()
        self.SF.display(kind='imshow')
        return fig

    @pytest.mark.mpl_image_compare
    def test_SF_display_contour(self):
        fig = plt.figure()
        self.SF.display(kind='contour')
        return fig

    @pytest.mark.mpl_image_compare
    def test_SF_display_contourf(self):
        fig = plt.figure()
        self.SF.display(kind='contourf')
        return fig

    @pytest.mark.mpl_image_compare
    def test_SF_display_mask(self):
        fig = plt.figure()
        self.SF.display('mask')
        return fig
