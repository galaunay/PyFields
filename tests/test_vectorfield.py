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
from PyFields import VectorField, Field


class TestVectorField(object):
    """ Done """

    def setup(self):
        sane_parameters()
        # data
        self.x = np.linspace(0, 13.5, 98)
        self.y = np.linspace(0, 8.19892, 125)
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        self.comp_x = 3*np.cos(X) - 5*np.sin(0.43*Y)
        self.comp_y = 5.4*np.cos(.25*X) - 3*np.sin(1.43*Y)
        self.comp_x_int = np.array(3*np.cos(X) - 5*np.sin(0.43*Y), dtype=int)
        self.comp_y_int = np.array(5.4*np.cos(.25*X) - 3*np.sin(1.43*Y),
                                   dtype=int)
        self.comp_x_nans = self.comp_x.copy()
        self.comp_y_nans = self.comp_y.copy()
        self.comp_x_nans[self.comp_x > 5] = np.nan
        self.comp_y_nans[self.comp_x > 5] = np.nan
        self.mask = self.comp_x > 4
        # create vector fields
        self.VF = VectorField(axis_x=self.x,
                              axis_y=self.y,
                              mask=None,
                              comp_x=self.comp_x,
                              comp_y=self.comp_y,
                              unit_x='m',
                              unit_y='mm',
                              unit_values='m/s')
        self.VF_int = VectorField(axis_x=self.x,
                                  axis_y=self.y,
                                  mask=None,
                                  comp_x=self.comp_x_int,
                                  comp_y=self.comp_y_int,
                                  unit_x='m',
                                  unit_y='mm',
                                  unit_values='m/s')
        self.VF_mask = VectorField(axis_x=self.x,
                                   axis_y=self.y,
                                   mask=self.mask,
                                   comp_x=self.comp_x,
                                   comp_y=self.comp_y,
                                   unit_x='m',
                                   unit_y='mm',
                                   unit_values='m/s')
        self.VF_nans = VectorField(axis_x=self.x,
                                   axis_y=self.y,
                                   mask=None,
                                   comp_x=self.comp_x_nans,
                                   comp_y=self.comp_y_nans,
                                   unit_x='m',
                                   unit_y='mm',
                                   unit_values='m/s')
        self.VFs = [self.VF, self.VF_mask, self.VF_nans, self.VF_int]

    def test_init(self):
        # normal init
        VF = VectorField(self.x, self.y, self.comp_x, self.comp_y,
                         None, 'm', 'mm', 'm/s')
        assert np.all(VF.axis_x == self.x)
        assert np.all(VF.axis_y == self.y)
        assert np.all(VF.comp_x == self.comp_x)
        assert not np.all(VF.mask)
        assert VF.unit_x.strUnit() == '[m]'
        assert VF.unit_y.strUnit() == '[mm]'
        assert VF.unit_values.strUnit() == '[m/s]'
        assert VF.dtype == (np.float, np.float)
        assert VF.dx == self.x[1] - self.x[0]
        assert VF.dy == self.y[1] - self.y[0]
        assert VF.shape[0] == len(self.x)
        assert VF.shape[1] == len(self.y)
        assert VF.shape[0] == self.comp_x.shape[0]
        assert VF.shape[1] == self.comp_x.shape[1]
        assert VF._Field__is_axis_x_regular
        assert VF._Field__is_axis_y_regular
        # with mask
        VF = VectorField(self.x, self.y, self.comp_x, self.comp_y,
                         self.mask, 'm', 'mm', 'm/s')
        assert np.all(VF.axis_x == self.x)
        assert np.all(VF.axis_y == self.y)
        assert np.all(VF.comp_x[~VF.mask] == self.comp_x[~self.mask])
        assert np.all(VF.mask == self.mask)
        assert VF.unit_x.strUnit() == '[m]'
        assert VF.unit_y.strUnit() == '[mm]'
        assert VF.unit_values.strUnit() == '[m/s]'
        assert VF.dtype == (np.float, np.float)
        assert VF.dx == self.x[1] - self.x[0]
        assert VF.dy == self.y[1] - self.y[0]
        assert VF.shape[0] == len(self.x)
        assert VF.shape[1] == len(self.y)
        assert VF.shape[0] == self.comp_x.shape[0]
        assert VF.shape[1] == self.comp_x.shape[1]
        assert VF._Field__is_axis_x_regular
        assert VF._Field__is_axis_y_regular
        # with nans
        VF = VectorField(self.x, self.y, self.comp_x_nans, self.comp_y_nans,
                         None, 'm', 'mm', 'm/s')
        assert np.all(VF.axis_x == self.x)
        assert np.all(VF.axis_y == self.y)
        assert np.all(VF.comp_x == self.comp_x_nans)
        assert np.all(VF.mask == np.isnan(self.comp_x_nans))
        assert VF.unit_x.strUnit() == '[m]'
        assert VF.unit_y.strUnit() == '[mm]'
        assert VF.unit_values.strUnit() == '[m/s]'
        assert VF.dtype == (np.float, np.float)
        assert VF.dx == self.x[1] - self.x[0]
        assert VF.dy == self.y[1] - self.y[0]
        assert VF.shape[0] == len(self.x)
        assert VF.shape[1] == len(self.y)
        assert VF.shape[0] == self.comp_x.shape[0]
        assert VF.shape[1] == self.comp_x.shape[1]
        assert VF._Field__is_axis_x_regular
        assert VF._Field__is_axis_y_regular
        # with integers
        VF = VectorField(self.x, self.y,
                         self.comp_x_int, self.comp_y_int,
                         None, 'm', 'mm', 'm/s')
        assert np.all(VF.axis_x == self.x)
        assert np.all(VF.axis_y == self.y)
        assert np.all(VF.comp_x == self.comp_x_int)
        assert np.all(VF.mask == np.isnan(self.comp_x_nans))
        assert VF.unit_x.strUnit() == '[m]'
        assert VF.unit_y.strUnit() == '[mm]'
        assert VF.unit_values.strUnit() == '[m/s]'
        assert VF.dtype == (np.int, np.int)
        assert VF.dx == self.x[1] - self.x[0]
        assert VF.dy == self.y[1] - self.y[0]
        assert VF.shape[0] == len(self.x)
        assert VF.shape[1] == len(self.y)
        assert VF.shape[0] == self.comp_x.shape[0]
        assert VF.shape[1] == self.comp_x.shape[1]
        assert VF._Field__is_axis_x_regular
        assert VF._Field__is_axis_y_regular
        # should raise error if length are incoherent
        with pytest.raises(ValueError):
            VF = VectorField(self.x[1:], self.y,
                             self.comp_x_int,
                             self.comp_y_int,
                             None, 'm', 'mm', 'm/s')
        with pytest.raises(ValueError):
            VF = VectorField(self.x, self.y[0:-12],
                             self.comp_x_int,
                             self.comp_y_int,
                             None, 'm', 'mm', 'm/s')
        # should raise an error if mask is not a boolean array
        with pytest.raises(ValueError):
            VF = VectorField(self.x, self.y,
                             self.comp_x_int, self.comp_y_int,
                             [1, 2, 'not everything'], 'm', 'mm', 'm/s')

    def test_comp_x_property(self):
        #should update mask
        tmp_VF = self.VF.copy()
        val = tmp_VF.comp_x
        mask = val > 2
        val[mask] = np.nan
        tmp_VF.comp_x = val
        assert np.all(tmp_VF.mask == mask)
        # should raise error for incoherent axis sizes
        with pytest.raises(ValueError):
            tmp_VF.comp_x = val[1::, :]
        with pytest.raises(ValueError):
            tmp_VF.comp_x = val[:, :-3]

    def test_mask_property(self):
        # should raise error for incoherent axis sizes
        tmp_VF = self.VF.copy()
        val = tmp_VF.comp_x
        mask = val > 2
        with pytest.raises(ValueError):
            tmp_VF.mask = mask[1::, :]
        with pytest.raises(ValueError):
            tmp_VF.mask = mask[:, :-12]

    def test_mask_as_sf(self):
        mask = self.VF.mask_as_sf
        assert np.all(mask.values == self.VF.mask)
        assert not np.any(mask.mask)
        assert np.all(mask.axis_x == self.VF.axis_x)
        assert np.all(mask.axis_y == self.VF.axis_y)
        assert np.all(mask.unit_x == self.VF.unit_x)
        assert np.all(mask.unit_y == self.VF.unit_y)
        assert np.all(mask.unit_values.strUnit() == "[]")

    def test_equality(self):
        tmp_VF = self.VF.copy()
        # should return True if equal
        assert tmp_VF == self.VF
        assert not tmp_VF != self.VF
        # should return False if different
        tmp_VF = self.VF.copy()
        comp_x = tmp_VF.comp_x
        comp_x[0, 0] = 43
        tmp_VF.comp_x = comp_x
        assert not tmp_VF == self.VF
        assert tmp_VF != self.VF
        tmp_VF = self.VF.copy()
        tmp_VF.unit_values = "Hz"
        assert not tmp_VF == self.VF
        assert tmp_VF != self.VF
        tmp_VF = self.VF.copy()
        mask = self.mask.copy()
        mask[2, 2] = True
        tmp_VF.mask = mask
        assert tmp_VF != self.VF

    def test_addition(self):
        # should add other vectorfield if coherent axis system
        tmp_VF = self.VF.copy()
        tmp_VF.comp_x *= 3
        res_VF = tmp_VF + self.VF
        assert np.all(res_VF.comp_x == 4*self.VF.comp_x)
        assert np.all(res_VF.mask == self.VF.mask)
        assert Field.__eq__(res_VF, self.VF)
        # should add with cropped vectorfield if coherent axis system
        tmp_VF = self.VF.copy().crop(intervx=[10, 50], ind=True)
        tmp_VF.comp_x *= 3
        res_VF = tmp_VF + self.VF
        assert res_VF.shape == tmp_VF.shape
        assert res_VF.shape[0] == 41
        assert np.all(res_VF.comp_x == 4*self.VF.comp_x[10:51])
        assert np.all(res_VF.mask == self.VF.mask[10:51])
        assert not Field.__eq__(res_VF, self.VF)
        # should add with numbers
        tmp_VF = self.VF + 5
        assert np.all(tmp_VF.comp_x == 5 + self.VF.comp_x)
        assert np.all(tmp_VF.mask == self.VF.mask)
        assert Field.__eq__(tmp_VF, self.VF)
        # should add with units if coherent
        tmp_VF = self.VF + 5.2*units.m/units.s
        assert np.all(tmp_VF.comp_x == 5.2 + self.VF.comp_x)
        assert np.all(tmp_VF.mask == self.VF.mask)
        assert Field.__eq__(tmp_VF, self.VF)
        # shoudl raise error if incoherent axis
        tmp_VF = self.VF.copy()
        tmp_VF.axis_x += 1
        with pytest.raises(ValueError):
            tmp_VF + self.VF
        tmp_VF = self.VF.copy()
        tmp_VF.axis_y += 3.4
        with pytest.raises(ValueError):
            tmp_VF + self.VF
        # should raise an error if incoherent units
        tmp_VF = self.VF.copy()
        tmp_VF.unit_x = 'Hz'
        with pytest.raises(unum.IncompatibleUnitsError):
            tmp_VF + self.VF

    def test_substitutaion(self):
        # should add other vectorfield if coherent axis system
        tmp_VF = self.VF.copy()
        tmp_VF.comp_x *= 3
        res_VF = tmp_VF - self.VF
        assert np.allclose(res_VF.comp_x, 2*self.VF.comp_x)
        assert np.all(res_VF.mask == 2*self.VF.mask)
        assert Field.__eq__(res_VF, self.VF)

    def test_division(self):
        # should divide with another vectorfield if coherent axis system
        tmp_VF = self.VF.copy()
        tmp_VF.comp_x *= 3
        res_VF = tmp_VF / self.VF
        assert np.allclose(res_VF.comp_x, 3)
        assert np.allclose(res_VF.mask, self.VF.mask)
        assert Field.__eq__(res_VF, self.VF)
        # should divide with another array if coherent size
        comp_x = self.VF.comp_x.copy()
        comp_x *= 3
        res_VF = self.VF / comp_x
        assert np.allclose(res_VF.comp_x, 1/3)
        assert np.allclose(res_VF.mask, self.VF.mask)
        assert Field.__eq__(res_VF, self.VF)
        comp_x = self.VF.comp_x.copy()
        comp_x *= 3
        res_VF = self.VF.__rtruediv__(comp_x)
        assert np.allclose(res_VF.comp_x, 3)
        assert np.allclose(res_VF.mask, self.VF.mask)
        assert Field.__eq__(res_VF, self.VF)
        # # should add with cropped vectorfield if coherent axis system
        # tmp_VF = self.VF.copy().crop(intervx=[10, 50], ind=True)
        # tmp_VF.comp_x *= 3
        # res_VF = tmp_VF / self.VF
        # assert res_VF.shape == tmp_VF.shape
        # assert res_VF.shape[0] == 41
        # assert np.allclose(res_VF.comp_x, 3)
        # assert np.allclose(res_VF.mask, self.VF.mask)
        # assert not Field.__eq__(res_VF, self.VF)
        # should add with numbers
        tmp_VF = self.VF / 5
        assert np.allclose(tmp_VF.comp_x, self.VF.comp_x/5)
        assert np.allclose(tmp_VF.mask, self.VF.mask)
        assert Field.__eq__(tmp_VF, self.VF)
        tmp_VF = 5 / self.VF
        assert np.allclose(tmp_VF.comp_x, 5/self.VF.comp_x)
        assert np.allclose(tmp_VF.mask, self.VF.mask)
        assert Field.__eq__(tmp_VF, self.VF)
        # should add with units if coherent
        tmp_VF = self.VF / (5.2*units.m/units.s)
        assert np.allclose(tmp_VF.comp_x, self.VF.comp_x/5.2)
        assert tmp_VF.unit_values.strUnit() == '[]'
        assert np.allclose(tmp_VF.mask, self.VF.mask)
        assert Field.__eq__(tmp_VF, self.VF)
        tmp_VF = self.VF.__rtruediv__(5.2*units.m/units.s)
        assert np.allclose(tmp_VF.comp_x, 5.2/self.VF.comp_x)
        assert tmp_VF.unit_values.strUnit() == '[]'
        assert np.allclose(tmp_VF.mask, self.VF.mask)
        assert Field.__eq__(tmp_VF, self.VF)
        # shoudl raise error if incoherent axis
        tmp_VF = self.VF.copy()
        tmp_VF.axis_x += 1
        with pytest.raises(ValueError):
            tmp_VF / self.VF
        tmp_VF = self.VF.copy()
        tmp_VF.axis_y += 3.4
        with pytest.raises(ValueError):
            tmp_VF / self.VF
        # should raise an error if incoherent units
        tmp_VF = self.VF.copy()
        tmp_VF.unit_x = 'Hz'
        with pytest.raises(unum.IncompatibleUnitsError):
            tmp_VF / self.VF

    def test_multiplication(self):
        # should divide with another vectorfield if coherent axis system
        tmp_VF = self.VF.copy()
        tmp_VF.comp_x *= 3
        res_VF = tmp_VF * self.VF
        assert np.allclose(res_VF.comp_x, self.VF.comp_x*3*self.VF.comp_x)
        assert np.allclose(res_VF.mask, self.VF.mask)
        assert Field.__eq__(res_VF, self.VF)
        # should divide with another array if coherent size
        comp_x = self.VF.comp_x.copy()
        comp_x *= 3
        res_VF = self.VF * comp_x
        assert np.allclose(res_VF.comp_x, self.VF.comp_x**2*3)
        assert np.allclose(res_VF.mask, self.VF.mask)
        assert Field.__eq__(res_VF, self.VF)
        comp_x = self.VF.comp_x.copy()
        comp_x *= 3
        res_VF = self.VF.__rmul__(comp_x)
        assert np.allclose(res_VF.comp_x, self.VF.comp_x**2*3)
        assert np.allclose(res_VF.mask, self.VF.mask)
        assert Field.__eq__(res_VF, self.VF)
        # # should add with cropped vectorfield if coherent axis system
        # tmp_VF = self.VF.copy().crop(intervx=[10, 50], ind=True)
        # tmp_VF.comp_x *= 3
        # res_VF = tmp_VF / self.VF
        # assert res_VF.shape == tmp_VF.shape
        # assert res_VF.shape[0] == 41
        # assert np.allclose(res_VF.comp_x, 3)
        # assert np.allclose(res_VF.mask, self.VF.mask)
        # assert not Field.__eq__(res_VF, self.VF)
        # should add with numbers
        tmp_VF = self.VF * 5
        assert np.allclose(tmp_VF.comp_x, self.VF.comp_x*5)
        assert np.allclose(tmp_VF.mask, self.VF.mask)
        assert Field.__eq__(tmp_VF, self.VF)
        # should add with units if coherent
        tmp_VF = self.VF * 5.2*units.m
        assert np.allclose(tmp_VF.comp_x, self.VF.comp_x*5.2)
        assert tmp_VF.unit_values.strUnit() == '[m2/s]'
        assert np.allclose(tmp_VF.mask, self.VF.mask)
        assert Field.__eq__(tmp_VF, self.VF)
        # shoudl raise error if incoherent axis
        tmp_VF = self.VF.copy()
        tmp_VF.axis_x += 1
        with pytest.raises(ValueError):
            tmp_VF * self.VF
        tmp_VF = self.VF.copy()
        tmp_VF.axis_y += 3.4
        with pytest.raises(ValueError):
            tmp_VF * self.VF
        # should raise an error if incoherent units
        tmp_VF = self.VF.copy()
        tmp_VF.unit_x = 'Hz'
        with pytest.raises(unum.IncompatibleUnitsError):
            tmp_VF * self.VF

    def test_abs(self):
        # should return absolute
        tmp_VF = abs(self.VF)
        assert np.allclose(tmp_VF.comp_x, abs(self.VF.comp_x))
        assert np.allclose(tmp_VF.mask, self.VF.mask)
        assert Field.__eq__(tmp_VF, self.VF)

    def test_power(self):
        # should raise to the power
        tmp_VF = self.VF**3.14
        comp_x = tmp_VF.comp_x
        comp_x[~tmp_VF.mask] = comp_x[~tmp_VF.mask]**3.14
        tmp_VF.comp_x = comp_x
        assert np.allclose(tmp_VF.comp_x[~tmp_VF.mask],
                           self.VF.comp_x[~tmp_VF.mask]**3.14)
        assert Field.__eq__(tmp_VF, self.VF)
        # should rais error if not numbers
        with pytest.raises(TypeError):
            a = 'test'
            tmp_VF**a

    def test_iter(self):
        # should iterate on axis
        for (i, j), (x, y), valx, valy in self.VF:
            assert self.VF.comp_x[i, j] == valx
            assert self.VF.comp_y[i, j] == valy

    def test_get_props(self):
        text = self.VF.get_props()
        assert text == """Shape: (98, 125)
Axe x: [0.0..13.5][m]
Axe y: [0.0..8.19892][mm]
Comp x: [-7.997578154671322..4.872896257509408][m/s]
Comp y: [-8.399664948682865..8.39965280416321][m/s]
Masked values: 0/12250"""

    def test_minmax_mean(self):
        # soul return min
        mini = self.VF.min
        assert mini == np.min(self.VF.magnitude)
        # soul return max
        maxi = self.VF.max
        assert maxi == np.max(self.VF.magnitude)
        # soul return mean
        mean = self.VF.mean
        assert mean == np.mean(self.VF.magnitude)

    def test_get_value(self):
        # should return value at pos
        valx, valy = self.VF.get_value(4, 5)
        assert valx == -6.1412866985105525
        # should return value at indice pos
        valx, valy = self.VF.get_value(4, 5, ind=True)
        assert valx == 1.8386067890952045
        # should return value at grid point
        valx, valy = self.VF.get_value(self.VF.axis_x[4], 5)
        assert valx == -1.637086192841739
        # should return value at grid point
        valx, valy = self.VF.get_value(4, self.VF.axis_y[9])
        assert valx == -3.222703508063412
        # should return value at grid point
        valx, valy = self.VF.get_value(self.VF.axis_x[4], self.VF.axis_y[9])
        assert valx == 1.2814969976054018
        # should return the unit if asked
        valx, valy = self.VF.get_value(self.VF.axis_x[4], self.VF.axis_y[9],
                                       unit=True)
        assert valx.asNumber() == 1.2814969976054018
        assert valx.strUnit() == '[m/s]'
        # should raise an error if the point is outside
        with pytest.raises(ValueError):
            self.VF.get_value(10000, 20000)
        with pytest.raises(ValueError):
            self.VF.get_value(10000, 20000, ind=True)

    def test_profile(self):
        # TODO: add those when profile will be implemented
        pass

    def test_histogram(self):
        # TODO: add those when profile will be implemented
        pass

    def test_interpolators(self):
        # should return interpolator
        interpx, interpy = self.VF.get_interpolators()
        valx = interpx(5, 7.2)
        valy = interpy(5, 7.2)
        assert np.isclose(valx[0], 0.62262345)
        assert np.isclose(valy[0], 3.99705280)
        # should work for arrays
        valx = interpx([3, 4, 5], [3, 7.2, 3])
        valy = interpy([3, 4, 5], [3, 7.2, 3])
        assert np.allclose(valx, [[-7.76660496, -6.76091338, -3.95322401],
                                  [-7.76660496, -6.76091338, -3.95322401],
                                  [-3.1907575, -2.18506593, 0.62262345]])
        assert np.allclose(valy, [[6.6839986, 5.65076028, 4.4361379],
                                  [6.6839986, 5.65076028, 4.4361379],
                                  [6.24491351, 5.2116752, 3.99705281]])

    def test_scale(self):
        # should scale
        VF = self.VF
        tmp_VF = VF.scale(scalex=2.2)
        assert tmp_VF.dx == 2.2*VF.dx
        assert np.all(tmp_VF.axis_x == 2.2*VF.axis_x)
        tmp_VF = VF.scale(scaley=1.43)
        assert tmp_VF.dy == 1.43*VF.dy
        assert np.all(tmp_VF.axis_y == 1.43*VF.axis_y)
        tmp_VF = VF.scale(scalex=10, scaley=1.43)
        assert tmp_VF.dx == 10*VF.dx
        assert np.all(tmp_VF.axis_x == 10*VF.axis_x)
        assert tmp_VF.dy == 1.43*VF.dy
        assert np.all(tmp_VF.axis_y == 1.43*VF.axis_y)
        tmp_VF = self.VF.scale(scalev=5.4)
        assert np.allclose(VF.comp_x*5.4, tmp_VF.comp_x)
        assert Field.__eq__(tmp_VF, VF)
        # should scale inplace
        VF = self.VF
        tmp_VF = VF.copy()
        tmp_VF = VF.scale(scalex=10, scaley=1.43, scalev=5.4)
        assert tmp_VF.dx == 10*VF.dx
        assert np.all(tmp_VF.axis_x == 10*VF.axis_x)
        assert tmp_VF.dy == 1.43*VF.dy
        assert np.all(tmp_VF.axis_y == 1.43*VF.axis_y)
        assert np.allclose(VF.comp_x*5.4, tmp_VF.comp_x)
        # should scale with units
        VF = self.VF
        sx = -2.2*units.m
        sy = -1.43*units.Hz
        sv = -5.4*1/(units.m/units.s)
        tmp_VF = VF.scale(scalex=sx)
        assert np.isclose(tmp_VF.dx, 2.2*VF.dx)
        assert np.allclose(tmp_VF.axis_x, -2.2*VF.axis_x[::-1])
        assert tmp_VF.unit_x.strUnit() == '[m2]'
        tmp_VF = VF.scale(scaley=sy)
        assert np.isclose(tmp_VF.dy, 1.43*VF.dy)
        assert np.all(tmp_VF.axis_y == -1.43*VF.axis_y[::-1])
        assert tmp_VF.unit_y.strUnit() == '[Hz.mm]'
        tmp_VF = VF.scale(scalex=sx, scaley=sy)
        assert np.isclose(tmp_VF.dx, 2.2*VF.dx)
        assert np.all(tmp_VF.axis_x == -2.2*VF.axis_x[::-1])
        assert np.isclose(tmp_VF.dy, 1.43*VF.dy)
        assert tmp_VF.unit_y.strUnit() == '[Hz.mm]'
        assert tmp_VF.unit_x.strUnit() == '[m2]'
        assert np.allclose(tmp_VF.axis_y, -1.43*VF.axis_y[::-1])
        tmp_VF = self.VF.scale(scalev=sv)
        assert np.allclose(VF.comp_x*-5.4, tmp_VF.comp_x)
        assert tmp_VF.unit_values.strUnit() == '[]'
        assert Field.__eq__(tmp_VF, VF)

    def test_rotate(self):
        # should rotate
        tmp_VF = self.VF.rotate(90)
        assert np.all(tmp_VF.axis_x == -self.VF.axis_y[::-1])
        assert np.all(tmp_VF.axis_y == self.VF.axis_x)
        assert tmp_VF.comp_x.shape[0] == self.VF.comp_x.shape[1]
        assert tmp_VF.comp_x.shape[1] == self.VF.comp_x.shape[0]
        assert tmp_VF.shape[0] == self.VF.shape[1]
        assert tmp_VF.shape[1] == self.VF.shape[0]
        tmp_VF = self.VF.rotate(-90)
        assert np.all(tmp_VF.axis_x == self.VF.axis_y)
        assert np.all(tmp_VF.axis_y == -self.VF.axis_x[::-1])
        assert tmp_VF.comp_x.shape[0] == self.VF.comp_x.shape[1]
        assert tmp_VF.comp_x.shape[1] == self.VF.comp_x.shape[0]
        assert tmp_VF.shape[0] == self.VF.shape[1]
        assert tmp_VF.shape[1] == self.VF.shape[0]
        tmp_VF = self.VF.rotate(-180)
        assert np.all(tmp_VF.axis_x == -self.VF.axis_x[::-1])
        assert np.all(tmp_VF.axis_y == -self.VF.axis_y[::-1])
        assert tmp_VF.comp_x.shape[0] == self.VF.comp_x.shape[0]
        assert tmp_VF.comp_x.shape[1] == self.VF.comp_x.shape[1]
        assert tmp_VF.shape[0] == self.VF.shape[0]
        assert tmp_VF.shape[1] == self.VF.shape[1]
        # should not modify source
        save_VF = self.VF.copy()
        self.VF.rotate(90)
        assert save_VF == self.VF
        # should rotate inplace whan asked
        tmp_VF = self.VF.copy()
        tmp_VF.rotate(270, inplace=True)
        assert np.all(tmp_VF.axis_x == self.VF.axis_y)
        assert np.all(tmp_VF.axis_y == -self.VF.axis_x[::-1])
        # should raise an error if angle is not a multiple of 90
        with pytest.raises(ValueError):
            self.VF.rotate(43)

    def test_change_unit(self):
        # should change unit
        tmp_VF = self.VF.copy()
        tmp_VF.change_unit('x', 'mm')
        assert tmp_VF.unit_x.strUnit() == '[mm]'
        assert np.allclose(tmp_VF.axis_x, self.VF.axis_x*1000)
        tmp_VF = self.VF.copy()
        tmp_VF.change_unit('y', 'km')
        assert tmp_VF.unit_y.strUnit() == '[km]'
        assert np.allclose(tmp_VF.axis_y, self.VF.axis_y/1e6)
        tmp_VF = self.VF.copy()
        tmp_VF.change_unit('values', 'mm/us')
        assert tmp_VF.unit_values.strUnit() == '[mm/us]'
        assert np.allclose(tmp_VF.comp_x, self.VF.comp_x/1e3)
        # should not change if unit is not coherent
        with pytest.raises(unum.IncompatibleUnitsError):
            self.VF.change_unit('x', 'Hz')
        with pytest.raises(unum.IncompatibleUnitsError):
            self.VF.change_unit('values', 'Hz/J')
        # should raise an error if the arg are not strings
        with pytest.raises(TypeError):
            self.VF.change_unit(45, 'Hz')
        with pytest.raises(TypeError):
            self.VF.change_unit('x', 43)
        with pytest.raises(TypeError):
            self.VF.change_unit(45, 'Hz')
        with pytest.raises(TypeError):
            self.VF.change_unit('values', 45)

    def test_crop(self):
        # should crop
        tmp_VF = self.VF.crop(intervx=[3, 10],
                              intervy=[2, 6])
        assert tmp_VF.axis_x[0] == 3.061855670103093
        assert tmp_VF.axis_x[-1] == 9.881443298969073
        assert len(tmp_VF.axis_x) == 50
        assert len(tmp_VF.axis_y) == 60
        assert tmp_VF.comp_x.shape[0] == 50
        assert tmp_VF.comp_x.shape[1] == 60
        assert tmp_VF.shape[0] == 50
        assert tmp_VF.shape[1] == 60
        #  should crop with indice
        tmp_VF = self.VF.crop(intervx=[3, 30],
                              intervy=[2, 60], ind=True)
        assert len(tmp_VF.axis_x) == 28
        assert len(tmp_VF.axis_y) == 59
        assert tmp_VF.comp_x.shape[0] == 28
        assert tmp_VF.comp_x.shape[1] == 59
        assert tmp_VF.shape[0] == 28
        assert tmp_VF.shape[1] == 59
        assert np.allclose(tmp_VF.axis_x, self.VF.axis_x[3:31])
        assert np.allclose(tmp_VF.axis_y, self.VF.axis_y[2:61])
        #
        tmp_VF = self.VF.crop(intervx=[3, 30], ind=True)
        assert len(tmp_VF.axis_x) == 28
        assert len(tmp_VF.axis_y) == 125
        assert tmp_VF.comp_x.shape[0] == 28
        assert tmp_VF.comp_x.shape[1] == 125
        assert tmp_VF.shape[0] == 28
        assert tmp_VF.shape[1] == 125
        assert np.allclose(tmp_VF.axis_x, self.VF.axis_x[3:31])
        assert np.allclose(tmp_VF.axis_y, self.VF.axis_y)
        #
        tmp_VF = self.VF.crop(intervy=[2, 60], ind=True)
        assert len(tmp_VF.axis_x) == 98
        assert len(tmp_VF.axis_y) == 59
        assert tmp_VF.comp_x.shape[0] == 98
        assert tmp_VF.comp_x.shape[1] == 59
        assert tmp_VF.shape[0] == 98
        assert tmp_VF.shape[1] == 59
        assert np.allclose(tmp_VF.axis_x, self.VF.axis_x)
        assert np.allclose(tmp_VF.axis_y, self.VF.axis_y[2:61])
        # should modify inplace if asked
        tmp_VF = self.VF.copy()
        tmp_VF.crop(intervx=[3, 10], intervy=[2, 6], inplace=True)
        assert tmp_VF.axis_x[0] == 3.061855670103093
        assert tmp_VF.axis_x[-1] == 9.881443298969073
        assert tmp_VF.comp_x.shape[0] == 50
        assert tmp_VF.comp_x.shape[1] == 60
        assert tmp_VF.shape[0] == 50
        assert tmp_VF.shape[1] == 60
        assert len(tmp_VF.axis_x) == 50
        assert len(tmp_VF.axis_y) == 60
        assert tmp_VF.shape[0] == 50
        assert tmp_VF.shape[1] == 60
        # should raise error when wrong types are provided
        with pytest.raises(ValueError):
            self.VF.crop(intervx="test")
        with pytest.raises(ValueError):
            self.VF.crop(intervy="test")
        with pytest.raises(ValueError):
            self.VF.crop(intervx=[1])
        with pytest.raises(ValueError):
            self.VF.crop(intervy=[5])
        with pytest.raises(ValueError):
            self.VF.crop(intervx=[110, 24])
        with pytest.raises(ValueError):
            self.VF.crop(intervy=[50, 1])
        with pytest.raises(ValueError):
            self.VF.crop(intervx=[10000, 20000])
        with pytest.raises(ValueError):
            self.VF.crop(intervy=[10000, 20000])

    def test_extend(self):
        # should extend
        tmp_VF = self.VF.extend(5, 8, 1, 3)
        assert len(self.VF.axis_x) + 13 == len(tmp_VF.axis_x)
        assert len(self.VF.axis_y) + 4 == len(tmp_VF.axis_y)
        assert self.VF.comp_x.shape[0] + 13 == tmp_VF.comp_x.shape[0]
        assert self.VF.comp_x.shape[1] + 4 == tmp_VF.comp_x.shape[1]
        assert self.VF.mask.shape[0] + 13 == tmp_VF.mask.shape[0]
        assert self.VF.mask.shape[1] + 4 == tmp_VF.mask.shape[1]
        assert np.allclose(self.VF.axis_x, tmp_VF.axis_x[5:-8])
        assert np.allclose(self.VF.axis_y, tmp_VF.axis_y[3:-1])
        # should extend in place if asked
        tmp_VF = self.VF.copy()
        tmp_VF.extend(5, 8, 1, 3, inplace=True)
        assert len(self.VF.axis_x) + 13 == len(tmp_VF.axis_x)
        assert len(self.VF.axis_y) + 4 == len(tmp_VF.axis_y)
        assert self.VF.comp_x.shape[0] + 13 == tmp_VF.comp_x.shape[0]
        assert self.VF.comp_x.shape[1] + 4 == tmp_VF.comp_x.shape[1]
        assert self.VF.mask.shape[0] + 13 == tmp_VF.mask.shape[0]
        assert self.VF.mask.shape[1] + 4 == tmp_VF.mask.shape[1]
        assert np.allclose(self.VF.axis_x, tmp_VF.axis_x[5:-8])
        assert np.allclose(self.VF.axis_y, tmp_VF.axis_y[3:-1])

    def test_crop_masked_border(self):
        # should remove masked borders
        tmp_VF = self.VF.copy()
        mask = tmp_VF.mask
        mask[0:2, :] = True
        tmp_VF.mask = mask
        crop_VF = tmp_VF.crop_masked_border()
        assert crop_VF.shape[0] == tmp_VF.shape[0] - 2
        tmp_VF.crop_masked_border(inplace=True)
        assert self.VF.shape[0] - 2 == tmp_VF.shape[0]
        #
        tmp_VF = self.VF.copy()
        mask = tmp_VF.mask
        mask[-5::, :] = True
        tmp_VF.mask = mask
        crop_VF = tmp_VF.crop_masked_border()
        assert crop_VF.shape[0] == tmp_VF.shape[0] - 5
        tmp_VF.crop_masked_border(inplace=True)
        assert self.VF.shape[0] - 5 == tmp_VF.shape[0]
        tmp_VF = self.VF.copy()
        #
        mask = tmp_VF.mask
        mask[:, 0:2] = True
        tmp_VF.mask = mask
        crop_VF = tmp_VF.crop_masked_border()
        assert crop_VF.shape[1] == tmp_VF.shape[1] - 2
        tmp_VF.crop_masked_border(inplace=True)
        assert self.VF.shape[1] - 2 == tmp_VF.shape[1]
        #
        tmp_VF = self.VF.copy()
        mask = tmp_VF.mask
        mask[:, -5::] = True
        tmp_VF.mask = mask
        crop_VF = tmp_VF.crop_masked_border()
        assert crop_VF.shape[1] == tmp_VF.shape[1] - 5
        tmp_VF.crop_masked_border(inplace=True)
        assert self.VF.shape[1] - 5 == tmp_VF.shape[1]
        # should hard crop
        tmp_VF = self.VF.copy()
        mask = tmp_VF.mask
        mask[0:2, :] = True
        mask[3, 3] = True
        tmp_VF.mask = mask
        crop_VF = tmp_VF.crop_masked_border(hard=True)
        assert crop_VF.shape[0] == tmp_VF.shape[0] - 4
        tmp_VF.crop_masked_border(hard=True, inplace=True)
        assert self.VF.shape[0] - 4 == tmp_VF.shape[0]
        #
        tmp_VF = self.VF.copy()
        mask = tmp_VF.mask
        mask[-5::, :] = True
        mask[-6, 3] = True
        tmp_VF.mask = mask
        crop_VF = tmp_VF.crop_masked_border(hard=True)
        assert crop_VF.shape[0] == tmp_VF.shape[0] - 6
        tmp_VF.crop_masked_border(inplace=True, hard=True)
        assert self.VF.shape[0] - 6 == tmp_VF.shape[0]
        tmp_VF = self.VF.copy()
        #
        tmp_VF = self.VF.copy()
        mask = tmp_VF.mask
        mask[:, 0:2] = True
        mask[3, 2] = True
        tmp_VF.mask = mask
        crop_VF = tmp_VF.crop_masked_border(hard=True)
        assert crop_VF.shape[1] == tmp_VF.shape[1] - 3
        tmp_VF.crop_masked_border(inplace=True, hard=True)
        assert self.VF.shape[1] - 3 == tmp_VF.shape[1]
        #
        tmp_VF = self.VF.copy()
        mask = tmp_VF.mask
        mask[:, -5::] = True
        tmp_VF.mask = mask
        crop_VF = tmp_VF.crop_masked_border(hard=True)
        assert crop_VF.shape[1] == tmp_VF.shape[1] - 5
        tmp_VF.crop_masked_border(inplace=True)
        assert self.VF.shape[1] - 5 == tmp_VF.shape[1]

    def test_fill(self):
        value = 5.4
        for kind in ['linear', 'cubic', 'nearest', 'value']:
            for crop in [True, False]:
                # should fill holes
                tmp_VF = self.VF.copy()
                mask = tmp_VF.mask
                mask[4, 5] = True
                tmp_VF.mask = mask
                assert np.isnan(tmp_VF.comp_x[4, 5])
                res_VF = tmp_VF.fill(kind=kind, value=value, crop=crop)
                assert not np.isnan(res_VF.comp_x[4, 5])
                assert np.isnan(tmp_VF.comp_x[4, 5])
                if kind == 'value':
                    assert res_VF.comp_x[4, 5] == 5.4
                # should fill holes inplace
                tmp_VF = self.VF.copy()
                mask = tmp_VF.mask
                mask[4, 5] = True
                tmp_VF.mask = mask
                assert np.isnan(tmp_VF.comp_x[4, 5])
                tmp_VF.fill(inplace=True, kind=kind, value=value, crop=crop)
                assert not np.isnan(tmp_VF.comp_x[4, 5])
                if kind == 'value':
                    assert tmp_VF.comp_x[4, 5] == 5.4

    def test_smooth(self):
        # should smooth
        tmp_VF = self.VF.smooth()
        assert self.VF.comp_x[5, 5] == 1.594074929763259
        assert tmp_VF.comp_x[5, 5] == 1.5794236449287167
        # should smooth
        tmp_VF = self.VF.smooth('gaussian')
        assert self.VF.comp_x[5, 5] == 1.594074929763259
        assert tmp_VF.comp_x[5, 5] == 1.5721711469265691
        # should smooth
        tmp_VF = self.VF.copy()
        tmp_VF.smooth(inplace=True)
        assert self.VF.comp_x[5, 5] == 1.594074929763259
        assert tmp_VF.comp_x[5, 5] == 1.5794236449287167
        # should smooth
        tmp_VF = self.VF.copy()
        tmp_VF.smooth('gaussian', inplace=True)
        assert self.VF.comp_x[5, 5] == 1.594074929763259
        assert tmp_VF.comp_x[5, 5] == 1.5721711469265691
        # should smooth
        tmp_VF = self.VF.smooth(size=5)
        assert self.VF.comp_x[5, 5] == 1.594074929763259
        assert tmp_VF.comp_x[5, 5] == 1.5502931977507135
        # should smooth
        tmp_VF = self.VF.smooth('gaussian', size=5)
        assert self.VF.comp_x[5, 5] == 1.594074929763259
        assert tmp_VF.comp_x[5, 5] == 1.0270509831619823
        # should smooth
        tmp_VF = self.VF.copy()
        tmp_VF.smooth(inplace=True, size=5)
        assert self.VF.comp_x[5, 5] == 1.594074929763259
        assert tmp_VF.comp_x[5, 5] == 1.5502931977507135
        # should smooth
        tmp_VF = self.VF.copy()
        tmp_VF.smooth('gaussian', inplace=True, size=5)
        assert self.VF.comp_x[5, 5] == 1.594074929763259
        assert tmp_VF.comp_x[5, 5] == 1.0270509831619823

    def test_make_evenly_spaced(self):
        tmp_VF = self.VF.make_evenly_spaced()
        assert tmp_VF.comp_x[5, 5] == 1.5733117031856432
        tmp_VF = self.VF.copy()
        tmp_VF.make_evenly_spaced(inplace=True)
        assert tmp_VF.comp_x[5, 5] == 1.5733117031856432

    def test_reduce_resolution(self):
        tmp_VF = self.VF.reduce_resolution(2)
        assert len(tmp_VF.axis_x) == int(len(self.VF.axis_x)/2)
        assert len(tmp_VF.axis_y) == int(len(self.VF.axis_y)/2)
        tmp_VF = self.VF.copy()
        tmp_VF.reduce_resolution(2, inplace=True)
        assert len(tmp_VF.axis_x) == int(len(self.VF.axis_x)/2)
        assert len(tmp_VF.axis_y) == int(len(self.VF.axis_y)/2)
        tmp_VF = self.VF.reduce_resolution(5)
        assert len(tmp_VF.axis_x) == int(len(self.VF.axis_x)/5)
        assert len(tmp_VF.axis_y) == int(len(self.VF.axis_y)/5)
        tmp_VF = self.VF.copy()
        tmp_VF.reduce_resolution(5, inplace=True)
        assert len(tmp_VF.axis_x) == int(len(self.VF.axis_x)/5)
        assert len(tmp_VF.axis_y) == int(len(self.VF.axis_y)/5)

    @pytest.mark.mpl_image_compare
    def test_VF_display_imshow(self):
        fig = plt.figure()
        self.VF.display(component='x', kind='imshow')
        return fig

    @pytest.mark.mpl_image_compare
    def test_VF_display_contour(self):
        fig = plt.figure()
        self.VF.display(component='y', kind='contour')
        return fig

    @pytest.mark.mpl_image_compare
    def test_VF_display_contourf(self):
        fig = plt.figure()
        self.VF.display(component='magnitude', kind='contourf')
        return fig

    @pytest.mark.mpl_image_compare
    def test_VF_display_mask(self):
        fig = plt.figure()
        self.VF.display(component='mask')
        return fig

    @pytest.mark.mpl_image_compare
    def test_VF_display_quiver(self):
        fig = plt.figure()
        self.VF.display(kind='quiver')
        return fig

    @pytest.mark.mpl_image_compare
    def test_VF_display_stream(self):
        fig = plt.figure()
        self.VF.display(kind='stream')
        return fig

    @pytest.mark.mpl_image_compare
    def test_VF_display_magnitude(self):
        fig = plt.figure()
        self.VF.display('magnitude', kind='imshow')
        return fig
