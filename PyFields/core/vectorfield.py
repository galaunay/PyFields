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

import copy

import matplotlib.pyplot as plt
import numpy as np

import unum
from . import field as fld, scalarfield as sf
from ..utils import make_unit

# TODO: check for remaining 'ScalarField'


class VectorField(fld.Field):
    """
    Class representing a vector field (2D field, with two components on each
    point).
    """

    def __init__(self, axis_x, axis_y, comp_x, comp_y, mask=None,
                 unit_x="", unit_y="", unit_values=""):
        """
        Object representing a vector field (2D field, with two component on
        each point).

        Parameters
        ----------
        axis_x : nx1 array
            x axis values.
        axis_y : mx1 array
            y axis values.
        comp_x : array or masked array
            Values of the x component at the axis points
        comp_y : array or masked array
            Values of the y component at the axis points
        unit_x, unit_y : String or Unum object, optional
            x and y axis units.
        unit_values : String or Unum object, optional
            Field values unit.
        """
        # build axis system
        fld.Field.__init__(self, axis_x=axis_x, axis_y=axis_y,
                           unit_x=unit_x, unit_y=unit_y)
        self.__mask = None
        self.__comp_x = None
        self.__comp_y = None
        # store values properties
        self.comp_x = np.array(comp_x)
        self.comp_y = np.array(comp_y)
        if self.comp_y.shape != self.comp_x.shape:
            raise ValueError("'comp_x' and 'comp_y' should have the"
                             " same dimensions")
        if self.comp_x.shape[0] != len(self.axis_x) or \
           self.comp_x.shape[1] != len(self.axis_y):
            raise ValueError('Incoherent shapes, axis sizes are {}, {},'
                             'but comp_x size is {}'
                             .format(len(self.axis_x), len(self.axis_y),
                                     self.comp_x.shape))
        self.unit_values = unit_values
        # store mask (if necessary)
        if mask is None:
            nans = np.logical_or(np.isnan(comp_x), np.isnan(comp_y))
            if np.any(nans):
                mask = nans
            else:
                mask = False
        else:
            try:
                mask = np.array(mask, dtype=bool)
            except TypeError:
                raise TypeError("'mask' should be a boolean or an array of"
                                "boolean, but is currently {}"
                                .format(mask))
            if mask.shape != () and np.any(mask.shape != self.shape):
                raise ValueError("'mask' should be of the same size as the"
                                 " axis system: {},"
                                 " but is currently of size {}"
                                 .format(self.shape, mask.shape))
            nans = np.logical_or(np.isnan(comp_x), np.isnan(comp_y))
            mask = np.logical_or(mask, nans)
        self.mask = mask

    def __eq__(self, another):
        if not isinstance(another, VectorField):
            return False
        if not super().__eq__(another):
            return False
        if not np.all(self.mask == another.mask):
            return False
        if not np.all(self.comp_x[~self.mask] ==
                      another.comp_x[~another.mask]):
            return False
        if not np.all(self.comp_y[~self.mask] ==
                      another.comp_y[~another.mask]):
            return False
        try:
            self.unit_values == another.unit_values
        except unum.IncompatibleUnitsError:
            return False
        return True

    def __neg__(self):
        tmpvf = self.copy()
        tmpvf.comp_x = -tmpvf.comp_x
        tmpvf.comp_y = -tmpvf.comp_y
        return tmpvf

    def __add__(self, otherone):
        # if we add with a VectorField object
        if isinstance(otherone, VectorField):
            # test unities system
            self.unit_values + otherone.unit_values
            self.unit_x + otherone.unit_x
            self.unit_y + otherone.unit_y
            # identical shape and axis
            if super().__eq__(otherone):
                tmpvf = self.copy()
                fact = otherone.unit_values/self.unit_values
                tmpvf.comp_x += otherone.comp_x*fact.asNumber()
                tmpvf.comp_y += otherone.comp_y*fact.asNumber()
                tmpvf.mask = np.logical_or(self.mask, otherone.mask)
            # different shape, partially same axis
            else:
                # getting shared points
                new_ind_x = np.array([val in otherone.axis_x
                                      for val in self.axis_x])
                new_ind_y = np.array([val in otherone.axis_y
                                      for val in self.axis_y])
                new_ind_xo = np.array([val in self.axis_x
                                       for val in otherone.axis_x])
                new_ind_yo = np.array([val in self.axis_y
                                       for val in otherone.axis_y])
                if not np.any(new_ind_x) or not np.any(new_ind_y):
                    raise ValueError("Incompatible shapes")
                new_ind_Y, new_ind_X = np.meshgrid(new_ind_y, new_ind_x)
                new_ind_comp = np.logical_and(new_ind_X, new_ind_Y)
                new_ind_Yo, new_ind_Xo = np.meshgrid(new_ind_yo, new_ind_xo)
                new_ind_compo = np.logical_and(new_ind_Xo, new_ind_Yo)
                # getting new axis and values
                new_axis_x = self.axis_x[new_ind_x]
                new_axis_y = self.axis_y[new_ind_y]
                fact = otherone.unit_values/self.unit_values
                new_comp_x = (self.comp_x[new_ind_comp] +
                              otherone.comp_x[new_ind_compo] *
                              fact.asNumber())
                new_comp_y = (self.comp_y[new_ind_comp] +
                              otherone.comp_y[new_ind_compo] *
                              fact.asNumber())
                new_comp_x = new_comp_x.reshape((len(new_axis_x),
                                                 len(new_axis_y)))
                new_comp_y = new_comp_y.reshape((len(new_axis_x),
                                                 len(new_axis_y)))
                new_mask = np.logical_or(self.mask[new_ind_comp],
                                         otherone.mask[new_ind_compo])
                new_mask = new_mask.reshape((len(new_axis_x), len(new_axis_y)))
                # creating sf
                tmpvf = VectorField(new_axis_x, new_axis_y,
                                    new_comp_x, new_comp_y,
                                    mask=new_mask, unit_x=self.unit_x,
                                    unit_y=self.unit_y,
                                    unit_values=self.unit_values)
            return tmpvf
        # if we add with a unit object
        elif isinstance(otherone, unum.Unum):
            try:
                self.unit_values + otherone
            except unum.IncompatibleUnitsError as m:
                raise ValueError("Units don't match: {}".format(m))
            tmpsf = self.copy()
            tmpsf.comp_x += (otherone/self.unit_values).asNumber()
            tmpsf.comp_y += (otherone/self.unit_values).asNumber()
            return tmpsf
        else:
            try:
                tmpsf = self.copy()
                tmpsf.comp_x += otherone
                tmpsf.comp_y += otherone
                return tmpsf
            except TypeError:
                raise TypeError("You can only add a vectorfield "
                                "with others vectorfields or numbers,"
                                "not with {}".format(otherone))

    def __radd__(self, obj):
        return self.__add__(obj)

    def __sub__(self, obj):
        return self.__add__(-obj)

    def __rsub__(self, obj):
        return self.__neg__() + obj

    def __truediv__(self, obj):
        # units object
        if isinstance(obj, unum.Unum):
            tmpvf = self.copy()
            unit_values = tmpvf.unit_values / obj
            tmpvf.comp_x *= unit_values.asNumber()
            tmpvf.comp_y *= unit_values.asNumber()
            unit_values /= unit_values.asNumber()
            tmpvf.unit_values = unit_values
            return tmpvf
        # other vectorfield
        if isinstance(obj, VectorField):
            if np.any(self.axis_x != obj.axis_x)\
                    or np.any(self.axis_y != obj.axis_y)\
                    or self.unit_x != obj.unit_x\
                    or self.unit_y != obj.unit_y:
                raise ValueError("Fields are not consistent")
            tmpsf = self.copy()
            filt_nan = np.logical_and(obj.comp_x != 0, obj.comp_y != 0)
            comp_x = np.zeros(shape=self.comp_x.shape)
            comp_x[filt_nan] = self.comp_x[filt_nan]/obj.comp_x[filt_nan]
            comp_y = np.zeros(shape=self.comp_y.shape)
            comp_y[filt_nan] = self.comp_y[filt_nan]/obj.comp_y[filt_nan]
            mask = np.logical_or(self.mask, obj.mask)
            mask = np.logical_or(mask, np.logical_not(filt_nan))
            unit = self.unit_values / obj.unit_values
            tmpsf.comp_x = comp_x*unit.asNumber()
            tmpsf.comp_y = comp_y*unit.asNumber()
            tmpsf.mask = mask
            tmpsf.unit_values = unit/unit.asNumber()
            return tmpsf
        # array
        try:
            obj[0]
            obj = np.array(obj, subok=True)
            if not obj.shape == self.shape:
                raise ValueError()
            tmpsf = self.copy()
            mask = np.logical_or(self.mask, obj == 0)
            not_mask = np.logical_not(mask)
            comp_x = tmpsf.comp_x
            comp_y = tmpsf.comp_y
            comp_x[not_mask] /= obj[not_mask]
            comp_y[not_mask] /= obj[not_mask]
            tmpsf.comp_x = comp_x
            tmpsf.comp_y = comp_y
            tmpsf.mask = mask
            return tmpsf
        except TypeError:
            pass
        # number
        try:
            tmpsf = self.copy()
            tmpsf.comp_x /= obj
            tmpsf.comp_y /= obj
            return tmpsf
        except:
            pass
        # else...
        raise TypeError("Unsupported operation between {} and a "
                        "VectorField object".format(type(obj)))

    def __rtruediv__(self, obj):
        tmpsf = self.copy()
        # units object
        if isinstance(obj, unum.Unum):
            tmpsf.comp_x = obj.asNumber()/tmpsf.comp_x
            tmpsf.comp_y = obj.asNumber()/tmpsf.comp_y
            tmpsf.unit_values = obj/obj.asNumber()/tmpsf.unit_values
            return tmpsf
        try:
            obj[0]
            obj = np.array(obj, subok=True)
            if not obj.shape == self.shape:
                raise ValueError()
            mask = np.logical_or(self.mask, obj == 0)
            not_mask = np.logical_not(mask)
            comp_x = tmpsf.comp_x
            comp_y = tmpsf.comp_y
            comp_x[not_mask] = obj[not_mask] / tmpsf.comp_x[not_mask]
            comp_y[not_mask] = obj[not_mask] / tmpsf.comp_y[not_mask]
            tmpsf.comp_x = comp_x
            tmpsf.comp_y = comp_y
            tmpsf.mask = mask
            return tmpsf
        except TypeError:
            pass
        # number
        try:
            tmpsf.comp_x = obj/tmpsf.comp_x
            tmpsf.comp_y = obj/tmpsf.comp_y
            tmpsf.unit_values = 1/tmpsf.unit_values
            return tmpsf
        except:
            raise TypeError("Unsupported operation between {} and a "
                            "VectorField object".format(type(obj)))

    def __mul__(self, obj):
        # units
        if isinstance(obj, unum.Unum):
            tmpsf = self.copy()
            tmpsf.comp_x *= obj.asNumber()
            tmpsf.comp_y *= obj.asNumber()
            tmpsf.unit_values *= obj/obj.asNumber()
            tmpsf.mask = self.mask
            return tmpsf
        # sclarfield
        if isinstance(obj, VectorField):
            if np.any(self.axis_x != obj.axis_x)\
                    or np.any(self.axis_y != obj.axis_y)\
                    or self.unit_x != obj.unit_x\
                    or self.unit_y != obj.unit_y:
                raise ValueError("Fields are not consistent")
            tmpsf = self.copy()
            comp_x = self.comp_x * obj.comp_x
            comp_y = self.comp_y * obj.comp_y
            mask = np.logical_or(self.mask, obj.mask)
            unit = self.unit_values * obj.unit_values
            tmpsf.comp_x = comp_x*unit.asNumber()
            tmpsf.comp_y = comp_y*unit.asNumber()
            tmpsf.mask = mask
            tmpsf.unit_values = unit/unit.asNumber()
            return tmpsf
        # array
        try:
            obj[0]
            obj = np.array(obj, subok=True)
            if not obj.shape == self.shape:
                raise ValueError()
            tmpsf = self.copy()
            mask = self.mask
            not_mask = np.logical_not(mask)
            comp_x = tmpsf.comp_x
            comp_y = tmpsf.comp_y
            comp_x[not_mask] *= obj[not_mask]
            comp_y[not_mask] *= obj[not_mask]
            tmpsf.comp_x = comp_x
            tmpsf.comp_y = comp_y
            tmpsf.mask = mask
            return tmpsf
        except TypeError:
            pass
        # numbers
        try:
            tmpsf = self.copy()
            tmpsf.comp_x *= obj
            tmpsf.comp_y *= obj
            tmpsf.mask = self.mask
            return tmpsf
        except:
            raise TypeError("Unsupported operation between {} and a "
                            "VectorField object".format(type(obj)))
    __rmul__ = __mul__

    def __abs__(self):
        tmpsf = self.copy()
        tmpsf.comp_x = np.abs(tmpsf.comp_x)
        tmpsf.comp_y = np.abs(tmpsf.comp_y)
        return tmpsf

    def __pow__(self, number):
        tmpsf = self.copy()
        tmpsf.comp_x[np.logical_not(tmpsf.mask)] \
            = np.power(tmpsf.comp_x[np.logical_not(tmpsf.mask)], number)
        tmpsf.comp_y[np.logical_not(tmpsf.mask)] \
            = np.power(tmpsf.comp_y[np.logical_not(tmpsf.mask)], number)
        tmpsf.mask = self.mask
        tmpsf.unit_values = np.power(tmpsf.unit_values, number)
        return tmpsf

    def __iter__(self):
        for ij, xy in fld.Field.__iter__(self):
            i = ij[0]
            j = ij[1]
            if not self.mask[i, j]:
                yield ij, xy, self.comp_x[i, j], self.comp_y[i, j]

    def __repr__(self):
        return self.get_props()

    @property
    def comp_x(self):
        comp_x = self.__comp_x.copy()
        if self.__mask is not None:
            try:
                comp_x[self.mask] = np.nan
            except ValueError:
                comp_x[self.mask] = 0
        return comp_x

    @comp_x.setter
    def comp_x(self, new_comp_x):
        new_comp_x = np.asarray(new_comp_x)
        if self.shape[0] == new_comp_x.shape[0]\
           and self.shape[1] == new_comp_x.shape[1]:
            self.__comp_x = new_comp_x
        else:
            raise ValueError("'comp_x' should have the same shape as the "
                             "axis system: {}, not {}."
                             .format(self.shape, new_comp_x.shape))

    @comp_x.deleter
    def comp_x(self):
        raise Exception("Can't delete that")

    @property
    def comp_x_as_sf(self):
        tmp_vf = sf.ScalarField(self.axis_x, self.axis_y, self.comp_x,
                                mask=self.mask, unit_x=self.unit_x,
                                unit_y=self.unit_y,
                                unit_values=self.unit_values)
        return tmp_vf

    @property
    def comp_y(self):
        comp_y = self.__comp_y.copy()
        if self.__mask is not None:
            try:
                comp_y[self.mask] = np.nan
            except ValueError:
                comp_y[self.mask] = 0
        return comp_y

    @comp_y.setter
    def comp_y(self, new_comp_y):
        new_comp_y = np.asarray(new_comp_y)
        if self.shape[0] == new_comp_y.shape[0]\
           and self.shape[1] == new_comp_y.shape[1]:
            self.__comp_y = new_comp_y
        else:
            raise ValueError("'comp_y' should have the same shape as the "
                             "axis system: {}, not {}."
                             .format(self.shape, new_comp_y.shape))

    @comp_y.deleter
    def comp_y(self):
        raise Exception("Can't delete that")

    @property
    def comp_y_as_sf(self):
        tmp_vf = sf.ScalarField(self.axis_x, self.axis_y, self.comp_y,
                                mask=self.mask, unit_x=self.unit_x,
                                unit_y=self.unit_y,
                                unit_values=self.unit_values)
        return tmp_vf

    @property
    def dtype(self):
        return (self.comp_x.dtype, self.comp_y.dtype)

    @property
    def mask(self):
        return np.logical_or(self.__mask,
                             np.logical_or(
                                 np.isnan(self.__comp_x),
                                 np.isnan(self.__comp_y)))

    @mask.setter
    def mask(self, new_mask):
        # check 'new_mask' coherence
        if isinstance(new_mask, (bool, np.bool)):
            fill_value = new_mask
            new_mask = np.empty(self.shape, dtype=bool)
            new_mask.fill(fill_value)
        else:
            try:
                new_mask = np.asarray(new_mask, dtype=bool)
            except TypeError:
                raise TypeError("'mask' should be a boolean or an array"
                                "of booleans, but is currently {}"
                                .format(new_mask))
        if self.shape[0] != new_mask.shape[0]\
           or self.shape[1] != new_mask.shape[1]:
            raise ValueError("'mask' should be of the same size as the "
                             "axis system: {}, but is currently: {}"
                             .format(self.shape, new_mask.shape))
        self.__mask = new_mask

    @mask.deleter
    def mask(self):
        raise Exception("Can't delete that")

    @property
    def mask_as_sf(self):
        tmp_vf = sf.ScalarField(self.axis_x, self.axis_y, self.mask,
                                mask=None, unit_x=self.unit_x,
                                unit_y=self.unit_y,
                                unit_values='')
        return tmp_vf

    @property
    def unit_values(self):
        return self.__unit_values

    @unit_values.setter
    def unit_values(self, new_unit_values):
        if isinstance(new_unit_values, unum.Unum):
            if np.isclose(new_unit_values.asNumber(), 1):
                self.__unit_values = new_unit_values
            else:
                raise ValueError('New values unit is not 1')
        else:
            try:
                self.__unit_values = make_unit(new_unit_values)
            except TypeError:
                raise TypeError('Unrecognized unity representation.')

    @unit_values.deleter
    def unit_values(self):
        raise Exception("Can't delete that.")

    @property
    def min(self):
        return np.min(self.magnitude[~self.mask])

    @property
    def max(self):
        return np.max(self.magnitude[~self.mask])

    @property
    def mean(self):
        return np.mean(self.magnitude[~self.mask])

    @property
    def std(self):
        return np.std(self.magnitude[~self.mask])

    @property
    def magnitude(self):
        """
        Return a scalar field with the velocity field magnitude.
        """
        comp_x, comp_y = self.comp_x, self.comp_y
        mask = self.mask
        values = (comp_x**2 + comp_y**2)**(.5)
        values[mask] = np.NaN
        return values

    @property
    def magnitude_as_sf(self):
        """
        Return a scalarfield with the velocity field magnitude.
        """
        tmp_vf = sf.ScalarField(self.axis_x, self.axis_y, self.magnitude,
                                mask=self.mask, unit_x=self.unit_x,
                                unit_y=self.unit_y,
                                unit_values=self.unit_values)
        return tmp_vf

    @property
    def theta(self):
        """
        Return a scalar field with the vector angle (in reference of the unit_y
        vector [1, 0]).

        Parameters
        ----------
        low_velocity_filter : number
            If not zero, points where V < Vmax*low_velocity_filter are masked.

        Returns
        -------
        theta_sf : sf.ScalarField object
            Contening theta field.
        """
        # get data
        comp_x, comp_y = self.comp_x, self.comp_y
        not_mask = np.logical_not(self.mask)
        theta = np.zeros(self.shape)
        # getting angle
        norm = self.magnitude
        not_mask = np.logical_and(not_mask, norm != 0)
        theta[not_mask] = comp_x[not_mask]/norm[not_mask]
        theta[not_mask] = np.arccos(theta[not_mask])
        tmp_comp_y = comp_y.copy()
        tmp_comp_y[~not_mask] = 0
        sup_not_mask = tmp_comp_y < 0
        theta[sup_not_mask] = 2*np.pi - theta[sup_not_mask]
        return theta

    @property
    def theta_as_sf(self):
        """
        Return a scalarfield with the velocity field angles.
        """
        tmp_vf = sf.ScalarField(self.axis_x, self.axis_y, self.theta,
                                mask=False, unit_x=self.unit_x,
                                unit_y=self.unit_y,
                                unit_values=self.unit_values)
        return tmp_vf

    def get_props(self):
        """
        Print the VectorField main properties
        """
        text = "Shape: {}\n".format(self.shape)
        unit_x = self.unit_x.strUnit()
        text += "Axe x: [{}..{}]{}\n".format(self.axis_x[0], self.axis_x[-1],
                                             unit_x)
        unit_y = self.unit_y.strUnit()
        text += "Axe y: [{}..{}]{}\n".format(self.axis_y[0], self.axis_y[-1],
                                             unit_y)
        unit_values = self.unit_values.strUnit()
        xmin = np.min(self.comp_x[~self.mask])
        xmax = np.max(self.comp_x[~self.mask])
        ymin = np.min(self.comp_y[~self.mask])
        ymax = np.max(self.comp_y[~self.mask])
        text += "Comp x: [{}..{}]{}\n".format(xmin, xmax, unit_values)
        text += "Comp y: [{}..{}]{}\n".format(ymin, ymax, unit_values)
        nmb_mask = np.sum(self.mask)
        nmb_tot = self.shape[0]*self.shape[1]
        text += "Masked values: {}/{}".format(nmb_mask, nmb_tot)
        return text

    def get_value(self, x, y, ind=False, unit=False):
        """
        Return the vector field components on the point (x, y).

        Parameters
        ----------
        x, y: numbers
            Positions where to get the components.
        ind: boolean
            If 'True', x and y are indices, else (default),
            x and y are in axis units
        unit: boolean
            If 'True', also return the components unities.
            (default to False)
        """
        return np.array([self.comp_x_as_sf.get_value(x, y,
                                                     ind=ind, unit=unit),
                         self.comp_y_as_sf.get_value(x, y,
                                                     ind=ind, unit=unit)])

    def get_profile(self, x=None, y=None, component='x', ind=False,
                    interp='linear'):
        """
        Return a profile of the vector field, at the given position.
        If position is an interval, the fonction return an average profile
        in this interval.

        Parameters
        ----------
        x, y: numbers or 2x1 array of numbers
            Position of the wanted profile.
        component: string in ['x', 'y']
            Component to get a profile from.
            (default to 'x')
        ind : boolean
            If 'True', position has to be given in indices
            If 'False' (default), position has to be given in axis unit.
        interp : string in ['nearest', 'linear']
            if 'nearest', get the profile at the nearest position on the grid,
            if 'linear', use linear interpolation to get the profile at the
            exact position

        Returns
        -------
        profile : prof.Profile object
            Wanted profile
        """
        if component == 'x':
            return self.comp_x_as_sf.get_profile(x, y, ind, interp=interp)
        elif component == 'y':
            return self.comp_y_as_sf.get_profile(x, y, ind, interp=interp)
        else:
            raise TypeError("'component' must be 'x' or 'y'")

    def get_histograms(self, bins=200, cum=False,
                       normalized=False):
        """
        Return a vectorfield component histograms.

        Parameters
        ==========
        cum: boolean
            If True, get a cumulative histogram.
        normalized: boolean
            If True, normalize the histogram.
        bins: integer
            Number of bins (default to 200).

        Returns
        =======
        histx, histy: Profile objects.
            Histograms.
        """
        histx = self.comp_x_as_sf.get_histogram(bins=bins, cum=cum,
                                                normalized=normalized)
        histy = self.comp_y_as_sf.get_histogram(bins=bins, cum=cum,
                                                normalized=normalized)
        return histx, histy

    def get_interpolators(self, interp="linear"):
        """
        Return the field interpolators.

        Parameters
        ----------
        kind : {‘linear’, ‘cubic’, ‘quintic’}, optional
            The kind of spline interpolation to use. Default is ‘linear’.

        Returns
        -------
        interpx, interpy
            Interpolators for each components

        Example
        -------
        >>> interp = SF.get_interpolator(interp='linear')
        >>> print(interp(4, 5.3))
        ... [34.3]
        >>> print(interp([3, 4, 5], 5.3))
        ... [23, 34.3, 54]
        """
        interpx = self.comp_x_as_sf.get_interpolator(interp=interp)
        interpy = self.comp_y_as_sf.get_interpolator(interp=interp)
        return interpx, interpy

    def copy(self):
        """
        Return a copy of the vectorfield.
        """
        return copy.deepcopy(self)

    def get_norm(self, norm=2):
        """
        Return the field norm.

        Parameters
        ----------
        norm: positive integer
            Norm order.

        Returns
        -------
        norm : number
            Norm.
        """
        values = np.concatenate((self.comp_x[~self.mask],
                                 self.comp_y[~self.mask]))
        res = (np.sum(np.abs(values)**norm))**(1./norm)
        return res

    def scale(self, scalex=None, scaley=None, scalev=None, inplace=False):
        """
        Scale the VectorField.

        Parameters
        ----------
        scalex, scaley, scalev : numbers or Unum objects
            Scale for the axis and the values.
        inplace : boolean
            .
        """
        if inplace:
            tmp_f = self
        else:
            tmp_f = self.copy()
        # xy
        revx, revy = fld.Field.scale(tmp_f, scalex=scalex, scaley=scaley,
                                     inplace=True, output_reverse=True)
        # v
        if scalev is None:
            pass
        elif isinstance(scalev, unum.Unum):
            new_unit = tmp_f.unit_values*scalev
            fact = new_unit.asNumber()
            new_unit /= fact
            tmp_f.unit_values = new_unit
            tmp_f.comp_x *= fact
            tmp_f.comp_y *= fact
        else:
            tmp_f.comp_x *= scalev
            tmp_f.comp_y *= scalev
        if revx and revy:
            tmp_f.comp_x = -tmp_f.comp_x[::-1, ::-1]
            tmp_f.comp_y = -tmp_f.comp_y[::-1, ::-1]
        elif revx:
            tmp_f.comp_x = -tmp_f.comp_x[::-1, :]
            tmp_f.comp_y = tmp_f.comp_y[::-1, :]
        elif revy:
            tmp_f.comp_x = tmp_f.comp_x[:, ::-1]
            tmp_f.comp_y = -tmp_f.comp_y[:, ::-1]
        # returning
        if not inplace:
            return tmp_f

    def rotate(self, angle, inplace=False):
        """
        Rotate the vector field.

        Parameters
        ----------
        angle : integer
            Angle in degrees (signe in trigonometric convention).
            In order to preserve the orthogonal grid, only multiples of
            90° are accepted.
        inplace : boolean, optional
            If True, the field is rotated in place, else (default),
            a rotated copy is returned.

        Returns
        -------
        rotated_field : Field object
            Rotated field.
        """
        # get data
        if inplace:
            tmp_field = self
        else:
            tmp_field = self.copy()
        # normalize angle
        angle = angle % 360
        # rotate the parent
        fld.Field.rotate(tmp_field, angle, inplace=True)
        # rotate
        nmb_rot90 = int(angle/90)
        mask = tmp_field.mask
        comp_x = tmp_field.comp_x
        comp_y = tmp_field.comp_y
        tmp_field.__comp_x = np.rot90(comp_x, nmb_rot90)
        tmp_field.__comp_y = np.rot90(comp_y, nmb_rot90)
        tmp_field.__mask = np.rot90(mask, nmb_rot90)
        # returning
        if not inplace:
            return tmp_field

    def change_unit(self, axis, new_unit):
        """
        Change the unit of an axis.

        Parameters
        ----------
        axis : string
            'y' for changing the profile y axis unit
            'x' for changing the profile x axis unit
            'values' or changing values unit
        new_unit : Unum.unit object or string
            The new unit.
        """
        if not isinstance(new_unit, unum.Unum):
            new_unit = make_unit(new_unit)
        if axis not in ['x', 'y', 'values']:
            raise TypeError("'axis' should be 'x', 'y', or 'values'")
        if axis == 'x':
            fld.Field.change_unit(self, axis, new_unit)
        elif axis == 'y':
            fld.Field.change_unit(self, axis, new_unit)
        elif axis == 'values':
            old_unit = self.unit_values
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.comp_x *= fact
            self.comp_y *= fact
            self.unit_values = new_unit/fact
        else:
            raise ValueError()

    def crop(self, intervx=None, intervy=None, ind=False,
             inplace=False):
        """
        Crop the vector field.

        Parameters
        ----------
        intervx : array, optional
            Wanted interval along x.
        intervy : array, optional
            Wanted interval along y.
        ind : boolean, optional
            If 'True', intervals are understood as indices along axis.
            If 'False' (default), intervals are understood in axis units.
        inplace : boolean, optional
            If 'True', the field is croped in place,
            else (default), a copy is returned.
        """
        if inplace:
            comp_x = self.comp_x
            comp_y = self.comp_y
            mask = self.mask
            indmin_x, indmax_x, indmin_y, indmax_y = \
                fld.Field.crop(self, intervx, intervy, full_output=True,
                               ind=ind, inplace=True)
            self.__comp_x = comp_x[indmin_x:indmax_x + 1,
                                   indmin_y:indmax_y + 1]
            self.__comp_y = comp_y[indmin_x:indmax_x + 1,
                                   indmin_y:indmax_y + 1]
            self.__mask = mask[indmin_x:indmax_x + 1,
                               indmin_y:indmax_y + 1]
        else:
            indmin_x, indmax_x, indmin_y, indmax_y, cropfield = \
                fld.Field.crop(self, intervx=intervx, intervy=intervy,
                               full_output=True, ind=ind, inplace=False)
            cropfield.__comp_x = self.comp_x[indmin_x:indmax_x + 1,
                                             indmin_y:indmax_y + 1]
            cropfield.__comp_y = self.comp_y[indmin_x:indmax_x + 1,
                                             indmin_y:indmax_y + 1]
            cropfield.__mask = self.mask[indmin_x:indmax_x + 1,
                                         indmin_y:indmax_y + 1]
            return cropfield

    def extend(self, nmb_left=0, nmb_right=0, nmb_up=0, nmb_down=0, value=None,
               inplace=False, ind=True):
        """
        Add columns and/or lines of masked values to the field.

        Parameters
        ----------
        nmb_left, nmb_right, nmb_up, nmb_down : integers
            Number of lines/columns to add in each direction.
        inplace : bool
            If 'True', extend the field in place,
            else (default), return an extended copy of the field.

        Returns
        -------
        Extended_field : Field object
            Extended field.

        Note
        ----
        If the axis values are not equally spaced, a linear extrapolation
        is used to obtain the new axis values.

        """
        if inplace:
            tmp_vf = self
        else:
            tmp_vf = self.copy()
        if not ind:
            dx = self.dx
            dy = self.dy
            nmb_left = np.ceil(nmb_left/dx)
            nmb_right = np.ceil(nmb_right/dx)
            nmb_up = np.ceil(nmb_up/dy)
            nmb_down = np.ceil(nmb_down/dy)
            ind = True
        # check params
        if not (isinstance(nmb_left, int) or nmb_left % 1 == 0):
            raise TypeError()
        if not (isinstance(nmb_right, int) or nmb_right % 1 == 0):
            raise TypeError()
        if not (isinstance(nmb_up, int) or nmb_up % 1 == 0):
            raise TypeError()
        if not (isinstance(nmb_down, int) or nmb_down % 1 == 0):
            raise TypeError()
        nmb_left = int(nmb_left)
        nmb_right = int(nmb_right)
        nmb_up = int(nmb_up)
        nmb_down = int(nmb_down)
        if np.any(np.array([nmb_left, nmb_right, nmb_up, nmb_down]) < 0):
            raise ValueError()
        # used herited method to extend the field
        fld.Field.extend(tmp_vf, nmb_left=nmb_left,
                         nmb_right=nmb_right, nmb_up=nmb_up,
                         nmb_down=nmb_down, inplace=True)
        new_shape = tmp_vf.shape
        # extend the value ans mask
        if value is None:
            new_comp_x = np.zeros(new_shape, dtype=float)
            new_comp_y = np.zeros(new_shape, dtype=float)
            new_mask = np.ones(new_shape, dtype=bool)
        else:
            new_comp_x = np.ones(new_shape, dtype=float)*value
            new_comp_y = np.ones(new_shape, dtype=float)*value
            new_mask = np.zeros(new_shape, dtype=bool)
        if nmb_right == 0:
            slice_x = slice(nmb_left, new_comp_x.shape[0] + 2)
        else:
            slice_x = slice(nmb_left, -nmb_right)
        if nmb_up == 0:
            slice_y = slice(nmb_down, new_comp_x.shape[1] + 2)
        else:
            slice_y = slice(nmb_down, -nmb_up)
        new_comp_x[slice_x, slice_y] = self.comp_x
        new_comp_y[slice_x, slice_y] = self.comp_y
        new_mask[slice_x, slice_y] = self.mask
        tmp_vf.__comp_x = new_comp_x
        tmp_vf.__comp_y = new_comp_y
        tmp_vf.__mask = new_mask
        # return
        if not inplace:
            return tmp_vf

    def crop_masked_border(self, hard=False, inplace=False):
        """
        Crop the masked border of the field in place or not.

        Parameters
        ----------
        hard : boolean, optional
            If 'True', partially masked border are croped as well.
        """
        #
        if inplace:
            tmp_vf = self
        else:
            tmp_vf = self.copy()
        # checking masked values presence
        mask = tmp_vf.mask
        if not np.any(mask):
            return None
        # hard cropping
        if hard:
            # remove trivial borders
            tmp_vf.crop_masked_border(hard=False, inplace=True)
            # until there is no more masked values
            while np.any(tmp_vf.mask):
                # getting number of masked value on each border
                bd1 = np.sum(tmp_vf.mask[0, :])
                bd2 = np.sum(tmp_vf.mask[-1, :])
                bd3 = np.sum(tmp_vf.mask[:, 0])
                bd4 = np.sum(tmp_vf.mask[:, -1])
                # getting more masked border
                more_masked = np.argmax([bd1, bd2, bd3, bd4])
                # deleting more masked border
                if more_masked == 0:
                    len_x = len(tmp_vf.axis_x)
                    tmp_vf.crop(intervx=[1, len_x], ind=True, inplace=True)
                elif more_masked == 1:
                    len_x = len(tmp_vf.axis_x)
                    tmp_vf.crop(intervx=[0, len_x - 2], ind=True,
                                inplace=True)
                elif more_masked == 2:
                    len_y = len(tmp_vf.axis_y)
                    tmp_vf.crop(intervy=[1, len_y], ind=True,
                                inplace=True)
                elif more_masked == 3:
                    len_y = len(tmp_vf.axis_y)
                    tmp_vf.crop(intervy=[0, len_y - 2], ind=True,
                                inplace=True)
        # soft cropping
        else:
            axis_x_m = np.logical_not(np.all(mask, axis=1))
            axis_y_m = np.logical_not(np.all(mask, axis=0))
            axis_x_min = np.where(axis_x_m)[0][0]
            axis_x_max = np.where(axis_x_m)[0][-1]
            axis_y_min = np.where(axis_y_m)[0][0]
            axis_y_max = np.where(axis_y_m)[0][-1]
            tmp_vf.crop([axis_x_min, axis_x_max],
                        [axis_y_min, axis_y_max],
                        ind=True, inplace=True)
        # returning
        if not inplace:
            return tmp_vf

    def fill(self, kind='linear', value=0., inplace=False, reduce_tri=True,
             crop=False):
        """
        Fill the masked parts of the vector field.

        Parameters
        ----------
        kind : string, optional
            Type of algorithm used to fill.
            'value' : fill with the given value
            'nearest' : fill with the nearest value
            'linear' (default): fill using linear interpolation
            (Delaunay triangulation)
            'cubic' : fill using cubic interpolation (Delaunay triangulation)
        value : number
            Value used to fill (for kind='value').
        inplace : boolean, optional
            If 'True', fill the vector field in place.
            If 'False' (default), return a filled version of the field.
        reduce_tri : boolean, optional
            If 'True', treatment is used to reduce the triangulation effort
            (faster when a lot of masked values)
            If 'False', no treatment
            (faster when few masked values)
        crop : boolean, optional
            If 'True', SF borders are croped before filling.
        """
        if inplace:
            tmp_f = self
        else:
            tmp_f = self.copy()
        #
        vx = self.comp_x_as_sf
        vy = self.comp_y_as_sf
        vx.fill(kind=kind, value=value, inplace=True, reduce_tri=reduce_tri,
                crop=crop)
        vy.fill(kind=kind, value=value, inplace=True, reduce_tri=reduce_tri,
                crop=crop)
        tmp_f.comp_x = vx.values
        tmp_f.comp_y = vy.values
        tmp_f.mask = False
        #
        return tmp_f

    def smooth(self, tos='uniform', size=None, inplace=False, **kw):
        """
        Smooth the vectorfield.

        Warning : fill up the field (should be used carefully with masked field
        borders)

        Parameters
        ----------
        tos : string, optional
            Type of smoothing, can be 'uniform' (default) or 'gaussian'
            (See ndimage module documentation for more details)
        size : number, optional
            Size of the smoothing (is radius for 'uniform' and
            sigma for 'gaussian') in indice number.
            Default is 3 for 'uniform' and 1 for 'gaussian'.
        inplace : boolean, optional
            If True, Field is smoothed in place,
            else, the smoothed field is returned.
        kw : dic
            Additional parameters for ndimage methods
            (See ndimage documentation)
        """
        if inplace:
            tmp_f = self
        else:
            tmp_f = self.copy()
        #
        vx = self.comp_x_as_sf
        vy = self.comp_y_as_sf
        vx.smooth(tos=tos, size=size, inplace=True, **kw)
        vy.smooth(tos=tos, size=size, inplace=True, **kw)
        tmp_f.comp_x = vx.values
        tmp_f.comp_y = vy.values
        tmp_f.mask = False
        # return
        return tmp_f

    def make_evenly_spaced(self, interp="linear", res=1, inplace=False):
        """
        Use interpolation to make the field evenly spaced.

        Parameters
        ----------
        interp : {‘linear’, ‘cubic’, ‘quintic’}, optional
            The kind of spline interpolation to use. Default is ‘linear’.
        res : number
            Resolution of the resulting field.
            A value of 1 meaning a spatial resolution equal to the smallest
            space along the two axis for the initial field.
            A value of 2 means half this resolution.
        inplace: boolean
            If True, modify the vector field in place, else, return a
            modified version of it.
        """
        if inplace:
            tmp_f = self
        else:
            tmp_f = self.copy()
        # get data
        vx = self.comp_x_as_sf
        vy = self.comp_y_as_sf
        vx.make_evenly_spaced(interp=interp, res=res, inplace=True)
        vy.make_evenly_spaced(interp=interp, res=res, inplace=True)
        tmp_f.__init__(vx.axis_x, vx.axis_y,
                       vx.values, vy.values,
                       mask=np.logical_or(vx.mask, vy.mask),
                       unit_x=vx.unit_x, unit_y=vx.unit_y,
                       unit_values=vx.unit_values)
        # return
        return tmp_f

    def reduce_resolution(self, fact, inplace=False):
        """
        Reduce the spatial resolution of the vector field by a factor 'fact'.

        Parameters
        ----------
        fact : int
            Reducing factor.
        inplace : boolean, optional
            .
        """
        if inplace:
            tmp_vf = self
        else:
            tmp_vf = self.copy()
        #
        fact = int(fact)
        if fact < 1:
            raise ValueError()
        if fact == 1:
            if inplace:
                pass
            else:
                return tmp_vf
        if fact % 2 == 0:
            pair = True
        else:
            pair = False
        # get new axis
        axis_x = tmp_vf.axis_x
        axis_y = tmp_vf.axis_y
        if pair:
            new_axis_x = (axis_x[np.arange(fact/2 - 1, len(axis_x) - fact/2,
                                           fact, dtype=int)] +
                          axis_x[np.arange(fact/2, len(axis_x) - fact/2 + 1,
                                           fact, dtype=int)])/2.
            new_axis_y = (axis_y[np.arange(fact/2 - 1, len(axis_y) - fact/2,
                                           fact, dtype=int)] +
                          axis_y[np.arange(fact/2, len(axis_y) - fact/2 + 1,
                                           fact, dtype=int)])/2.
        else:
            new_axis_x = axis_x[np.arange((fact - 1)/2,
                                          len(axis_x) - (fact - 1)/2,
                                          fact, dtype=int)]
            new_axis_y = axis_y[np.arange((fact - 1)/2,
                                          len(axis_y) - (fact - 1)/2,
                                          fact, dtype=int)]
        # get new values
        comp_x = tmp_vf.comp_x
        comp_y = tmp_vf.comp_y
        mask = tmp_vf.mask
        if pair:
            inds_x = np.arange(fact/2, len(axis_x) - fact/2 + 1,
                               fact, dtype=int)
            inds_y = np.arange(fact/2, len(axis_y) - fact/2 + 1,
                               fact, dtype=int)
            new_comp_x = np.zeros((len(inds_x), len(inds_y)))
            new_comp_y = np.zeros((len(inds_x), len(inds_y)))
            new_mask = np.zeros((len(inds_x), len(inds_y)))
            for i in np.arange(len(inds_x)):
                intervx = slice(inds_x[i] - int(fact/2),
                                inds_x[i] + int(fact/2))
                for j in np.arange(len(inds_y)):
                    intervy = slice(inds_y[j] - int(fact/2),
                                    inds_y[j] + int(fact/2))
                    if np.all(mask[intervx, intervy]):
                        new_mask[i, j] = True
                        new_comp_x[i, j] = 0.
                        new_comp_y[i, j] = 0.
                    else:
                        new_comp_x[i, j] = np.mean(comp_x[intervx, intervy]
                                                   [~mask[intervx, intervy]])
                        new_comp_y[i, j] = np.mean(comp_y[intervx, intervy]
                                                   [~mask[intervx, intervy]])

        else:
            inds_x = np.arange((fact - 1)/2, len(axis_x) - (fact - 1)/2, fact)
            inds_y = np.arange((fact - 1)/2, len(axis_y) - (fact - 1)/2, fact)
            new_comp_x = np.zeros((len(inds_x), len(inds_y)))
            new_comp_y = np.zeros((len(inds_x), len(inds_y)))
            new_mask = np.zeros((len(inds_x), len(inds_y)))
            for i in np.arange(len(inds_x)):
                intervx = slice(int(inds_x[i] - (fact - 1)/2),
                                int(inds_x[i] + (fact - 1)/2 + 1))
                for j in np.arange(len(inds_y)):
                    intervy = slice(int(inds_y[j] - (fact - 1)/2),
                                    int(inds_y[j] + (fact - 1)/2 + 1))
                    print(intervx, intervy)
                    if np.all(mask[intervx, intervy]):
                        new_mask[i, j] = True
                        new_comp_x[i, j] = 0.
                        new_comp_y[i, j] = 0.
                    else:
                        new_comp_x[i, j] = np.mean(comp_x[intervx, intervy]
                                                   [~mask[intervx, intervy]])
                        new_comp_y[i, j] = np.mean(comp_y[intervx, intervy]
                                                   [~mask[intervx, intervy]])
        # returning
        tmp_vf.__init__(new_axis_x, new_axis_y,
                        new_comp_x, new_comp_y,
                        mask=new_mask,
                        unit_x=tmp_vf.unit_x,
                        unit_y=tmp_vf.unit_y,
                        unit_values=tmp_vf.unit_values)
        if not inplace:
            return tmp_vf

    def display(self, component=None, kind=None,
                annotate=True, **plotargs):
        """
        Display the vector field.

        Parameters
        ----------
        component : string, optional
            Component to display, can be 'V', 'x', 'y', 'magnitude' or 'mask'
        kind : string, optinnal
            can be 'quiver', 'stream', 'imshow', 'contour', 'contourf'
        annotate: boolean
            If True (default) add label and legedn to the graph.
        **plotargs : dict
            Arguments passed to the plotting function.

        Returns
        -------
        fig : figure reference
            Reference to the displayed figure.
        """
        # default
        if component is None:
            if kind in ['quiver', 'stream', None]:
                component = 'V'
            else:
                component = 'magnitude'
        if kind is None:
            if component in ['V']:
                kind = 'quiver'
            else:
                kind = 'imshow'
        # check
        if component not in ['V', 'x', 'y', 'magnitude', 'mask']:
            raise ValueError("'component' must be 'x', 'y', 'magnitude', "
                             "'V' or 'mask', not {}".format(component))
        if kind not in ['quiver', 'stream', 'imshow', 'contour', 'contourf']:
            raise ValueError("'kind' must be 'quiver', 'stream', "
                             "'imshow', 'contour', or 'contourf',"
                             " not {}".format(kind))
        if component == 'V':
            if kind in ['imshow', 'contour', 'contourf']:
                raise ValueError('Can plot a {} with component {}'
                                 .format(kind, component))
        else:
            if kind in ['quiver', 'stream']:
                raise ValueError('Can plot a {} with component {}'
                                 .format(kind, component))
        # use scalarfield to display
        if component == "x":
            return self.comp_x_as_sf.display(component='values',
                                             kind=kind,
                                             annotate=annotate,
                                             **plotargs)
        if component == "y":
            return self.comp_y_as_sf.display(component='values',
                                             kind=kind,
                                             annotate=annotate,
                                             **plotargs)
        if component == "magnitude":
            return self.magnitude_as_sf.display(component='values',
                                                kind=kind,
                                                annotate=annotate,
                                                **plotargs)
        if component == "mask":
            return self.comp_x_as_sf.display(component='mask',
                                             kind=kind,
                                             annotate=annotate,
                                             **plotargs)
        # getting datas
        axis_x, axis_y = self.axis_x, self.axis_y
        unit_x, unit_y = self.unit_x, self.unit_y
        X, Y = np.meshgrid(self.axis_y, self.axis_x)
        # quiver
        if kind == 'quiver':
            if 'color' in list(plotargs.keys()):
                C = plotargs.pop('color')
                if 'c' in plotargs.keys():
                    plotargs.pop('c')
            else:
                C = self.magnitude.transpose()
                if 'c' in plotargs.keys():
                    plotargs.pop('c')
            plot = plt.quiver(axis_x, axis_y,
                              self.comp_x.transpose(),
                              self.comp_y.transpose(),
                              C, **plotargs)
        if kind == 'stream':
            # set adptative linewidth
            if 'lw' in list(plotargs.keys()):
                tmp_lw = plotargs.pop('lw')
            elif 'linewidth' in list(plotargs.keys()):
                tmp_lw = plotargs.pop('linewidth')
            else:
                tmp_lw = 1
            if np.array(tmp_lw).shape != ():
                pass
            else:
                magn = self.magnitude
                magn[np.isnan(magn)] = 0
                tmp_lw *= 0.1 + 0.9*magn/np.max(magn)
                tmp_lw = tmp_lw.transpose()
            # set color
            if 'color' in list(plotargs.keys()):
                color = plotargs.pop('color')
                if 'c' in plotargs.keys():
                    plotargs.pop('c')
            else:
                color = self.magnitude.transpose()
                if 'c' in plotargs.keys():
                    plotargs.pop('c')
            # plot
            Vx = self.comp_x.transpose()
            Vy = self.comp_y.transpose()
            print(axis_x)
            print(axis_y)
            print(Vx)
            print(Vy)
            print(color)
            plot = plt.streamplot(axis_x, axis_y, Vx, Vy, color=color,
                                  linewidth=tmp_lw, **plotargs)
        # Annotate quiver/streams
        if annotate:
            # labels
            if unit_x.strUnit() == "[]":
                plt.xlabel("x")
            else:
                plt.xlabel("x " + unit_x.strUnit())
            if unit_y.strUnit() == "[]":
                plt.ylabel("y")
            else:
                plt.ylabel("y " + unit_y.strUnit())
            # title
            plt.title("")
            # colorbar
            try:
                if kind == 'stream':
                    cb = plt.colorbar(plot.lines)
                else:
                    cb = plt.colorbar(plot)
                if self.unit_values.strUnit() == "[]":
                    cb.set_label("Magnitude")
                else:
                    cb.set_label(self.unit_values.strUnit())
            except TypeError:
                pass
            # search for limits in case of masked field
            if component != 'mask':
                mask = self.mask
                for i in np.arange(len(self.axis_x)):
                    if not np.all(mask[i, :]):
                        break
                xmin = self.axis_x[i]
                for i in np.arange(len(self.axis_x) - 1, -1, -1):
                    if not np.all(mask[i, :]):
                        break
                xmax = self.axis_x[i]
                for i in np.arange(len(self.axis_y)):
                    if not np.all(mask[:, i]):
                        break
                ymin = self.axis_y[i]
                for i in np.arange(len(self.axis_y) - 1, -1, -1):
                    if not np.all(mask[:, i]):
                        break
                ymax = self.axis_y[i]
                plt.xlim([xmin, xmax])
                plt.ylim([ymin, ymax])
        else:
            raise ValueError("'component' must be 'x', 'y', 'magnitude', "
                             "'V' or 'mask'")
        return plt.gca()
