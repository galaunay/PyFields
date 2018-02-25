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
import warnings

import scipy.ndimage.measurements as msr
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spinterp
import unum
from scipy import ndimage

from . import field as fld
from . import profile as prf
from ..utils import make_unit


class ScalarField(fld.Field):
    """
    Class representing a scalar field (2D field, with one component on each
    point).
    """

    def __init__(self, axis_x, axis_y, values, mask=None,
                 unit_x="", unit_y="", unit_values=""):
        """
        Object representing a scalar field (2D field, with one component on each
        point).

        Parameters
        ----------
        axis_x : nx1 array
            x axis values.
        axis_y : mx1 array
            y axis values.
        values : nxm array or masked array
            Values of the field at the axis points.
        unit_x, unit_y : String or Unum object, optional
            x and y axis units.
        unit_values : String or Unum object, optional
            Field values unit.
        """
        # build axis system
        fld.Field.__init__(self, axis_x=axis_x, axis_y=axis_y,
                           unit_x=unit_x, unit_y=unit_y)
        self.__mask = None
        self.__values = None
        # store values properties
        self.values = np.array(values)
        if self.values.shape[0] != len(self.axis_x) or \
           self.values.shape[1] != len(self.axis_y):
            raise ValueError('Incoherent shapes, axis sizes are {}, {},'
                             'but values size is {}'
                             .format(len(self.axis_x), len(self.axis_y),
                                     self.values.shape))
        self.unit_values = unit_values
        # store mask (if necessary)
        if mask is None:
            nans = np.isnan(values)
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
            mask = np.logical_or(mask, np.isnan(values))
        self.mask = mask

    @property
    def values(self):
        values = self.__values.copy()
        if self.__mask is not None:
            try:
                values[self.mask] = np.nan
            except ValueError:
                values[self.mask] = 0
        return values

    @values.setter
    def values(self, new_values):
        new_values = np.asarray(new_values)
        if self.shape[0] == new_values.shape[0]\
           and self.shape[1] == new_values.shape[1]:
            self.__values = new_values
        else:
            raise ValueError("'values' should have the same shape as the "
                             "axis system: {}, not {}."
                             .format(self.shape, new_values.shape))

    @values.deleter
    def values(self):
        raise Exception("Can't delete that")

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def mask(self):
        return np.logical_or(self.__mask,
                             np.isnan(self.__values))

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
        tmp_sf = ScalarField(self.axis_x, self.axis_y, self.mask,
                             mask=None, unit_x=self.unit_x,
                             unit_y=self.unit_y,
                             unit_values='')
        return tmp_sf

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

    def __eq__(self, another):
        if not isinstance(another, ScalarField):
            return False
        if not super().__eq__(another):
            return False
        if not np.all(self.mask == another.mask):
            return False
        if not np.all(self.values[~self.mask] ==
                      another.values[~another.mask]):
            return False
        try:
            self.unit_values == another.unit_values
        except unum.IncompatibleUnitsError:
            return False
        return True

    def __neg__(self):
        tmpsf = self.copy()
        tmpsf.values = -tmpsf.values
        return tmpsf

    def __add__(self, otherone):
        # if we add with a ScalarField object
        if isinstance(otherone, ScalarField):
            # test unities system
            self.unit_values + otherone.unit_values
            self.unit_x + otherone.unit_x
            self.unit_y + otherone.unit_y
            # identical shape and axis
            if super().__eq__(otherone):
                tmpsf = self.copy()
                fact = otherone.unit_values/self.unit_values
                tmpsf.values += otherone.values*fact.asNumber()
                tmpsf.mask = np.logical_or(self.mask, otherone.mask)
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
                new_ind_value = np.logical_and(new_ind_X, new_ind_Y)
                new_ind_Yo, new_ind_Xo = np.meshgrid(new_ind_yo, new_ind_xo)
                new_ind_valueo = np.logical_and(new_ind_Xo, new_ind_Yo)
                # getting new axis and values
                new_axis_x = self.axis_x[new_ind_x]
                new_axis_y = self.axis_y[new_ind_y]
                fact = otherone.unit_values/self.unit_values
                new_values = (self.values[new_ind_value] +
                              otherone.values[new_ind_valueo] *
                              fact.asNumber())
                new_values = new_values.reshape((len(new_axis_x),
                                                 len(new_axis_y)))
                new_mask = np.logical_or(self.mask[new_ind_value],
                                         otherone.mask[new_ind_valueo])
                new_mask = new_mask.reshape((len(new_axis_x), len(new_axis_y)))
                # creating sf
                tmpsf = ScalarField(new_axis_x, new_axis_y, new_values,
                                    mask=new_mask, unit_x=self.unit_x,
                                    unit_y=self.unit_y,
                                    unit_values=self.unit_values)
            return tmpsf
        # if we add with a unit object
        elif isinstance(otherone, unum.Unum):
            try:
                self.unit_values + otherone
            except unum.IncompatibleUnitsError as m:
                raise ValueError("Units don't match: {}".format(m))
            tmpsf = self.copy()
            tmpsf.values += (otherone/self.unit_values).asNumber()
            return tmpsf
        else:
            try:
                tmpsf = self.copy()
                tmpsf.values += otherone
                return tmpsf
            except TypeError:
                raise TypeError("You can only add a scalarfield "
                                "with others scalarfields or numbers,"
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
            tmpsf = self.copy()
            unit_values = tmpsf.unit_values / obj
            tmpsf.values *= unit_values.asNumber()
            unit_values /= unit_values.asNumber()
            tmpsf.unit_values = unit_values
            return tmpsf
        # other scalarfield
        if isinstance(obj, ScalarField):
            if np.any(self.axis_x != obj.axis_x)\
                    or np.any(self.axis_y != obj.axis_y)\
                    or self.unit_x != obj.unit_x\
                    or self.unit_y != obj.unit_y:
                raise ValueError("Fields are not consistent")
            tmpsf = self.copy()
            filt_nan = obj.values != 0
            values = np.zeros(shape=self.values.shape)
            values[filt_nan] = self.values[filt_nan]/obj.values[filt_nan]
            mask = np.logical_or(self.mask, obj.mask)
            mask = np.logical_or(mask, np.logical_not(filt_nan))
            unit = self.unit_values / obj.unit_values
            tmpsf.values = values*unit.asNumber()
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
            values = tmpsf.values
            values[not_mask] /= obj[not_mask]
            tmpsf.values = values
            tmpsf.mask = mask
            return tmpsf
        except TypeError:
            pass
        # number
        try:
            tmpsf = self.copy()
            tmpsf.values /= obj
            return tmpsf
        except:
            pass
        # else...
        raise TypeError("Unsupported operation between {} and a "
                        "ScalarField object".format(type(obj)))

    __div__ = __truediv__

    def __rtruediv__(self, obj):
        tmpsf = self.copy()
        # units object
        if isinstance(obj, unum.Unum):
            tmpsf.values = obj.asNumber()/tmpsf.values
            tmpsf.unit_values = obj/obj.asNumber()/tmpsf.unit_values
            return tmpsf
        # # scalarfield
        # if isinstance(obj, ScalarField):
        #     if np.any(self.axis_x != obj.axis_x)\
        #        or np.any(self.axis_y != obj.axis_y)\
        #        or self.unit_x != obj.unit_x\
        #        or self.unit_y != obj.unit_y:
        #         raise ValueError("Fields are not consistent")
        #     values = obj.values / self.values
        #     mask = np.logical_or(self.mask, obj.mask)
        #     unit = obj.unit_values / self.unit_values
        #     tmpsf.values = values*unit.asNumber()
        #     tmpsf.mask = mask
        #     tmpsf.unit_values = unit/unit.asNumber()
        #     return tmpsf
        # array
        try:
            obj[0]
            obj = np.array(obj, subok=True)
            if not obj.shape == self.shape:
                raise ValueError()
            mask = np.logical_or(self.mask, obj == 0)
            not_mask = np.logical_not(mask)
            values = tmpsf.values
            values[not_mask] = obj[not_mask] / tmpsf.values[not_mask]
            tmpsf.values = values
            tmpsf.mask = mask
            return tmpsf
        except TypeError:
            pass
        # number
        try:
            tmpsf.values = obj/tmpsf.values
            tmpsf.unit_values = 1/tmpsf.unit_values
            return tmpsf
        except:
            raise TypeError("Unsupported operation between {} and a "
                            "ScalarField object".format(type(obj)))

    def __mul__(self, obj):
        # units
        if isinstance(obj, unum.Unum):
            tmpsf = self.copy()
            tmpsf.values *= obj.asNumber()
            tmpsf.unit_values *= obj/obj.asNumber()
            tmpsf.mask = self.mask
            return tmpsf
        # sclarfield
        if isinstance(obj, ScalarField):
            if np.any(self.axis_x != obj.axis_x)\
                    or np.any(self.axis_y != obj.axis_y)\
                    or self.unit_x != obj.unit_x\
                    or self.unit_y != obj.unit_y:
                raise ValueError("Fields are not consistent")
            tmpsf = self.copy()
            values = self.values * obj.values
            mask = np.logical_or(self.mask, obj.mask)
            unit = self.unit_values * obj.unit_values
            tmpsf.values = values*unit.asNumber()
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
            values = tmpsf.values
            values[not_mask] *= obj[not_mask]
            tmpsf.values = values
            tmpsf.mask = mask
            return tmpsf
        except TypeError:
            pass
        # numbers
        try:
            tmpsf = self.copy()
            tmpsf.values *= obj
            tmpsf.mask = self.mask
            return tmpsf
        except:
            raise TypeError("Unsupported operation between {} and a "
                            "ScalarField object".format(type(obj)))
    __rmul__ = __mul__

    def __abs__(self):
        tmpsf = self.copy()
        tmpsf.values = np.abs(tmpsf.values)
        return tmpsf

    def __pow__(self, number):
        tmpsf = self.copy()
        tmpsf.values[np.logical_not(tmpsf.mask)] \
            = np.power(tmpsf.values[np.logical_not(tmpsf.mask)], number)
        tmpsf.mask = self.mask
        tmpsf.unit_values = np.power(tmpsf.unit_values, number)
        return tmpsf

    def __iter__(self):
        data = self.values
        mask = self.mask
        for ij, xy in fld.Field.__iter__(self):
            i = ij[0]
            j = ij[1]
            if not mask[i, j]:
                yield ij, xy, data[i, j]

    def __repr__(self):
        return self.get_props()

    @property
    def min(self):
        return np.min(self.values[np.logical_not(self.mask)])

    @property
    def max(self):
        return np.max(self.values[np.logical_not(self.mask)])

    @property
    def mean(self):
        return np.mean(self.values[np.logical_not(self.mask)])

    def get_props(self):
        """
        Print the ScalarField main properties.
        """
        text = "Shape: {}\n".format(self.shape)
        unit_x = self.unit_x.strUnit()
        text += "Axis x: [{}..{}]{}\n".format(self.axis_x[0], self.axis_x[-1],
                                            unit_x)
        unit_y = self.unit_y.strUnit()
        text += "Axis y: [{}..{}]{}\n".format(self.axis_y[0], self.axis_y[-1],
                                            unit_y)
        unit_values = self.unit_values.strUnit()
        text += "Values: [{}..{}]{}\n".format(self.min, self.max, unit_values)
        nmb_mask = np.sum(self.mask)
        nmb_tot = self.shape[0]*self.shape[1]
        text += "Masked values: {}/{}\n".format(nmb_mask, nmb_tot)
        return text

    def get_value(self, x, y, ind=False, unit=False):
        """
        Return the scalar field value on the point (x, y).
        If ind is true, x and y are indices,
        else, x and y are value on axis (interpolated if necessary).
        """
        # check
        if ind:
            if x > len(self.axis_x) - 1 or y > len(self.axis_y) - 1\
                    or x < 0 or y < 0:
                raise ValueError("indices out of bound.")
        else:
            if x > self.axis_x[-1] or y > self.axis_y[-1]\
                    or x < self.axis_x[0] or y < self.axis_y[0]:
                raise ValueError("x or y value out of bound.")
        # unit or not ?
        if unit:
            unit = self.unit_values
        else:
            unit = 1.
        # if ind is true, it's easy
        if ind:
            return self.values[x, y]*unit
        # else, interpolate
        else:
            ind_x = None
            ind_y = None
            # getting indices interval
            inds_x = self.get_indice_on_axis('x', x)
            inds_y = self.get_indice_on_axis('y', y)
            # if something masked
            if np.sum(self.mask[inds_x, inds_y]) != 0:
                res = np.NaN
            # if we are on a grid point
            elif inds_x[0] == inds_x[1] and inds_y[0] == inds_y[1]:
                res = self.values[inds_x[0], inds_y[0]]*unit
            # if we are on a x grid branch
            elif inds_x[0] == inds_x[1]:
                ind_x = inds_x[0]
                pos_y1 = self.axis_y[inds_y[0]]
                pos_y2 = self.axis_y[inds_y[1]]
                value1 = self.values[ind_x, inds_y[0]]
                value2 = self.values[ind_x, inds_y[1]]
                i_value = ((value2*np.abs(pos_y1 - y) +
                           value1*np.abs(pos_y2 - y)) /
                           np.abs(pos_y1 - pos_y2))
                res = i_value*unit
            # if we are on a y grid branch
            elif inds_y[0] == inds_y[1]:
                ind_y = inds_y[0]
                pos_x1 = self.axis_x[inds_x[0]]
                pos_x2 = self.axis_x[inds_x[1]]
                value1 = self.values[inds_x[0], ind_y]
                value2 = self.values[inds_x[1], ind_y]
                i_value = ((value2*np.abs(pos_x1 - x) +
                            value1*np.abs(pos_x2 - x)) /
                           np.abs(pos_x1 - pos_x2))
                return i_value*unit
            # if we are in the middle of nowhere (linear interpolation)
            else:
                ind_x = inds_x[0]
                ind_y = inds_y[0]
                a, b = np.meshgrid(self.axis_x[ind_x:ind_x + 2],
                                   self.axis_y[ind_y:ind_y + 2], indexing='ij')
                values = self.values[ind_x:ind_x + 2, ind_y:ind_y + 2]
                a = a.flatten()
                b = b.flatten()
                pts = list(zip(a, b))
                interp_vx = spinterp.LinearNDInterpolator(pts,
                                                          values.flatten())
                i_value = float(interp_vx(x, y))
                res = i_value*unit
            return res

    # def get_zones_centers(self, bornes=[0.75, 1], rel=True,
    #                       kind='ponderated'):
    #     """
    #     Return a pts.Points object contening centers of the zones
    #     lying in the given bornes.

    #     Parameters
    #     ----------
    #     bornes : 2x1 array, optionnal
    #         Trigger values determining the zones.
    #         '[inferior borne, superior borne]'
    #     rel : Boolean
    #         If 'rel' is 'True' (default), values of 'bornes' are relative to
    #         the extremum values of the field.
    #         If 'rel' is 'False', values of bornes are treated like absolute
    #         values.
    #     kind : string, optional
    #         if 'kind' is 'center', given points are geometrical centers,
    #         if 'kind' is 'extremum', given points are
    #         extrema (min or max) on zones
    #         if 'kind' is 'ponderated'(default, given points are centers of
    #         mass, ponderated by the scaler field.

    #     Returns
    #     -------
    #     pts : pts.Points object
    #         Contening the centers coordinates
    #     """
    #     # correcting python's problem with egality...
    #     bornes[0] -= 0.00001*abs(bornes[0])
    #     bornes[1] += 0.00001*abs(bornes[1])
    #     # checking parameters coherence
    #     if not isinstance(bornes, ARRAYTYPES):
    #         raise TypeError("'bornes' must be an array")
    #     if not isinstance(bornes, np.ndarray):
    #         bornes = np.array(bornes, dtype=float)
    #     if not bornes.shape == (2,):
    #         raise ValueError("'bornes' must be a 2x1 array")
    #     if bornes[0] == bornes[1]:
    #         return None
    #     if not bornes[0] < bornes[1]:
    #         raise ValueError("'bornes' must be crescent")
    #     if not isinstance(rel, bool):
    #         raise TypeError("'rel' must be a boolean")
    #     if not isinstance(kind, STRINGTYPES):
    #         raise TypeError("'kind' must be a string")
    #     # compute minimum and maximum if 'rel=True'
    #     if rel:
    #         if bornes[0]*bornes[1] < 0:
    #             raise ValueError("In relative 'bornes' must have the same"
    #                              " sign")
    #         mini = self.min
    #         maxi = self.max
    #         if np.abs(bornes[0]) > np.abs(bornes[1]):
    #             bornes[1] = abs(maxi - mini)*bornes[1] + maxi
    #             bornes[0] = abs(maxi - mini)*bornes[0] + maxi
    #         else:
    #             bornes[1] = abs(maxi - mini)*bornes[1] + mini
    #             bornes[0] = abs(maxi - mini)*bornes[0] + mini
    #     # check if the zone exist
    #     else:
    #         mini = self.min
    #         maxi = self.max
    #         if maxi < bornes[0] or mini > bornes[1]:
    #             return None
    #     # getting data
    #     values = self.values
    #     mask = self.mask
    #     if np.any(mask):
    #         warnings.warn("There is masked values, algorithm can give "
    #                       "strange results")
    #     # check if there is more than one point superior
    #     aoi = np.logical_and(values >= bornes[0], values <= bornes[1])
    #     if np.sum(aoi) == 1:
    #         inds = np.where(aoi)
    #         x = self.axis_x[inds[0][0]]
    #         y = self.axis_y[inds[1][0]]
    #         return pts.Points([[x, y]], unit_x=self.unit_x,
    #                           unit_y=self.unit_y)
    #     zones = np.logical_and(np.logical_and(values >= bornes[0],
    #                                           values <= bornes[1]),
    #                            np.logical_not(mask))
    #     # compute the center with labelzones
    #     labeledzones, nmbzones = msr.label(zones)
    #     inds = []
    #     if kind == 'extremum':
    #         mins, _, ind_min, ind_max = msr.extrema(values,
    #                                                 labeledzones,
    #                                                 np.arange(nmbzones) + 1)
    #         for i in np.arange(len(mins)):
    #             if bornes[np.argmax(np.abs(bornes))] < 0:
    #                 inds.append(ind_min[i])
    #             else:
    #                 inds.append(ind_max[i])
    #     elif kind == 'center':
    #         inds = msr.center_of_mass(np.ones(self.shape),
    #                                   labeledzones,
    #                                   np.arange(nmbzones) + 1)
    #     elif kind == 'ponderated':
    #         inds = msr.center_of_mass(np.abs(values), labeledzones,
    #                                   np.arange(nmbzones) + 1)
    #     else:
    #         raise ValueError("Invalid value for 'kind'")
    #     coords = []
    #     for ind in inds:
    #         indx = ind[0]
    #         indy = ind[1]
    #         if indx % 1 == 0:
    #             x = self.axis_x[int(indx)]
    #         else:
    #             dx = self.axis_x[1] - self.axis_x[0]
    #             x = self.axis_x[int(indx)] + dx*(indx % 1)
    #         if indy % 1 == 0:
    #             y = self.axis_y[int(indy)]
    #         else:
    #             dy = self.axis_y[1] - self.axis_y[0]
    #             y = self.axis_y[int(indy)] + dy*(indy % 1)
    #         coords.append([x, y])
    #     coords = np.array(coords, dtype=float)
    #     if len(coords) == 0:
    #         return None
    #     return pts.Points(coords, unit_x=self.unit_x, unit_y=self.unit_y)

    # def get_nearest_extrema(self, pts, extrema='max', ind=False):
    #     """
    #     For a given set of points, return the positions of the nearest local
    #     extrema (minimum or maximum).

    #     Parameters
    #     ----------
    #     pts : Nx2 array
    #         Set of pts.Points position.

    #     Returns
    #     -------
    #     extremum_pos : Nx2 array
    #     """
    #     # get data
    #     tmp_sf = self.copy()
    #     tmp_sf.mirroring(direction='x', position=tmp_sf.axis_x[0],
    #                      inds_to_mirror=1, inplace=True)
    #     tmp_sf.mirroring(direction='x', position=tmp_sf.axis_x[-1],
    #                      inds_to_mirror=1, inplace=True)
    #     tmp_sf.mirroring(direction='y', position=tmp_sf.axis_y[0],
    #                      inds_to_mirror=1, inplace=True)
    #     tmp_sf.mirroring(direction='y', position=tmp_sf.axis_y[-1],
    #                      inds_to_mirror=1, inplace=True)
    #     dx = tmp_sf.axis_x[1] - tmp_sf.axis_x[0]
    #     dy = tmp_sf.axis_y[1] - tmp_sf.axis_y[0]
    #     # get gradient field
    #     grad_x, grad_y = np.gradient(tmp_sf.values, dx, dy)
    #     from . import vectorfield as vf
    #     tmp_vf = vf.VectorField()
    #     tmp_vf.import_from_arrays(tmp_sf.axis_x, tmp_sf.axis_y, grad_x, grad_y,
    #                               unit_x=tmp_sf.unit_x, unit_y=tmp_sf.unit_y,
    #                               unit_values=tmp_sf.unit_values)
    #     # extract the streamline from the gradient field
    #     from ..field_treatment import get_streamlines
    #     if extrema == 'min':
    #         reverse = True
    #     else:
    #         reverse = False
    #     sts = get_streamlines(tmp_vf, pts, reverse=reverse, resolution=0.1)
    #     # get the final converged points
    #     extremum_pos = []
    #     if isinstance(sts, ARRAYTYPES):
    #         for i, st in enumerate(sts):
    #             if len(st.xy) == 0:
    #                 extremum_pos.append(pts[i])
    #             else:
    #                 extremum_pos.append(st.xy[-1])
    #     else:
    #         extremum_pos.append(sts.xy[-1])
    #     extremum_pos = np.array(extremum_pos)
    #     # returning
    #     return extremum_pos

    def get_profile(self, x=None, y=None, ind=False, interp='linear'):
        """
        Return a profile of the scalar field, at the given position.
        If position is an interval, the fonction return an average profile
        in this interval.

        Parameters
        ----------
        x, y: numbers or 2x1 array of numbers
            Position of the wanted profile.
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
        # checking parameters
        if x is not None and y is not None:
            raise ValueError('You can only specify x or y')
        if x is None and y is None:
            raise ValueError('You have to specify x or y')
        # getting data
        if x is not None:
            axis = self.axis_x
            oaxis = self.axis_y
            unit = self.unit_y
            pos = x
        else:
            axis = self.axis_y
            oaxis = self.axis_x
            unit = self.unit_x
            pos = y
        # Checks
        pos_array = True
        try:
            pos[0]
        except TypeError:
            pos_array = False
        if pos_array:
            if ind:
                for pos in pos:
                    if pos > axis.max():
                        pos = axis.max()
                    if pos < axis.min():
                        pos = axis.min()
            else:
                if np.min(pos) < -len(axis) + 1 or \
                   np.max(pos) > len(axis) - 1:
                    raise ValueError("'position' must be included in"
                                     " the choosen axis values")
        else:
            if ind:
                if pos > axis.max() or pos < axis.min():
                    raise ValueError("'position' must be included in the "
                                     "choosen axis values (here [{0},{1}])"
                                     .format(axis.min(), axis.max()))
            else:
                if np.min(pos) < 0 or np.max(pos) > len(axis) - 1:
                    raise ValueError("'position' must be included in the"
                                     "choosen axis values (here [{0},{1}])"
                                     .format(0, len(axis) - 1))
        # Get profile for linear interpolation
        if not pos_array and interp == 'linear':
            if ind:
                position = self.axis_x[pos]
            vals = [self.get_value(position, axis_i) for axis_i in oaxis]
            tmp_prof = prf.Profile(x=axis, y=vals, mask=False,
                                   unit_x=unit,
                                   unit_y=self.unit_values)
            return tmp_prof
        # Get profile for other interpolation
        if not pos_array:
            if not ind:
                for i in np.arange(1, len(axis)):
                    if (axis[i] >= position and axis[i-1] <= position) \
                            or (axis[i] <= position and axis[i-1] >= position):
                        break
                if np.abs(position - axis[i]) > np.abs(position - axis[i-1]):
                    finalindice = i-1
                else:
                    finalindice = i
                if x is not None:
                    prof_mask = self.mask[finalindice, :]
                    profile = self.values[finalindice, :]
                    axis = self.axis_y
                else:
                    prof_mask = self.mask[:, finalindice]
                    profile = self.values[:, finalindice]
                    axis = self.axis_x
            else:
                if x is not None:
                    prof_mask = self.mask[position, :]
                    profile = self.values[position, :]
                    axis = self.axis_y
                else:
                    prof_mask = self.mask[:, position]
                    profile = self.values[:, position]
                    axis = self.axis_x
        # Calculation of the profile for an interval of position
        else:
            if not ind:
                axis_mask = np.logical_and(axis >= position[0],
                                           axis <= position[1])
                if x is not None:
                    prof_mask = self.mask[axis_mask, :].mean(0)
                    profile = self.values[axis_mask, :].mean(0)
                    axis = self.axis_y
                else:
                    prof_mask = self.mask[:, axis_mask].mean(1)
                    profile = self.values[:, axis_mask].mean(1)
                    axis = self.axis_x
            else:
                if x is not None:
                    prof_mask = self.mask[position[0]:position[1], :].mean(0)
                    profile = self.values[position[0]:position[1], :].mean(0)
                    axis = self.axis_y
                else:
                    prof_mask = self.mask[:, position[0]:position[1]].mean(1)
                    profile = self.values[:, position[0]:position[1]].mean(1)
                    axis = self.axis_x
        return prf.Profile(axis, profile, prof_mask, unit, self.unit_values)

    def get_histogram(self, bins=200, cum=False, normalized=False):
        """
        Return the scalarfield values histogram.

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
        hist: Profile object.
            Histogram.
        """
        hist, xs = np.histogram(self.values.flatten(),
                                bins=bins,
                                density=normalized)
        xs = xs[0:-1] + np.mean(xs[0:2])
        if cum:
            hist = np.cumsum(hist)
        return prf.Profile(xs, hist, mask=False, unit_x=self.unit_values,
                           unit_y="counts")

    def get_interpolator(self, interp="linear"):
        """
        Return the field interpolator.

        Parameters
        ----------
        kind : {‘linear’, ‘cubic’, ‘quintic’}, optional
            The kind of spline interpolation to use. Default is ‘linear’.

        Example
        -------
        >>> interp = SF.get_interpolator(interp='linear')
        >>> print(interp(4, 5.3))
        ... [34.3]
        >>> print(interp([3, 4, 5], 5.3))
        ... [23, 34.3, 54]
        """
        return spinterp.interp2d(self.axis_x, self.axis_y,
                                 self.values.transpose(),
                                 kind=interp)

    def integrate(self):
        """
        Return the integral of the field.
        If you want the integral on a subset of the field, use 'crop' before.

        Returns
        -------
        integral : float
            Result of the integrale computation.
        unit : Unit object
            The unit of the integrale result.

        Note
        ----
        Discretized integral is computed with a very rustic algorithm
        which just sum the value on the surface.
        """
        if np.any(self.mask):
            raise Exception("Masked values on the surface")
        integral = (self.values.sum() *
                    np.abs(self.axis_x[-1] - self.axis_x[0]) *
                    np.abs(self.axis_y[-1] - self.axis_y[0]) /
                    len(self.axis_x) /
                    len(self.axis_y))
        unit = self.unit_values*self.unit_x*self.unit_y
        return integral, unit

    def copy(self):
        """
        Return a copy of the scalarfield.
        """
        return copy.deepcopy(self)

    def scale(self, scalex=None, scaley=None, scalev=None, inplace=False):
        """
        Scale the ScalarField.

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
        # use Field method to scale the axis
        revx, revy = fld.Field.scale(tmp_f, scalex=scalex, scaley=scaley,
                                     inplace=True, output_reverse=True)
        # scale the values
        if scalev is None:
            pass
        elif isinstance(scalev, unum.Unum):
            new_unit = tmp_f.unit_values*scalev
            fact = new_unit.asNumber()
            new_unit /= fact
            tmp_f.unit_values = new_unit
            tmp_f.values *= fact
        else:
            tmp_f.values *= scalev
        if revx and revy:
            tmp_f.values = tmp_f.values[::-1, ::-1]
        elif revx:
            tmp_f.values = tmp_f.values[::-1, :]
        elif revy:
            tmp_f.values = tmp_f.values[:, ::-1]
        # returning
        if not inplace:
            return tmp_f

    def rotate(self, angle, inplace=False):
        """
        Rotate the scalar field.

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
        values = tmp_field.values
        tmp_field.__values = np.rot90(values, nmb_rot90)
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
        if not axis in ['x', 'y', 'values']:
            raise TypeError("'axis' should be 'x', 'y', or 'values'")
        if axis == 'x':
            fld.Field.change_unit(self, axis, new_unit)
        elif axis == 'y':
            fld.Field.change_unit(self, axis, new_unit)
        elif axis == 'values':
            old_unit = self.unit_values
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.values *= fact
            self.unit_values = new_unit/fact
        else:
            raise ValueError()

    def crop(self, intervx=None, intervy=None, ind=False,
             inplace=False):
        """
        Crop the scalar field.

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
            values = self.values
            mask = self.mask
            indmin_x, indmax_x, indmin_y, indmax_y = \
                fld.Field.crop(self, intervx, intervy, full_output=True,
                               ind=ind, inplace=True)
            self.__values = values[indmin_x:indmax_x + 1,
                                   indmin_y:indmax_y + 1]
            self.__mask = mask[indmin_x:indmax_x + 1,
                               indmin_y:indmax_y + 1]
        else:
            indmin_x, indmax_x, indmin_y, indmax_y, cropfield = \
                fld.Field.crop(self, intervx=intervx, intervy=intervy,
                               full_output=True, ind=ind, inplace=False)
            cropfield.__values = self.values[indmin_x:indmax_x + 1,
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
            tmp_sf = self
        else:
            tmp_sf = self.copy()
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
        fld.Field.extend(tmp_sf, nmb_left=nmb_left,
                                 nmb_right=nmb_right, nmb_up=nmb_up,
                                 nmb_down=nmb_down, inplace=True)
        new_shape = tmp_sf.shape
        # extend the value ans mask
        if value is None:
            new_values = np.zeros(new_shape, dtype=float)
            new_mask = np.ones(new_shape, dtype=bool)
        else:
            new_values = np.ones(new_shape, dtype=float)*value
            new_mask = np.zeros(new_shape, dtype=bool)
        if nmb_right == 0:
            slice_x = slice(nmb_left, new_values.shape[0] + 2)
        else:
            slice_x = slice(nmb_left, -nmb_right)
        if nmb_up == 0:
            slice_y = slice(nmb_down, new_values.shape[1] + 2)
        else:
            slice_y = slice(nmb_down, -nmb_up)
        new_values[slice_x, slice_y] = self.values
        new_mask[slice_x, slice_y] = self.mask
        tmp_sf.__values = new_values
        tmp_sf.__mask = new_mask
        # return
        if not inplace:
            return tmp_sf

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
        Fill the masked parts of the scalar field.

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
            If 'True', fill the ScalarField in place.
            If 'False' (default), return a filled version of the field.
        reduce_tri : boolean, optional
            If 'True', treatment is used to reduce the triangulation effort
            (faster when a lot of masked values)
            If 'False', no treatment
            (faster when few masked values)
        crop : boolean, optional
            If 'True', SF borders are croped before filling.
                """
        #
        if inplace:
            tmp_sf = self
        else:
            tmp_sf = self.copy()
        # check parameters coherence
        if kind not in ['linear', 'value', 'nearest', 'cubic']:
            raise TypeError("'kind' must be 'linear', 'value', "
                            "'nearest' or 'cubic'")
        if crop:
            tmp_sf.crop_masked_border(hard=False, inplace=True)
        # if there is nothing to do...
        if not np.any(tmp_sf.mask):
            return self
        # getting data
        x, y = tmp_sf.axis_x, tmp_sf.axis_y
        values = tmp_sf.values
        mask = tmp_sf.mask
        if kind in ['nearest', 'linear', 'cubic']:
            X, Y = np.meshgrid(x, y, indexing='ij')
            xy = [X.flat[:], Y.flat[:]]
            xy = np.transpose(xy)
            filt = np.logical_not(mask)
            xy_masked = xy[mask.flatten()]
        # getting the zone to interpolate
        if reduce_tri and kind in ['nearest', 'linear', 'cubic']:
            import scipy.ndimage as spim
            dilated = spim.binary_dilation(tmp_sf.mask,
                                           np.arange(9).reshape((3, 3)))
            filt_good = np.logical_and(filt, dilated)
            xy_good = xy[filt_good.flatten()]
            values_good = values[filt_good]
        elif not reduce_tri and kind in ['nearest', 'linear', 'cubic']:
            xy_good = xy[filt.flatten()]
            values_good = values[filt]
        else:
            pass
        # if interpolation
        if kind == 'value':
            values[mask] = value
        elif kind == 'nearest':
            nearest = spinterp.NearestNDInterpolator(xy_good, values_good)
            values[mask] = nearest(xy_masked)
        elif kind == 'linear':
            linear = spinterp.LinearNDInterpolator(xy_good, values_good)
            values[mask] = linear(xy_masked)
            new_mask = np.isnan(values)
            if np.any(new_mask):
                nearest = spinterp.NearestNDInterpolator(xy_good, values_good)
                values[new_mask] = nearest(xy[new_mask.flatten()])
        elif kind == 'cubic':
            cubic = spinterp.CloughTocher2DInterpolator(xy_good, values_good)
            values[mask] = cubic(xy_masked)
            new_mask = np.isnan(values)
            if np.any(new_mask):
                nearest = spinterp.NearestNDInterpolator(xy_good, values_good)
                values[new_mask] = nearest(xy[new_mask.flatten()])
        # returning
        tmp_sf.__mask = False
        tmp_sf.__values = values
        if not inplace:
            return tmp_sf

    def smooth(self, tos='uniform', size=None, inplace=False, **kw):
        """
        Smooth the scalarfield.

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
            tmp_sf = self
        else:
            tmp_sf = self.copy()
        if tos not in ['uniform', 'gaussian']:
            raise TypeError("'tos' must be 'uniform' or 'gaussian'")
        if size is None and tos == 'uniform':
            size = 3
        elif size is None and tos == 'gaussian':
            size = 1
        # filling up the field before smoothing
        self.fill(inplace=True)
        values = tmp_sf.values
        # smoothing
        if tos == "uniform":
            values = ndimage.uniform_filter(values, size, **kw)
        elif tos == "gaussian":
            values = ndimage.gaussian_filter(values, size, **kw)
        # storing
        if inplace:
            self.values = values
        else:
            tmp_sf.values = values
            return tmp_sf

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
            If True, modify the scalar field in place, else, return a
            modified version of it.
        """
        if inplace:
            tmp_sf = self
        else:
            tmp_sf = self.copy()
            # get data
        axisx = tmp_sf.axis_x
        axisy = tmp_sf.axis_y
        dx = np.min(axisx[1:] - axisx[:-1])/res
        dy = np.min(axisy[1:] - axisy[:-1])/res
        Dx = axisx[-1] - axisx[0]
        Dy = axisy[-1] - axisy[0]
        #
        interp = tmp_sf.get_interpolator(interp=interp)
        new_x = np.linspace(axisx[0], axisx[-1], int(Dx/dx))
        new_y = np.linspace(axisy[0], axisy[-1], int(Dy/dy))
        new_values = interp(new_x, new_y)
        # store
        tmp_sf.__init__(new_x, new_y, new_values.transpose(),
                        mask=False, unit_x=tmp_sf.unit_x,
                        unit_y=tmp_sf.unit_y,
                        unit_values=tmp_sf.unit_values)
        if not inplace:
            return tmp_sf

    def reduce_resolution(self, fact, inplace=False):
        """
        Reduce the spatial resolution of the scalar field by a factor 'fact'.

        Parameters
        ----------
        fact : int
            Reducing factor.
        inplace : boolean, optional
            .
        """
        if inplace:
            tmp_sf = self
        else:
            tmp_sf = self.copy()
        #
        fact = int(fact)
        if fact < 1:
            raise ValueError()
        if fact == 1:
            if inplace:
                pass
            else:
                return tmp_sf
        if fact % 2 == 0:
            pair = True
        else:
            pair = False
        # get new axis
        axis_x = tmp_sf.axis_x
        axis_y = tmp_sf.axis_y
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
        values = tmp_sf.values
        mask = tmp_sf.mask
        if pair:
            inds_x = np.arange(fact/2, len(axis_x) - fact/2 + 1,
                               fact, dtype=int)
            inds_y = np.arange(fact/2, len(axis_y) - fact/2 + 1,
                               fact, dtype=int)
            new_values = np.zeros((len(inds_x), len(inds_y)))
            new_mask = np.zeros((len(inds_x), len(inds_y)))
            for i in np.arange(len(inds_x)):
                intervx = slice(inds_x[i] - int(fact/2),
                                inds_x[i] + int(fact/2))
                for j in np.arange(len(inds_y)):
                    intervy = slice(inds_y[j] - int(fact/2),
                                    inds_y[j] + int(fact/2))
                    if np.all(mask[intervx, intervy]):
                        new_mask[i, j] = True
                        new_values[i, j] = 0.
                    else:
                        new_values[i, j] = np.mean(values[intervx, intervy]
                                                   [~mask[intervx, intervy]])

        else:
            inds_x = np.arange((fact - 1)/2, len(axis_x) - (fact - 1)/2, fact)
            inds_y = np.arange((fact - 1)/2, len(axis_y) - (fact - 1)/2, fact)
            new_values = np.zeros((len(inds_x), len(inds_y)))
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
                        new_values[i, j] = 0.
                    else:
                        new_values[i, j] = np.mean(values[intervx, intervy]
                                                   [~mask[intervx, intervy]])
        # returning
        tmp_sf.__init__(new_axis_x, new_axis_y, new_values,
                        mask=new_mask,
                        unit_x=tmp_sf.unit_x,
                        unit_y=tmp_sf.unit_y,
                        unit_values=tmp_sf.unit_values)
        if not inplace:
            return tmp_sf

    def display(self, component='values', kind='imshow',
                annotate=True, **plotargs):
        """
        Display the scalar field.

        Parameters
        ----------
        component : string, optional
            Component to display, can be 'values' or 'mask'
        kind : string, optinnal
            If 'imshow': (default) each datas are plotted (imshow),
            if 'contour': contours are ploted (contour),
            if 'contourf': filled contours are ploted (contourf).
        annotate: boolean
            If True (default) add label and legedn to the graph.
        **plotargs : dict
            Arguments passed to the plotting function.

        Returns
        -------
        fig : figure reference
            Reference to the displayed figure.
        """
        # check
        if component not in ['values', 'mask']:
            raise ValueError("'component' must be 'values' or 'mask'")
        if kind not in ['imshow', 'contour', 'contourf']:
            raise ValueError("'kind' must be 'imshow', 'contour', "
                             "or 'contourf'")
        # getting datas
        axis_x, axis_y = self.axis_x, self.axis_y
        dx, dy = self.dx, self.dy
        unit_x, unit_y = self.unit_x, self.unit_y
        X, Y = np.meshgrid(self.axis_y, self.axis_x)
        # getting wanted component
        if component is None or component == 'values':
            values = self.values.astype(dtype=float)
        elif component == 'mask':
            values = self.mask
        # display data
        if kind == 'imshow':
            plot = plt.imshow(values.transpose(),
                              extent=(axis_x[0] - dx/2., axis_x[-1] + dx/2.,
                                      axis_y[0] - dy/2., axis_y[-1] + dy/2.),
                              origin='lower', **plotargs)
        elif kind == 'contour':
            plot = plt.contour(axis_x, axis_y, values.transpose(), **plotargs)
        elif kind == 'contourf':
            plot = plt.contourf(axis_x, axis_y, values.transpose(), **plotargs)
        # annotate
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
                cb = plt.colorbar(plot)
                if self.unit_values.strUnit() == "[]":
                    cb.set_label("Values")
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
        return plt.gca()
