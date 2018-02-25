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

import numpy as np
import unum
import copy
from ..utils import make_unit


class Field(object):

    def __init__(self, axis_x, axis_y, unit_x="", unit_y=""):
        # initialize
        self.__axis_x = np.array([], dtype=float)
        self.__axis_y = np.array([], dtype=float)
        self.__is_axis_x_regular = None
        self.__is_axis_y_regular = None
        self.__dx = None
        self.__dy = None
        self.__shape = [None, None]
        self.__unit_x = make_unit('')
        self.__unit_y = make_unit('')
        # fill
        self.axis_x = axis_x
        self.axis_y = axis_y
        self.unit_x = unit_x
        self.unit_y = unit_y

    def __iter__(self):
        for i, x in enumerate(self.axis_x):
            for j, y in enumerate(self.axis_y):
                yield [i, j], [x, y]

    @property
    def axis_x(self):
        return self.__axis_x

    @axis_x.setter
    def axis_x(self, new_axis_x):
        new_axis_x = np.sort(np.asarray(new_axis_x, dtype=float))
        if new_axis_x.shape == self.__axis_x.shape or len(self.__axis_x) == 0:
            # update length if necessary
            if len(self.__axis_x) == 0:
                self.__shape[0] = len(new_axis_x)
            # update axis values
            self.__axis_x = new_axis_x
            # check if regular
            dxs = new_axis_x[1:] - new_axis_x[0:-1]
            self.__is_axis_x_regular = np.allclose(dxs, dxs[0])
            if self.__is_axis_x_regular:
                self.__dx = dxs[0]
            else:
                self.__dx = None
        else:
            raise ValueError('New axis has to be of the same size of the old'
                             f' one ({len(self.axis_x)}), but is of size'
                             f' {len(new_axis_x)}.')

    @axis_x.deleter
    def axis_x(self):
        raise Exception("Can't delete that.")

    @property
    def dx(self):
        if self.__dx is not None:
            return self.__dx
        else:
            raise Exception('dx is not defined, axis x is not regular')

    @property
    def axis_y(self):
        return self.__axis_y

    @axis_y.setter
    def axis_y(self, new_axis_y):
        new_axis_y = np.sort(np.asarray(new_axis_y, dtype=float))
        if new_axis_y.shape == self.__axis_y.shape or len(self.__axis_y) == 0:
            # update length if necessary
            if len(self.__axis_y) == 0:
                self.__shape[1] = len(new_axis_y)
            # update axis values
            self.__axis_y = new_axis_y
            # check if regular
            dys = new_axis_y[1:] - new_axis_y[0:-1]
            self.__is_axis_y_regular = np.allclose(dys, dys[0])
            if self.__is_axis_y_regular:
                self.__dy = dys[0]
            else:
                self.__dy = None
        else:
            raise ValueError('New axis has to be of the same size of the old'
                             f' one ({len(self.axis_y)}), but is of size'
                             f' {len(new_axis_y)}.')

    @axis_y.deleter
    def axis_y(self):
        raise Exception("Can't delete that.")

    @property
    def dy(self):
        if self.__dy is not None:
            return self.__dy
        else:
            raise Exception('dy is not defined, axis y is not regular')

    @property
    def unit_x(self):
        return self.__unit_x

    @unit_x.setter
    def unit_x(self, new_unit_x):
        if isinstance(new_unit_x, unum.Unum):
            if np.isclose(new_unit_x.asNumber(), 1):
                self.__unit_x = new_unit_x
            else:
                raise ValueError('New unity value is not 1')
        else:
            try:
                self.__unit_x = make_unit(new_unit_x)
            except TypeError:
                raise TypeError('Unrecognized unity representation.')

    @unit_x.deleter
    def unit_x(self):
        raise Exception("Can't delete that.")

    @property
    def unit_y(self):
        return self.__unit_y

    @unit_y.setter
    def unit_y(self, new_unit_y):
        if isinstance(new_unit_y, unum.Unum):
            if np.isclose(new_unit_y.asNumber(), 1):
                self.__unit_y = new_unit_y
            else:
                raise ValueError('New unity value is not 1')
        else:
            try:
                self.__unit_y = make_unit(new_unit_y)
            except TypeError:
                raise TypeError('Unrecognized unity representation.')

    @unit_y.deleter
    def unit_y(self):
        raise Exception("Can't delete that.")

    @property
    def shape(self):
        return self.__shape

    def __eq__(self, other):
        if not isinstance(other, Field):
            raise TypeError()
        if not np.all(self.axis_x == other.axis_x):
            return False
        if not np.all(self.axis_y == other.axis_y):
            return False
        if not np.all(self.unit_y == other.unit_y):
            return False
        if not np.all(self.unit_x == other.unit_x):
            return False
        return True

    def copy(self):
        """
        Return a copy of the Field object.
        """
        return copy.deepcopy(self)

    def get_indice_on_axis(self, direction, value, kind='bounds'):
        """
        Return, on the given axis, the indices of the positions
        surrounding the given value.

        Parameters
        ----------
        direction : string in ['x', 'y']
            Axis choice.
        value : number
        kind : string, optional
            If 'bounds' (default), return the bounding indices.
            if 'nearest', return the nearest indice
            if 'decimal', return a decimal indice (interpolated)

        Returns
        -------
        interval : 2x1 array of integer or integer
            Bounding, nearest or decimal indice.
        """
        # checks
        if direction not in ['x', 'y']:
            raise ValueError("'direction' should be 'x' or 'y', "
                             f"not '{direction}'")
        if direction == 'x':
            axis = self.axis_x
        elif direction == 'y':
            axis = self.axis_y
        if value < axis[0] or value > axis[-1]:
            raise ValueError("'value' is out of bound: "
                             f"is {value} and should be between"
                             f" {axis[0]} and {axis[-1]}.")
        if kind not in ['bounds', 'nearest', 'decimal']:
            raise ValueError("'kind' should be 'bounds', 'nearest', or "
                             f"'decimal', but is {kind}")
        # getting the bounds indices
        ind = np.searchsorted(axis, value)
        if axis[ind] == value:
            inds = [ind, ind]
        else:
            inds = [ind - 1, ind]
        # returning bounds
        if kind == 'bounds':
            return inds
        # returning nearest
        elif kind == 'nearest':
            if inds[0] == inds[1]:
                return inds[0]
            if np.abs(axis[inds[0]] - value) < np.abs(axis[inds[1]] - value):
                return inds[0]
            else:
                return inds[1]
        # returning decimal
        elif kind == 'decimal':
            if inds[0] == inds[1]:
                return inds[0]
            value_1 = axis[inds[0]]
            value_2 = axis[inds[1]]
            delta = np.abs(value_2 - value_1)
            return (inds[0]*np.abs(value - value_2)/delta +
                    inds[1]*np.abs(value - value_1)/delta)

    # def get_points_around(self, center, radius, ind=False):
    #     """
    #     Return the list of points or the scalar field that are in a circle
    #     centered on 'center' and of radius 'radius'.

    #     Parameters
    #     ----------
    #     center : array
    #         Coordonate of the center point (in axis units).
    #     radius : float
    #         radius of the cercle (in axis units).
    #     ind : boolean, optional
    #         If 'True', radius and center represent indices on the field.
    #         if 'False', radius and center are expressed in axis unities.

    #     Returns
    #     -------
    #     indices : array
    #         Array contening the indices of the contened points.
    #         [(ind1x, ind1y), (ind2x, ind2y), ...].
    #         You can easily put them in the axis to obtain points coordinates
    #     """
    #     # checking parameters
    #     if not isinstance(center, ARRAYTYPES):
    #         raise TypeError("'center' must be an array")
    #     center = np.array(center, dtype=float)
    #     if not center.shape == (2,):
    #         raise ValueError("'center' must be a 2x1 array")
    #     if not isinstance(radius, NUMBERTYPES):
    #         raise TypeError("'radius' must be a number")
    #     if not radius > 0:
    #         raise ValueError("'radius' must be positive")
    #     # getting indice data when 'ind=False'
    #     if not ind:
    #         dx = self.axis_x[1] - self.axis_x[0]
    #         dy = self.axis_y[1] - self.axis_y[0]
    #         delta = (dx + dy)/2.
    #         radius = radius/delta
    #         center_x = self.get_indice_on_axis(1, center[0], kind='decimal')
    #         center_y = self.get_indice_on_axis(2, center[1], kind='decimal')
    #         center = np.array([center_x, center_y])
    #     # pre-computing somme properties
    #     radius2 = radius**2
    #     radius_int = radius/np.sqrt(2)
    #     # isolating possibles indices
    #     inds_x = np.arange(np.int(np.ceil(center[0] - radius)),
    #                        np.int(np.floor(center[0] + radius)) + 1)
    #     inds_y = np.arange(np.int(np.ceil(center[1] - radius)),
    #                        np.int(np.floor(center[1] + radius)) + 1)
    #     inds_x, inds_y = np.meshgrid(inds_x, inds_y)
    #     inds_x = inds_x.flatten()
    #     inds_y = inds_y.flatten()
    #     # loop on possibles points
    #     inds = []
    #     for i in np.arange(len(inds_x)):
    #         x = inds_x[i]
    #         y = inds_y[i]
    #         # test if the point is in the square 'compris' in the cercle
    #         if x <= center[0] + radius_int \
    #                 and x >= center[0] - radius_int \
    #                 and y <= center[1] + radius_int \
    #                 and y >= center[1] - radius_int:
    #             inds.append([x, y])
    #         # test if the point is the center
    #         elif all([x, y] == center):
    #             pass
    #         # test if the point is in the circle
    #         elif ((x - center[0])**2 + (y - center[1])**2 <= radius2):
    #             inds.append([x, y])
    #     return np.array(inds, subok=True)

    def scale(self, scalex=None, scaley=None, inplace=False):
        """
        Scale the Field.

        Parameters
        ----------
        scalex, scaley : numbers or Unum objects
            Scale to apply on the associated axis
        inplace : boolean, optional
            If True, scale the field in place, else (default)
            return a scaled copy.
        """
        if inplace:
            tmp_f = self
        else:
            tmp_f = self.copy()
        # x
        reversex = False
        if scalex is None:
            pass
        elif isinstance(scalex, unum.Unum):
            new_unit = tmp_f.unit_x * scalex
            fact = new_unit.asNumber()
            new_unit /= fact
            tmp_f.unit_x = new_unit
            tmp_f.axis_x *= fact
            if fact < 0:
                reversex = True
        else:
            try:
                tmp_f.axis_x *= scalex
                if scalex < 0:
                    reversex = True
            except TypeError:
                raise TypeError("'scalex' should be a number or an Unum "
                                f"object, but is currently {scalex}")
        if reversex:
            tmp_f.axis_x = tmp_f.axis_x[::-1]
        # y
        reversey = False
        if scaley is None:
            pass
        elif isinstance(scaley, unum.Unum):
            new_unit = tmp_f.unit_y*scaley
            fact = new_unit.asNumber()
            new_unit /= fact
            tmp_f.unit_y = new_unit
            tmp_f.axis_y *= fact
            if fact < 0:
                reversey = True
        else:
            try:
                tmp_f.axis_y *= scaley
                if scaley < 0:
                    reversey = True
            except TypeError:
                raise TypeError("'scaley' should be a number or an Unum "
                                f"object, but is currently {scaley}")
        if reversey:
            tmp_f.axis_y = tmp_f.axis_y[::-1]
        # returning
        if not inplace:
            return tmp_f

    def rotate(self, angle, inplace=False):
        """
        Rotate the field.

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
        # check params
        if angle % 90 != 0:
            raise ValueError("'angle' should be a multiple of 90,"
                             f" and is currently {angle}")
        if not isinstance(inplace, bool):
            raise TypeError("'inplace' should be True or False, and is"
                            f" currently {inplace}")
        # copy or not
        if inplace:
            tmp_field = self
        else:
            tmp_field = self.copy()
        # normalize angle
        angle = angle % 360
        # rotate
        if angle == 0:
            pass
        elif angle == 90:
            tmp_field.__axis_x, tmp_field.__axis_y \
                = tmp_field.axis_y[::-1], tmp_field.axis_x
            tmp_field.__unit_x, tmp_field.__unit_y \
                = tmp_field.unit_y, tmp_field.unit_x
        elif angle == 180:
            tmp_field.__axis_x, tmp_field.__axis_y \
                = tmp_field.axis_x[::-1], tmp_field.axis_y[::-1]
        elif angle == 270:
            tmp_field.__axis_x, tmp_field.__axis_y \
                = tmp_field.axis_y, tmp_field.axis_x[::-1]
            tmp_field.__unit_x, tmp_field.__unit_y \
                = tmp_field.unit_y, tmp_field.unit_x
        else:
            raise Exception()
        # correction in case of non-crescent axis
        if tmp_field.axis_x[-1] < tmp_field.axis_x[0]:
            tmp_field.__axis_x = -tmp_field.axis_x
        if tmp_field.axis_y[-1] < tmp_field.axis_y[0]:
            tmp_field.__axis_y = -tmp_field.axis_y
        # returning
        if not inplace:
            return tmp_field

    def change_unit(self, axis, new_unit):
        """
        Put a field axis in the wanted unit.
        Change the axis value in agreement with the new unit.

        Parameters
        ----------
        axis : string in ['x', 'y']
            Axis to change the unit for.
        new_unit : Unum.unit object or string
            New unit.

        Note
        ----
        To associate a completely different unit to an axis
        (e.g. 'm' to 's'), use 'Field.axis_x = "s"'.
        """
        if not isinstance(new_unit, unum.Unum):
            try:
                new_unit = make_unit(new_unit)
            except TypeError:
                raise TypeError("'new_unit' should be a valid unit.")
        if axis not in ['x', 'y']:
            raise TypeError("'axis' should be 'x' or 'y', and is "
                            f"currently {axis}")
        if axis == 'x':
            old_unit = self.unit_x
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.unit_x = new_unit/fact
            self.axis_x *= fact
        elif axis == 'y':
            old_unit = self.unit_y
            new_unit = old_unit.asUnit(new_unit)
            fact = new_unit.asNumber()
            self.unit_y = new_unit/fact
            self.axis_y *= fact

    def set_origin(self, x=None, y=None):
        """
        Modify the axis in order to place the origin at the given point (x, y).

        Parameters
        ----------
        x, y : numbers
            Position of the new origin.
        """
        if x is not None:
            try:
                self.axis_x -= x
            except TypeError:
                raise TypeError(f"'x' must be a number, and is currently {x}")
        if y is not None:
            try:
                self.axis_y -= y
            except TypeError:
                raise TypeError(f"'y' must be a number, and is currently {y}")

    def crop(self, intervx=None, intervy=None, full_output=False,
             ind=False, inplace=False):
        """
        Crop the field.

        Parameters
        ----------
        intervx : array, optional
            Wanted interval along x.
        intervy : array, optional
            Wanted interval along y.
        full_output : boolean, optional
            If 'True', cutting indices are also returned
        ind : boolean, optional
            If 'True', intervals are understood as indices along axis.
            If 'False' (default), intervals are understood in axis units.
        inplace : boolean, optional
            If 'True', the field is croped in place,
            else (default), a copy is returned.
        """
        # default values
        axis_x, axis_y = self.axis_x, self.axis_y
        if intervx is None:
            if ind:
                intervx = [0, len(axis_x)]
            else:
                intervx = [axis_x[0], axis_x[-1]]
        if intervy is None:
            if ind:
                intervy = [0, len(axis_y)]
            else:
                intervy = [axis_y[0], axis_y[-1]]
        # checking parameters
        try:
            intervx = np.array(intervx, dtype=float)
        except TypeError:
            raise TypeError("'intervx' must be an array of two numbers")
        if intervx.ndim != 1 or intervx.shape != (2,):
            raise ValueError("'intervx' must be an array of two numbers")
        if intervx[0] > intervx[1]:
            raise ValueError("'intervx' values must be crescent")
        try:
            intervy = np.array(intervy, dtype=float)
        except TypeError:
            raise TypeError("'intervy' must be an array of two numbers")
        if intervy.ndim != 1 or intervy.shape != (2,):
            raise ValueError("'intervy' must be an array of two numbers")
        if intervy[0] > intervy[1]:
            raise ValueError("'intervy' values must be crescent")
        # checking crooping windows
        if ind:
            if intervx[0] > len(axis_x) or intervx[1] <= 0 or \
                    intervy[0] > len(axis_y) or intervy[1] <= 0:
                raise ValueError("Invalid cropping window.")
        else:
            if (intervx[1] <= axis_x[0])\
                    or intervx[0] >= axis_x[-1]\
                    or intervy[1] <= axis_y[0]\
                    or intervy[0] >= axis_y[-1]:
                raise ValueError("Invalid cropping window.")
        # finding interval indices
        if ind:
            indmin_x = int(intervx[0])
            indmax_x = int(intervx[1])
            indmin_y = int(intervy[0])
            indmax_y = int(intervy[1])
        else:
            if intervx[0] <= axis_x[0]:
                indmin_x = 0
            else:
                indmin_x = self.get_indice_on_axis('x', intervx[0])[-1]
            if intervx[1] >= axis_x[-1]:
                indmax_x = len(axis_x) - 1
            else:
                indmax_x = self.get_indice_on_axis('x', intervx[1])[0]
            if intervy[0] <= axis_y[0]:
                indmin_y = 0
            else:
                indmin_y = self.get_indice_on_axis('y', intervy[0])[-1]
            if intervy[1] >= axis_y[-1]:
                indmax_y = len(axis_y) - 1
            else:
                indmax_y = self.get_indice_on_axis('y', intervy[1])[0]
        # cropping the field
        axis_x = self.axis_x[indmin_x:indmax_x + 1]
        axis_y = self.axis_y[indmin_y:indmax_y + 1]
        if inplace:
            self.__axis_x = axis_x
            self.__axis_y = axis_y
            self.__shape = [len(axis_x), len(axis_y)]
            if full_output:
                return indmin_x, indmax_x, indmin_y, indmax_y
        else:
            cropfield = self.copy()
            cropfield.__axis_x = axis_x
            cropfield.__axis_y = axis_y
            cropfield.__shape = [len(axis_x), len(axis_y)]
            if full_output:
                return indmin_x, indmax_x, indmin_y, indmax_y, cropfield
            else:
                return cropfield

    def extend(self, nmb_left=0, nmb_right=0, nmb_up=0, nmb_down=0,
               inplace=False):
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
        new_axis_x = self.axis_x.copy()
        new_axis_y = self.axis_y.copy()
        if nmb_left != 0:
            dx = self.axis_x[1] - self.axis_x[0]
            x0 = self.axis_x[0]
            new_xs = np.arange(x0 - dx*nmb_left, x0, dx)
            new_axis_x = np.concatenate((new_xs, new_axis_x))
        if nmb_right != 0:
            dx = self.axis_x[-1] - self.axis_x[-2]
            x0 = self.axis_x[-1]
            new_xs = np.arange(x0 + dx, x0 + dx*(nmb_right + 1), dx)
            new_axis_x = np.concatenate((new_axis_x, new_xs))
        if nmb_down != 0:
            dy = self.axis_y[1] - self.axis_y[0]
            y0 = self.axis_y[0]
            new_ys = np.arange(y0-dy*nmb_down, y0, dy)
            new_axis_y = np.concatenate((new_ys, new_axis_y))
        if nmb_up != 0:
            dy = self.axis_y[-1] - self.axis_y[-2]
            y0 = self.axis_y[-1]
            new_ys = np.arange(y0 + dy, y0 + dy*(nmb_up + 1), dy)
            new_axis_y = np.concatenate((new_axis_y, new_ys))
        if inplace:
            self.__axis_x = new_axis_x
            self.__axis_y = new_axis_y
        else:
            fi = self.copy()
            fi.__axis_x = new_axis_x
            fi.__axis_y = new_axis_y
            return fi

    def __clean(self):
        self.__init__()
