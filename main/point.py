# Regrouping objects with respect to points
# Author: Kaan Eraslan
# Licensing: see, License

# Packages

import numpy as np
from sympy.geometry import point


class PointND(point.Point):
    "N dimensional point object"

    def __init__(self, coords: ()):
        self.coords = coords
        self.ndim = len(coords)
        super().__init__(coords)

    def __str__(self):
        return "Point at {}".format(str(self.coords))

    def __repr__(self):
        return "{0}-d point at {1}.".format(self.ndim, str(self.coords))

    def __hash__(self):
        "Hash object"
        point_coordinates = self.coords
        return hash(point_coordinates)

    @staticmethod
    def _checkPointDirection(point1, point2):
        "Check whether two points are in same direction"
        assert point1.ndim == point2.ndim
        return point1.unit == point2.unit

    def checkPointDirection(self, point):
        "Wrapper for class instance"
        return self._checkPointDirection(point1=self, point2=point)


class Point2D(point.Point2D):
    "Regroups methods with respect to point in 2d"

    def __init__(self, x=0, y=0,
                 coordlist=None,
                 degree=0):
        if coordlist is not None:
            assert len(coordlist) == 2
            super().__init__(coords=coordlist)
            x = coordlist[0]
            y = coordlist[1]
        else:
            super().__init__(coords=(x, y))
        #
        self.angle_degree = degree
        self.radian = self.angle_degree * np.pi / 180
        self.old_x = 0
        self.old_y = 0
        self.new_y = 0
        self.new_x = 0
        self.x = x
        self.y = y

    def __call__(self):
        "Implements direct calls"
        return self.__str__()

    def carte2polar(self):
        "Transform cartesian coordinates to polar coordinates"
        x = self.x
        y = self.y
        distance = np.sqrt(x**2 + y**2)
        # formally sqrt((x - 0)^2 + (y-0)^2)
        angle = np.arctan2(y, x)

        return distance, angle


class ImagePoint2D(Point2D):
    "Extends euclidean space points to images"

    def __init__(self,
                 image: np.ndarray[[np.uint8]],
                 x=None, y=None,
                 costfn=lambda x: x,
                 emap=None,  #: np.ndarray[[np.uint8]],
                 spacePoint=None) -> None:
        self.x = x
        self.y = y
        if spacePoint is not None:
            self.x = spacePoint.x
            self.y = spacePoint.y
        if spacePoint is None and ( x is None and y is None ):
            raise ValueError(
                "Constructor needs either x and y or a space point"
                " they can not be both none."
            )
        super().__init__(x=self.x, y=self.y)
        self.costfn = costfn
        self.emap = emap
        self.image = image
        self.pixel_value = None
        self.pixel_energy = None

    def copy(self):
        "Duplicate the current instance of the class"
        point = ImagePoint2D(image=self.image,
                             x=self.x,
                             y=self.y)
        point.pixel_value = self.pixel_value
        point.pixel_energy = self.pixel_energy
        #
        return point

    def getPoint2D(self):
        "Get the euclidean space point representation of current point"
        point = Point2D(x=self.x, y=self.y)
        return point

    def getPointRowVal(self):
        "Get the row of the point from matrix"
        row = np.uint32(self.y)
        return self.image[row, :]

    def getPointColVal(self):
        "Get the col of the point from matrix"
        col = np.uint32(self.x)
        return self.image[:, col]

    def getPointVal(self):
        "Get value of the point from matrix"
        row, col = np.uint32(self.y), np.uint32(self.x)
        return self.image[row, col]

    @property
    def pixelValue(self):
        "Set pixel value of the point"
        row = np.uint32(self.y)
        col = np.uint32(self.x)
        if self.image.ndim == 2:
            return self.image[row, col]
        else:
            return self.image[row, col, :]

    @property
    def pixelEnergy(self):
        "Set pixel energy of the point"
        dim_nb = self.image.ndim
        row = np.uint32(self.y)
        col = np.uint32(self.x)
        if dim_nb == 2:
            if self.emap is not None:
                return self.emap[row, col]
            else:
                return self.pixelValue
        else:
            if self.emap is not None:
                return self.emap[row, col]
            else:
                return self.pixelValue.sum()

    def setPointProperties(self):
        "Wrapper for setting values to point properties"
        self.pixel_value = self.pixelValue
        self.pixel_energy = self.pixelEnergy
