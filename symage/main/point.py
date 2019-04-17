# Regrouping objects with respect to points
# Author: Kaan Eraslan
# Licensing: see, License

# Packages

import numpy as np
from sympy.geometry.point import Point as SymPoint
from sympy.geometry.point import Point2D as SymPoint2D
from sympy.geometry.point import Point3D as SymPoint3D


class PointND:
    "N dimensional point object"

    def __init__(self, coords: ()):
        self.coords = coords
        self.point = SymPoint(coords)
        self.ndim = len(coords)

    def __str__(self):
        return "Point at {}".format(str(enumerate(self.coords)))

    def __repr__(self):
        return "{0}-d point at {1}.".format(self.ndim,
                                            str(enumerate(self.coords)))

    def __hash__(self):
        "Hash object"
        point_coordinates = self.coords
        return hash(point_coordinates)

    def __sub__(self, other):
        return self.point - other.point

    def __add__(self, other):
        return self.point + other.point

    def __eq__(self, other):
        return self.point == other.point

    @staticmethod
    def _checkPointDirection(point1, point2):
        "Check whether two points are in same direction"
        assert point1.ndim == point2.ndim
        return point1.point.unit == point2.point.unit

    def checkPointDirection(self, point):
        "Wrapper for class instance"
        return self._checkPointDirection(point1=self, point2=point)


class Point2D(PointND):
    "Regroups methods with respect to point in 2d"

    def __init__(self, x=0, y=0,
                 coordlist=None,
                 degree=0):
        if coordlist is not None:
            assert len(coordlist) == 2
            argx = coordlist[0]
            argy = coordlist[1]
        else:
            argx = x
            argy = y
        #
        super().__init__(coords=(argx, argy))
        self.point = SymPoint2D(argx, argy)
        self.angle_degree = degree
        self.radian = self.angle_degree * np.pi / 180
        self.old_x = 0
        self.old_y = 0
        self.new_y = 0
        self.new_x = 0
        self.x = argx
        self.y = argy

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


class GrayImagePoint(Point2D):
    """
    Extends euclidean space points to images

    Third dimension is the pixel value
    """

    def __init__(self,
                 x: int, y: int, z: int
                 ):
        assert isinstance(x, int) or x is None
        assert isinstance(y, int) or y is None
        assert isinstance(z, int) or z is None
        if x is not None:
            assert x >= 0
        if y is not None:
            assert y >= 0
        if z is not None:
            assert z <= 255 and z >= 0
        self.x = x
        self.y = y
        self.z = z
        self.point = SymPoint3D(x, y, z)

    def copy(self):
        "Duplicate the current instance of the class"
        point = GrayImagePoint(x=self.x, y=self.y, z=self.z)
        #
        return point

    def getPoint2D(self):
        "Get the euclidean space point representation of current point"
        point = Point2D(x=self.x, y=self.y)
        return point

    def getPointRowValFromImage(self, image: np.ndarray):
        "Get the row of the point from matrix"
        row = np.uint32(self.y)
        return image[row, :]

    def getPointColValFromImage(self, image: np.ndarray):
        "Get the col of the point from matrix"
        col = np.uint32(self.x)
        return image[:, col]

    def getPointValFromImage(self, image: np.ndarray):
        "Get value of the point from matrix"
        row, col = np.uint32(self.y), np.uint32(self.x)
        return image[row, col]

    def setZvalFromImage(self, image: np.ndarray):
        "Given an image set z value of the point from it"
        zval = self.getPointValFromImage(image)
        self.z = zval
