# Regrouping objects with respect to points
# Author: Kaan Eraslan
# Licensing: see, License

# Packages

import numpy as np
from sympy.geometry import point
from utils import jit_points as jpoints


class Point(point.Point):
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

        ndimp = point1.ndim
        p1coords = point1.coords
        p2coords = point2.coords

        checkval = False
        divisions = []
        divisionSum = 0

        for i in range(ndimp):
            division = p1coords[i] / p2coords[i]
            divisionSum += division

        #
        divisionAverage = divisionSum / ndimp

        #
        divbools = [divisionAverage == div for div in divisions]
        checkval = False
        if all(divbools) and divisionAverage > 0:
            checkval = True

        return checkval

    def checkPointDirection(self, point):
        "Wrapper for class instance"
        return self._checkPointDirection(point1=self, point2=point)


class Point2D(point.Point2D):
    "Regroups methods with respect to point in 2d"

    def __init__(self, x=0, y=0, coordlist=None):
        if coordlist is not None:
            assert len(coordlist) == 2
            super().__init__(coords=coordlist)
            x = coordlist[0]
            y = coordlist[1]
        else:
            super().__init__(coords=(x, y))
        #
        self.angle_degree = 0
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

    def setXY(self, points: ()):
        "Set x and y respectively to current instance"
        self.x = points[0]
        self.y = points[1]
        #
        return None

    def get_new_xy(self):
        "get new x and y values with old x and y values"
        self.new_x, self.new_y = jpoints.get_new_xy(instance=self)
        #
        return (self.new_x, self.new_y)

    def get_new_x(self):
        "Get new x value based on the angle"
        prod3 = self.old_y * np.sin(self.radian)
        prod4 = self.old_x * np.cos(self.radian)
        self.new_x = int(prod3 + prod4)
        #
        return self.new_x

    def get_new_y(self):
        "get new y value based on the angle"
        prod1 = self.old_y * np.cos(self.radian)
        prod2 = self.old_x * np.sin(self.radian)
        self.new_y = int(prod1 - prod2)

        return self.new_y

    def get_old_x(self):
        "Get old x value based on the angle"
        prod1 = self.old_y * np.sin(self.radian)
        diff = self.new_x - prod1
        self.old_x = diff // np.cos(self.radian)

        return self.old_x

    def get_old_y(self):
        "get old y value based on the angle"
        prod = self.old_x * np.sin(self.radian)
        summ = self.new_y + prod
        self.old_y = summ // np.cos(self.radian)

        return self.old_y

    def get_old_x2(self):
        "get old x value based on the angle and old and new y values"
        prod = self.old_y * np.cos(self.radian)
        diff = prod - self.new_y
        self.old_x = diff // np.sin(self.radian)

        return self.old_x

    def get_old_y2(self):
        "get old y value based on the angle and old and new x values"
        prod = self.old_x * np.cos(self.radian)
        diff = self.new_x - prod
        self.old_y = diff // np.sin(self.radian)

        return self.old_y

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
                 x,y,
                 costfn=lambda x: x,
                 energyfn=None,
                 spacePoint=None) -> None:
        self.x = x
        self.y = y
        if spacePoint is not None:
            self.x = spacePoint.x
            self.y = spacePoint.y
        super().__init__(x=self.x, y=self.y)
        self.costfn = costfn
        self.energyfn = energyfn
        self.image = image
        self.pixel_value = None
        self.pixel_energy = None
        # self.parent = ImagePoint2D()
        # self.child = ImagePoint2D()
        #

    def __lt__(self, point):
        "Implement < operator for instances"
        return self.costfn(self) < self.costfn(point)

    def __gt__(self, point):
        "Implement > operator for instances"
        return self.costfn(self) > self.costfn(point)

    def __repr__(self):
        "Implement a string representation of the object"
        obj = """
        Point in Euclidean Space at x: {0}, y: {1},
        pixel energy: {2},
        pixel value: {3}
        """.format(self.x, self.y, str(self.pixel_energy),
                   str(self.pixel_value))
        return obj

    def copy(self):
        "Duplicate the current instance of the class"
        point = ImagePoint2D(image=self.image)
        point.pixel_value = self.pixel_value
        point.pixel_energy = self.pixel_energy
        point.x = self.x
        point.y = self.y
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

    def setPixelValueEnergy(self):
        "Set pixel value of the point"
        row = np.uint32(self.y)
        col = np.uint32(self.x)
        if self.image.ndim == 2:
            self.pixel_value = self.image[row, col]
            self.pixel_energy = self.pixel_value
        elif self.image.ndim == 3:
            self.pixel_value = self.image[row, col, :]
            self.pixel_energy = self.image[row, col, :].sum()
        #
        return self.pixel_value, self.pixel_energy

    def setPointProperties(self):
        "Wrapper for setting values to point properties"
        self.setPixelValueEnergy()
