# Regroups objects with respect to lines
# Author: Kaan Eraslan
# Licensing: see, LICENSE
# No warranties, see LICENSE

# Packages

import numpy as np
from sympy.geometry import line

from symage.main.point import GrayImagePoint, PointND
from symage.main.point import Point2D as Point2D
from symage.main.utils import assertCond


class LocatedVector:
    "Located vector n dimensional"

    def __init__(self, spoint: PointND, epoint: PointND) -> None:
        self.spoint = spoint  # start point
        self.epoint = epoint  # end point
        self.ndim = len(spoint.coords)
        self.segment = line.Segment(spoint.point, epoint.point)

    @staticmethod
    def isVectorEqual2Vector(vec1, vec2) -> bool:
        "Check whether 2 vectors are equal"
        # formula: for 2 vecs AB, and DC
        # they are equal if B-A = D-C
        # S.Lang 1986, p.10
        spoint1 = vec1.spoint
        epoint1 = vec1.epoint
        spoint2 = vec2.spoint
        epoint2 = vec2.epoint

        diff1 = spoint1 - epoint1
        diff2 = spoint2 - epoint2

        return diff1 == diff2

    def __eq__(self, vec):
        "Implement == operator for instances"
        return self.isVectorEqual2Vector(vec1=self, vec2=vec)

    @staticmethod
    def hasVectorSameDirection2Vector(vec1, vec2) -> bool:
        "Check whether 2 vectors have same direction"
        return vec1.segment.direction == vec2.segment.direction

    def hasSameDirection2Vector(self, vec):
        "Wrapper method for class instances"
        return self.hasVectorSameDirection2Vector(vec1=self, vec2=vec)

    def getNorm(self):
        "Wrapper for class instance"
        return self.segment.length


class LocatedVector2D(LocatedVector):
    "Located Vector in Euclidean Space"

    def __init__(
            self,
            initial_point=None,  # :Point2D:
            final_point=None,  # :Point2D:
            segment=None):
        if segment is None and (initial_point is None and final_point is None):
            raise TypeError(
                'please provide either initial and final points or'
                ' a segment to initiate the vector')
        elif (segment is None and initial_point is None
              and final_point is not None):
            raise TypeError(
                'please provide an initial point if a final point is provided'
                ' or provide a segment to initiate the vector')
        elif (segment is None and initial_point is not None
              and final_point is None):
            raise TypeError(
                'please provide an final point if an initial point is provided'
                ' or provide a segment to initiate the vector')
        elif (segment is not None and initial_point is not None
              and final_point is not None):
            raise TypeError(
                'please provide either initial and final points or'
                ' a segment to initiate the vector not both')
        elif (segment is not None and initial_point is None
              and final_point is None):
            spoint = Point2D(segment.points[0][0], segment.points[0][1])
            epoint = Point2D(segment.points[1][0], segment.points[1][1])
        elif (segment is None and initial_point is not None
              and final_point is not None):
            spoint = initial_point
            epoint = final_point

        self.pointList = []
        super().__init__(spoint, epoint)

    def __str__(self):
        return "LocatedVector2D from {0} to {1}".format(
            str(self.spoint), str(self.epoint))

    def __call__(self):
        return self.__str__()

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def _getPoint2VecDistancePoint(aVec, point, isMin: bool) -> [Point2D, int]:
        "Get closest/farthest point on vec with distance to the given point"
        points = aVec.pointList
        npoint = Point2D(coordlist=[0, 0])  # point at origin
        if isMin is True:
            dist = float('inf')
        else:
            dist = float('-inf')

        for p in points:
            # assert p.ndim == point.ndim
            vec = LocatedVector(p, point)
            vecnorm = vec.getNorm()

            if isMin is True:
                checkval = vecnorm <= dist
            else:
                checkval = vecnorm > dist

            if checkval:
                dist = vecnorm
                npoint = p
        return npoint, dist

    @classmethod
    def _getNearestPointAndDistance(cls, vec, point):
        "Get the closest point on the line with distance or not to given point"
        return cls._getPoint2VecDistancePoint(vec, point, isMin=True)

    @classmethod
    def _getFarthestPointAndDistance(cls, vec, point):
        "Get farthest point and distance on the vector with given point"
        return cls._getPoint2VecDistancePoint(vec, point, isMin=False)

    @classmethod
    def _getMinDistance2Point(
            cls,
            vec,  #: LocatedVector2D,
            point: Point2D) -> float:
        "Get distance to line"
        distancePoint = cls.getNearestPointAndDistance(vec, point)

        return distancePoint[1]

    @classmethod
    def _getMaxDistance2Point(cls, vec, point) -> float:
        "Get farthest point and distance on the vec with given point"
        distancePoint = cls.getFarthestPointAndDistance(vec, point)
        return distancePoint[1]

    @classmethod
    def _getNearestPointOnVec(cls, vec, point) -> Point2D:
        "Get closest point on vec to the given point"
        distancePoint = cls._getNearestPointAndDistance(vec, point)
        return distancePoint[0]

    @classmethod
    def _getFarthestPointOnVec(cls, vec, point) -> Point2D:
        "Get farthest point and distance on the vec with given point"
        distancePoint = cls._getFarthestPointAndDistance(vec, point)
        return distancePoint[0]

    @classmethod
    def _getManhattanDistance(cls, point1_: Point2D, point2_: Point2D) -> int:
        "Get manhattan distance between two points"
        return point1_.taxicab_distance(point2_)

    @staticmethod
    def _getStraightLine(point1: Point2D, point2: Point2D) -> list:
        """
        Get line from points including the points included in the line
        Bresenham's line algorithm adapted from pseudocode in wikipedia:
        https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

        image should be grayscale

        """
        #  define local variables for readability
        P1X = point1.x
        P1Y = point1.y
        P2X = point2.x
        P2Y = point2.y

        #  difference and absolute difference between points
        #  used to calculate slope and relative location between points
        diffX = P2X - P1X
        diffXa = np.absolute(diffX, dtype="int32")
        diffY = P2Y - P1Y
        diffYa = np.absolute(diffY, dtype="int32")
        #
        steepx = 1
        if P1X < P2X:
            steepx = 1
        else:
            steepx = -1
        #
        if P1Y < P2Y:
            steepy = 1
        else:
            steepy = -1
        #
        div_term = diffXa
        #
        if diffXa > diffYa:
            div_term = diffXa
        else:
            div_term = -diffYa
            #
        error = div_term / 2
        #
        error2 = 0
        #
        arrival_condition = bool((P1X, P1Y) == (P2X, P2Y))
        #
        line_points = []
        initial_p = Point2D(coordlist=[P1X, P1Y])
        line_points.append(initial_p)
        #
        while arrival_condition is False:
            error2 = error
            if error2 > -diffXa:
                error = error - diffYa
                P1X = P1X + steepx
                #
            if error2 < diffYa:
                error = error + diffXa
                P1Y = P1Y + steepy
                #
                # Check
            point = Point2D(coordlist=[P1X, P1Y])
            line_points.append(point)
            arrival_condition = bool((P1X, P1Y) == (P2X, P2Y))
        #
        return line_points

    @staticmethod
    def _getStraightDistance(p1: Point2D, p2: Point2D) -> np.float:
        "Get straight distance between two points"
        vec = LocatedVector(p1, p2)
        vecnorm = vec.getNorm()
        #
        return vecnorm

    @staticmethod
    def __getVec2VecDistancePointVec(vec1, vec2,
                                     isMin: bool,
                                     isMinFuncs: bool):
        """
        Get least/farthest distance vec point

        Given two vectors find the closest or farthest
        distance in between.

        There are several outputs that are possible:

        Find the closest distance between vectors using
        the closest point in vec1 to the vec2.

        Find the closest distance using farthest point
        in vec1 to the vec2

        Find the farthest distance using closest point
        in vec1 to the vec2

        Find the farthest distance using farthest point
        in vec1 to the vec2
        """
        spoints = vec1.pointList  # starting points
        nspoint = Point2D(coordlist=[0, 0])
        nepoint = Point2D(coordlist=[0, 1])
        svec = None

        if isMin is True:
            distance = float('inf')
        else:
            distance = float('-inf')

        for sp in spoints:
            #
            distancePoint = vec2._getPoint2VecDistancePoint(vec2,
                point=sp, isMin=isMinFuncs)
            epoint = distancePoint[0]
            dist = distancePoint[1]

            if isMin is True:
                checkval = dist <= distance
            else:
                checkval = dist > distance

            if checkval:
                distance = dist
                nspoint = sp
                nepoint = epoint
                svec = LocatedVector2D(nspoint, nepoint)
        #
        svec.setVecProperties()
        return nspoint, nepoint, svec, distance

    @classmethod
    def _getVec2VecDistancePointVec(cls,
                                    vec1,
                                    vec2,
                                    isMin: bool,
                                    isMinFuncs: bool):
        """
        Wrapper for staticmethod in order to override it
        easily in the subclass
        """
        return cls.__getVec2VecDistancePointVec(vec1, vec2, isMin, isMinFuncs)

    @classmethod
    def _getVec2VecDistancePointVecWithMinDistance(
            cls,
            vec1,
            vec2,
            isMin: bool):
        """
        Vec to vec distance

        Find the closest/farthest distance between
        vectors using
        the closest point in vec1 to the vec2.

        Return
        -------
        vec, startpoint, endpoint, distance
        """
        return cls._getVec2VecDistancePointVec(
            vec1,
            vec2,
            isMin,
            isMinFuncs=True)

    @classmethod
    def _getVec2VecDistancePointVecWithMaxDistance(
            cls, vec1, vec2, isMin: bool):
        """
        Vec to vec distance

        Find the closest/farthest distance between
        vectors using
        the farthest point in vec1 to the vec2.

        Return
        -------
        vec, startpoint, endpoint, distance
        """
        return cls._getVec2VecDistancePointVec(
            vec1,
            vec2,
            isMin,
            isMinFuncs=False)

    @classmethod
    def _getVec2VecMinDistPointVecMinD(cls, vec1, vec2):
        """
        Vec to vec min distance

        Find the closest distance between
        vectors using
        the closest point in vec1 to the vec2.

        Return
        -------
        vec, startpoint, endpoint, distance

        """
        return cls._getVec2VecDistancePointVecWithMinDistance(vec1, vec2,
                                                              isMin=True)

    @classmethod
    def _getVec2VecMaxDistPointVecMinD(cls, vec1, vec2):
        """
        Vec to vec max distance

        Find the closest distance between
        vectors using
        the closest point in vec1 to the vec2.

        Return
        -------
        vec, startpoint, endpoint, distance

        """
        return cls._getVec2VecDistancePointVecWithMinDistance(vec1, vec2,
                                                              isMin=False)

    @classmethod
    def _getVec2VecMinDistPointVecMaxD(cls, vec1, vec2):
        """
        Vec to vec min distance

        Find the closest distance between
        vectors using
        the closest point in vec1 to the vec2.

        Return
        -------
        vec, startpoint, endpoint, distance

        """
        return cls._getVec2VecDistancePointVecWithMaxDistance(vec1, vec2,
                                                              isMin=True)

    @classmethod
    def _getVec2VecMaxDistPointVecMaxD(cls,
                                       vec1,
                                       vec2):
        """
        Vec to vec min distance

        Find the closest distance between
        vectors using
        the closest point in vec1 to the vec2.

        Return
        -------
        vec, startpoint, endpoint, distance

        """
        return cls._getVec2VecDistancePointVecWithMaxDistance(vec1, vec2,
                                                              isMin=False)

    def setLine(self):
        "Set line from starting point to end point"
        self.pointList = self._getStraightLine(
            point1=self.spoint, point2=self.epoint)
        return None

    def setVecProperties(self):
        "Wrapper for setters"
        self.setLine()

        return None


class GrayImageLocatedVector2D(LocatedVector2D):
    "Extends the line in euclidean space to images"

    def __init__(
            self,
            image: np.ndarray,
            vec=None,
            initial_point=None,  # :GrayImagePoint:
            final_point=None,  # :GrayImagePoint:
            segment=None) -> None:
        ""
        if vec is None:
            super().__init__(initial_point, final_point, segment)
        else:
            spoint = vec.spoint
            epoint = vec.epoint
            super().__init__(spoint, epoint)
        self.image = image.copy()
        self.charge = 0

    @staticmethod
    def getConditionDistanceCharge(minDist,
                                   minCharge,
                                   distanceParentScope,
                                   distanceLocalScope,
                                   chargeParentScope,
                                   chargeLocalScope):
        """
        Set the evaluation condition based on the given parameters

        Each branch correspond to different use cases of the final
        condition of the algorithm.

        Here are the use cases that are covered from top to bottom:
        - Use only minimum distance as criteria when computing a point or vec,
          that is do not take into account the energy of the point

        - Use only maximum distance as criteria when computing a point or vec,
          that is do not take into account the energy of the point

        - Use only minimum charge as criteria when computing a point or vec,
          that is do not take into account the distance of the point

        - Use only minimum charge as criteria when computing a point or vec,
          that is do not take into account the distance of the point

        - Use only maximum charge as criteria when computing a point or vec,
          that is do not take into account the distance of the point

        - Use both: minimum distance and minimum charge
        - Use both: minimum distance and maximum charge
        - Use both: maximum distance and minimum charge
        - Use both: maximum distance and maximum charge
        """
        if minDist is True and minCharge is None:
            condition = distanceLocalScope <= distanceParentScope
        elif minDist is False and minCharge is None:
            condition = distanceLocalScope > distanceParentScope
        elif minCharge is True and minDist is None:
            condition = chargeLocalScope <= chargeParentScope
        elif minCharge is False and minDist is None:
            condition = chargeLocalScope > chargeParentScope
        elif minDist is True and minCharge is True:
            condition = bool(distanceLocalScope <= distanceParentScope
                             and chargeLocalScope <= chargeParentScope)
        elif minDist is True and minCharge is False:
            condition = bool(distanceLocalScope <= distanceParentScope
                             and chargeLocalScope > chargeParentScope)
        elif minDist is False and minCharge is True:
            condition = bool(distanceLocalScope > distanceParentScope
                             and chargeLocalScope <= chargeParentScope)
        elif minDist is False and minCharge is False:
            condition = bool(distanceLocalScope > distanceParentScope
                             and chargeLocalScope > chargeParentScope)
        return condition

    @staticmethod
    def getDistanceChargeVariables(minDist: bool, minCharge: bool):
        "Get distance charge variables based on given conditions"
        if minDist is True and minCharge is None:
            distance = float('inf')
            charge = None
        elif minDist is False and minCharge is None:
            distance = float('-inf')
            charge = None
        elif minCharge is True and minDist is None:
            distance = None
            charge = float('inf')
        elif minCharge is False and minDist is None:
            distance = None
            charge = float('-inf')
        elif minDist is True and minCharge is True:
            distance = float('inf')
            charge = float('inf')
        elif minDist is True and minCharge is False:
            distance = float('inf')
            charge = float('-inf')
        elif minDist is False and minCharge is True:
            distance = float('-inf')
            charge = float('inf')
        elif minDist is False and minCharge is False:
            distance = float('-inf')
            charge = float('-inf')
        return distance, charge

    @classmethod
    def _getVec2VecDistancePointChargeVec(
            cls,
            vec1,
            vec2,
            isMinDistance: bool,  # boolean or none
            isMinFuncs: bool,
            isMinCharge: bool  # boolean or none
    ) -> [Point2D, Point2D, LocatedVector2D, float, float]:
        "Overrides the base class static method to include charge"
        #
        spoints = vec1.pointList  # starting points
        nspoint = None
        nepoint = None
        svec = None
        distance, charge = cls.getDistanceChargeVariables(
            minDist=isMinDistance, minCharge=isMinCharge
        )

        for sp in spoints:
            #
            distancePoint = vec2._getPoint2VecDistancePoint(point=sp,
                                                            isMin=isMinFuncs)
            epoint = distancePoint[0]
            dist = distancePoint[1]
            tempvec = GrayImageLocatedVector2D(image=vec2.image,
                                               initial_point=sp,
                                               final_point=epoint)
            tempvec.setVecProperties()
            tempcharge = tempvec.charge

            checkval = cls.getConditionDstanceCharge(
                minDist=isMinDistance,
                minCharge=isMinCharge,
                distanceParentScope=distance,
                distanceLocalScope=dist,
                chargeParentScope=charge,
                chargeLocalScope=tempcharge)
            if checkval:
                distance = dist
                nspoint = sp
                nepoint = epoint
                svec = tempvec
                charge = tempcharge
        #
        return nspoint, nepoint, svec, distance, charge

    @classmethod
    def _getVec2VecDistancePointVec(cls, vec1, vec2,
                                    isMin: bool,
                                    isMinFuncs) -> [Point2D, Point2D,
                                                    LocatedVector2D, float]:
        "Overrides the base class method"
        return cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=isMin, isMinFuncs=isMinFuncs,
            isMinCharge=None)[:4]

    @classmethod
    def getVec2VecMinFuncDistancePointMinChargeVec(cls, vec1, vec2):
        "Get min charge between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2,
            isMinDistance=None,
            isMinFuncs=True,
            isMinCharge=True)
        return distancePointsVec

    @classmethod
    def getVec2VecMaxFuncDistancePointMinChargeVec(cls, vec1, vec2):
        "Get min charge between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2,
            isMinDistance=None,
            isMinFuncs=False,
            isMinCharge=True)
        return distancePointsVec

    @classmethod
    def getVec2VecMinFuncDistancePointMaxChargeVec(cls, vec1, vec2):
        "Get min charge between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2,
            isMinDistance=None,
            isMinFuncs=True,
            isMinCharge=False)
        return distancePointsVec

    @classmethod
    def getVec2VecMaxFuncDistancePointMaxChargeVec(cls, vec1, vec2):
        "Get min charge between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2,
            isMinDistance=None,
            isMinFuncs=False,
            isMinCharge=False)
        return distancePointsVec

    @classmethod
    def getVec2VecMinFuncMinDistancePointMinChargeVec(cls, vec1, vec2):
        "Get min charge between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2,
            isMinDistance=True,
            isMinFuncs=True,
            isMinCharge=True)
        return distancePointsVec

    @classmethod
    def getVec2VecMinFuncMaxDistancePointMinChargeVec(cls, vec1, vec2):
        "Get min charge between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2,
            isMinDistance=False,
            isMinFuncs=True,
            isMinCharge=True)
        return distancePointsVec

    @classmethod
    def getVec2VecMaxFuncMinDistancePointMaxChargeVec(cls, vec1, vec2):
        "Get min charge between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2,
            isMinDistance=True,
            isMinFuncs=False,
            isMinCharge=False)
        return distancePointsVec

    @classmethod
    def getVec2VecMaxFuncMaxDistancePointMaxChargeVec(cls, vec1, vec2):
        "Get min charge between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2,
            isMinDistance=False,
            isMinFuncs=False,
            isMinCharge=False)
        return distancePointsVec

    @classmethod
    def getVec2VecMinFuncMaxDistancePointMaxChargeVec(cls, vec1, vec2):
        "Get min charge between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2,
            isMinDistance=False,
            isMinFuncs=True,
            isMinCharge=False)
        return distancePointsVec

    @classmethod
    def getVec2VecMinFuncMinDistancePointMaxChargeVec(cls, vec1, vec2):
        "Get min charge between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2,
            isMinDistance=True,
            isMinFuncs=True,
            isMinCharge=False)
        return distancePointsVec

    @classmethod
    def getVec2VecMinFuncMaxDistancePointMaxChargeVec(cls, vec1, vec2):
        "Get min charge between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2,
            isMinDistance=False,
            isMinFuncs=True,
            isMinCharge=False)
        return distancePointsVec

    @classmethod
    def getVec2VecMaxFuncMinDistancePointMinChargeVec(cls, vec1, vec2):
        "Get min charge between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2,
            isMinDistance=True,
            isMinFuncs=False,
            isMinCharge=True)
        return distancePointsVec

    @classmethod
    def getVec2VecMaxFuncMaxDistancePointMinChargeVec(cls, vec1, vec2):
        "Get min charge between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2,
            isMinDistance=False,
            isMinFuncs=False,
            isMinCharge=True)
        return distancePointsVec

    def setLine(self) -> None:
        "Overrides base class method"
        plist = self._getStraightLine(point1=self.spoint, point2=self.epoint)
        plist = [
            GrayImagePoint(coordlist=p.coords).setZvalFromImage(self.image)
            for p in plist
        ]
        self.pointList = plist
        return None

    def setVecCharge(self) -> None:
        "Set vector charge"
        counter = 0
        for p in self.pointList:
            counter += p.z
        self.charge = counter
        return None

    def setVecProperties(self) -> None:
        "Overrides base class method"
        self.setLine()
        self.setVecCharge()
        return None
