# Regroups objects with respect to lines
# Author: Kaan Eraslan
# Licensing: see, LICENSE

# Packages

import numpy as np
from sympy.geometry import line

from point import ImagePoint2D, Point, Point2D

class LocatedVector(line.Segment):
    "Located vector n dimensional"

    def __init__(self, spoint: Point, epoint: Point) -> None:
        super().__init__(spoint, epoint)
        self.spoint = spoint  # start point
        self.epoint = epoint  # end point
        self.ndim = len(spoint.coords)

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

        checkval = False

        if diff1 == diff2:
            checkval = True

        return checkval

    def __eq__(self, vec):
        "Implement == operator for instances"
        return self.isVectorEqual2Vector(vec1=self, vec2=vec)

    @staticmethod
    def hasVectorSameDirection2Vector(vec1, vec2) -> bool:
        "Check whether 2 vectors have same direction"
        return vec1.direction == vec2.direction

    def hasSameDirection2Vector(self, vec):
        "Wrapper method for class instances"
        return self.hasVectorSameDirection2Vector(vec1=self, vec2=vec)

    def getNorm(self):
        "Wrapper for class instance"
        return self.length


class LocatedVector2D(LocatedVector):
    "Located Vector in Euclidean Space"

    def __init__(
            self,
            initial_point=None,  # :Point2D:
            final_point=None,  # :Point2D:
            segment=None):
        if segment is None and initial_point is None and final_point is None:
            raise ValueError(
                'please provide either initial and final points or'
                ' a segment to initiate the vector')
        elif (segment is None and initial_point is None
              and final_point is not None):
            raise ValueError(
                'please provide an initial point if a final point is provided'
                ' or provide a segment to initiate the vector')
        elif (segment is None and initial_point is not None
              and final_point is None):
            raise ValueError(
                'please provide an final point if an initial point is provided'
                ' or provide a segment to initiate the vector')
        elif (segment is not None and initial_point is not None
              and final_point is not None):
            raise ValueError(
                'please provide either initial and final points or'
                ' a segment to initiate the vector not both')
        elif (segment is not None and initial_point is None
              and final_point is None):
            spoint = segment.points[0]
            epoint = segment.points[1]
            super().__init__(spoint=spoint, epoint=epoint)
        elif (segment is None and initial_point is not None
              and final_point is not None):
            super().__init__(spoint=initial_point, epoint=final_point)

        self.pointList = []

    @staticmethod
    def _getPoint2VecDistancePoint(aVec, point, isMin: bool):
        "Get closest/farthest point on vec with distance to the given point"
        points = aVec.pointList
        npoint = Point2D([0, 0])  # point at origin
        if isMin is True:
            retval = float('inf')
        else:
            retval = float('-inf')

        for p in points:
            # assert p.ndim == point.ndim
            vec = LocatedVector(p, point)
            vecnorm = vec.getNorm()
            checkval = bool

            if isMin is True:
                checkval = vecnorm < retval
            else:
                checkval = vecnorm > retval

            if checkval:
                retval = vecnorm
                npoint = p
        return npoint, retval

    @classmethod
    def _getNearestPointAndDistance(cls, vec, point):
        "Get the closest point on the line with distance or not to given point"
        return cls._getPoint2VecDistancePoint(vec, point, True)

    @classmethod
    def _getMinDistance2Point(
            cls,
            vec,  #: LocatedVector2D,
            point: Point2D) -> float:
        "Get distance to line"
        distancePoint = cls.getNearestPointAndDistance(vec, point)

        return distancePoint[1]

    @classmethod
    def _getNearestPointOnVec(cls, vec, point) -> Point2D:
        "Get closest point on vec to the given point"
        distancePoint = cls.getNearestPointAndDistance(vec, point)
        return distancePoint[0]

    @classmethod
    def _getFarthestPointAndDistance(cls, vec, point):
        "Get farthest point and distance on the vector with given point"
        return cls._getPoint2VecDistancePoint(vec, point, False)

    @classmethod
    def _getMaxDistance2Point(cls, vec, point) -> float:
        "Get farthest point and distance on the vec with given point"
        distancePoint = cls.getFarthestPointAndDistance(vec, point)
        return distancePoint[1]

    @classmethod
    def _getFarthestPointOnVec(cls, vec, point) -> Point2D:
        "Get farthest point and distance on the vec with given point"
        distancePoint = cls.getFarthestPointAndDistance(vec, point)
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
        initial_p = Point2D([P1X, P1Y])
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
            point = Point2D([P1X, P1Y])
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
    def _getVec2VecDistancePointVec(vec1, vec2, isMin):
        "Get least/farthest distance"
        spoints = vec1.pointList  # starting points
        nspoint = Point2D([0, 0])
        nepoint = Point2D([0, 1])
        svec = None

        if isMin is True:
            distance = float('inf')
        else:
            distance = float('-inf')

        for sp in spoints:
            #
            distancePoint = vec2.getPoint2VecDistancePoint(
                point=sp, isMin=isMin)
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
    def getVec2VecDistancePointVec(cls, vec1, vec2, isMin):
        "Wrapper for staticmethod"
        return cls._getVec2VecDistancePointVec(vec1, vec2, isMin)

    @classmethod
    def getVec2VecMinDistance(cls, vec1, vec2):
        "Get min distance between two located vectors"
        distancePointsVec = cls.getVec2VecDistancePointVec(
            vec1, vec2, isMin=True)
        return distancePointsVec[3]

    @classmethod
    def getVec2VecMaxDistance(cls, vec1, vec2):
        "Get max distance between two located vectors"
        distancePointsVec = cls.getVec2VecDistancePointVec(
            vec1, vec2, isMin=False)
        return distancePointsVec[3]

    @classmethod
    def getVec2VecMinVec(cls, vec1, vec2):
        "Get min distance between two located vectors"
        distancePointsVec = cls.getVec2VecDistancePointVec(
            vec1, vec2, isMin=True)
        return distancePointsVec[2]

    @classmethod
    def getVec2VecMaxVec(cls, vec1, vec2):
        "Get max distance between two located vectors"
        distancePointsVec = cls.getVec2VecDistancePointVec(
            vec1, vec2, isMin=False)
        return distancePointsVec[2]

    @classmethod
    def getVec2VecMinSpoint(cls, vec1, vec2):
        "Get min distance between two located vectors"
        distancePointsVec = cls.getVec2VecDistancePointVec(
            vec1, vec2, isMin=True)
        return distancePointsVec[0]

    @classmethod
    def getVec2VecMaxSpoint(cls, vec1, vec2):
        "Get max distance between two located vectors"
        distancePointsVec = cls.getVec2VecDistancePointVec(
            vec1, vec2, isMin=False)
        return distancePointsVec[0]

    @classmethod
    def getVec2VecMinEpoint(cls, vec1, vec2):
        "Get min distance between two located vectors"
        distancePointsVec = cls.getVec2VecDistancePointVec(
            vec1, vec2, isMin=True)
        return distancePointsVec[1]

    @classmethod
    def getVec2VecMaxEpoint(cls, vec1, vec2):
        "Get max distance between two located vectors"
        distancePointsVec = cls.getVec2VecDistancePointVec(
            vec1, vec2, isMin=False)
        return distancePointsVec[1]

    def setLine(self):
        "Set line from starting point to end point"
        self.pointList = self._getStraightLine(
            point1=self.p_init, point2=self.p_final)
        return None

    def setVecProperties(self):
        "Wrapper for setters"
        self.setLine()

        return None

    def getMinDistance2Point(self, point) -> float:
        "Get min distance of the given point to line"
        assert point.ndim == self.ndim
        return self._getMinDistance2Point(vec=self, point=point)

    def getNearestPoint2Point(self, point) -> Point2D:
        "Get nearest point on the vector to the given point"
        assert point.ndim == self.ndim
        return self._getNearestPointOnVec(vec=self, point=point)

    def getMaxDistancePoint(self, point):
        "Get max distance of the given point to line"
        assert point.ndim == self.ndim
        return self._getMaxDistance2Point(vec=self, point=point)

    def getFarthestPoint2Point(self, point) -> Point2D:
        "Get farthest point on the vector to the given point"
        assert point.ndim == self.ndim
        return self._getFarthestPointOnVec(vec=self, point=point)

    def getMinMaxDistance2Vec(self, isMin, vec):
        "Get minimum or maximum distance to given vector from self"
        assert self.ndim == vec.ndim
        distance = None
        if isMin is True:
            distance = self.getVec2VecMinDistance(vec1=self, vec2=vec)
        else:
            distance = self.getVec2VecMaxDistance(vec1=self, vec2=vec)
        return distance

    def getMinMaxVec2Vec(self, isMin, vec):
        "Get minimum or maximum distance vector to given vector from self"
        assert self.ndim == vec.ndim
        if isMin is True:
            nvec = self.getVec2VecMinVec(vec1=self, vec2=vec)
        else:
            nvec = self.getVec2VecMaxVec(vec1=self, vec2=vec)
        return nvec

    def getMinDistance2Vec(self, vec):
        "Get min distance to given vec"
        assert self.ndim == vec.ndim
        return self.getVec2VecMinDistance(vec1=self, vec2=vec)

    def getMaxDistance2Vec(self, vec):
        "Get max distance to given vec"
        assert self.ndim == vec.ndim
        return self.getVec2VecMaxDistance(vec1=self, vec2=vec)

    def getMinDistanceVec2Vec(self, vec) -> LocatedVector:
        "Get vec that has the length min distance to given vec"
        assert self.ndim == vec.ndim
        return self.getMinMaxVec2Vec(isMin=True, vec=vec)

    def getMaxDistanceVec2Vec(self, vec) -> LocatedVector:
        "Get vec that has the length max distance to given vec"
        assert self.ndim == vec.ndim
        return self.getMinMaxVec2Vec(isMin=False, vec=vec)


class ImageLocatedVector2D(LocatedVector2D):
    "Extends the line in euclidean space to images"

    def __init__(
            self,
            image: np.ndarray[[np.uint8]],
            vec=None,
            initial_point=None,  # :Point2D:
            final_point=None,  # :Point2D:
            segment=None) -> None:
        ""
        if vec is None:
            super().__init__(initial_point, final_point, segment)
        else:
            spoint = vec.spoint
            epoint = vec.epoint
            super().__init__(spoint, epoint)
        self.image = image
        self.charge = 0

    @staticmethod
    def getConditionDistanceCharge(minDist, minCharge, distanceParentScope,
                                   distanceLocalScope, chargeParentScope,
                                   chargeLocalScope):
        "Set evaluation condition with respect to given booleans"
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
            isMinDistance,  # boolean or none
            isMinCharge):
        "Reinterpretates the base class static method to include charge"
        #
        spoints = vec1.pointList  # starting points
        nspoint = None
        nepoint = None
        svec = None
        distance, charge = cls.getDistanceChargeVariables(
            minDist=isMinDistance, minCharge=isMinCharge)

        for sp in spoints:
            #
            distancePoint = vec2.getPoint2VecDistancePoint(
                point=sp, isMin=isMinDistance)
            epoint = distancePoint[0]
            dist = distancePoint[1]
            tempvec = ImageLocatedVector2D(sp, epoint)
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
    def getVec2VecDistancePointVec(cls, vec1, vec2, isMin: bool):
        "Overrides the base class method"
        return cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=isMin, isMinCharge=None)[:4]

    @classmethod
    def getVec2VecMinCharge(cls, vec1, vec2):
        "Get min charge between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=None, isMinCharge=True)
        return distancePointsVec[4]

    @classmethod
    def getVec2VecMaxCharge(cls, vec1, vec2):
        "Get min distance between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=None, isMinCharge=False)
        return distancePointsVec[4]

    @classmethod
    def getVec2VecMinDistanceMaxChargeDist(cls, vec1, vec2):
        "Get min distance between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=True, isMinCharge=False)
        return distancePointsVec[3]

    @classmethod
    def getVec2VecMinDistanceMinChargeDist(cls, vec1, vec2):
        "Get min distance between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=True, isMinCharge=True)
        return distancePointsVec[3]

    @classmethod
    def getVec2VecMaxDistanceMaxChargeDist(cls, vec1, vec2):
        "Get max distance between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=False, isMinCharge=False)
        return distancePointsVec[3]

    @classmethod
    def getVec2VecMaxDistanceMinChargeDist(cls, vec1, vec2):
        "Get max distance between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=False, isMinCharge=True)
        return distancePointsVec[3]

    @classmethod
    def getVec2VecMinDistanceMaxChargeCharge(cls, vec1, vec2):
        "Get min distance between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=True, isMinCharge=False)
        return distancePointsVec[4]

    @classmethod
    def getVec2VecMinDistanceMinChargeCharge(cls, vec1, vec2):
        "Get min distance between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=True, isMinCharge=True)
        return distancePointsVec[4]

    @classmethod
    def getVec2VecMaxDistanceMaxChargeCharge(cls, vec1, vec2):
        "Get max distance between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=False, isMinCharge=False)
        return distancePointsVec[4]

    @classmethod
    def getVec2VecMaxDistanceMinChargeCharge(cls, vec1, vec2):
        "Get max distance between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=False, isMinCharge=True)
        return distancePointsVec[4]

    @classmethod
    def getVec2VecMinDistanceMaxChargeSPoint(cls, vec1, vec2):
        "Get min distance between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=True, isMinCharge=False)
        return distancePointsVec[0]

    @classmethod
    def getVec2VecMinDistanceMinChargeSPoint(cls, vec1, vec2):
        "Get min distance between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=True, isMinCharge=True)
        return distancePointsVec[0]

    @classmethod
    def getVec2VecMaxDistanceMaxChargeSPoint(cls, vec1, vec2):
        "Get max distance between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=False, isMinCharge=False)
        return distancePointsVec[0]

    @classmethod
    def getVec2VecMaxDistanceMinChargeSPoint(cls, vec1, vec2):
        "Get max distance between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=False, isMinCharge=True)
        return distancePointsVec[0]

    @classmethod
    def getVec2VecMinDistanceMaxChargeEPoint(cls, vec1, vec2):
        "Get min distance between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=True, isMinCharge=False)
        return distancePointsVec[1]

    @classmethod
    def getVec2VecMinDistanceMinChargeEPoint(cls, vec1, vec2):
        "Get min distance between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=True, isMinCharge=True)
        return distancePointsVec[1]

    @classmethod
    def getVec2VecMaxDistanceMaxChargeEPoint(cls, vec1, vec2):
        "Get max distance between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=False, isMinCharge=False)
        return distancePointsVec[1]

    @classmethod
    def getVec2VecMaxDistanceMinChargeEPoint(cls, vec1, vec2):
        "Get max distance between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=False, isMinCharge=True)
        return distancePointsVec[1]

    @classmethod
    def getVec2VecMinDistanceMaxChargeVec(cls, vec1, vec2):
        "Get min distance between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=True, isMinCharge=False)
        return distancePointsVec[2]

    @classmethod
    def getVec2VecMinDistanceMinChargeVec(cls, vec1, vec2):
        "Get min distance between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=True, isMinCharge=True)
        return distancePointsVec[2]

    @classmethod
    def getVec2VecMaxDistanceMaxChargeVec(cls, vec1, vec2):
        "Get max distance between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=False, isMinCharge=False)
        return distancePointsVec[2]

    @classmethod
    def getVec2VecMaxDistanceMinChargeVec(cls, vec1, vec2):
        "Get max distance between two located vectors"
        distancePointsVec = cls._getVec2VecDistancePointChargeVec(
            vec1, vec2, isMinDistance=False, isMinCharge=True)
        return distancePointsVec[2]

    def getMinCharge2Vec(self, vec):
        "Get minimum charge to vector"
        return self.getVec2VecMinCharge(vec1=self, vec2=vec)

    def getMaxCharge2Vec(self, vec):
        "Get minimum charge to vector"
        return self.getVec2VecMaxCharge(vec1=self, vec2=vec)

    def getMinDistMaxCharge2VecDistance(self, vec) -> float:
        """
        Get minimum distance with maximum charge to vector

        Return
        -------
        distance
        """
        return self.getVec2VecMinDistanceMaxChargeDist(vec1=self, vec2=vec)

    def getMinDistMinCharge2VecDistance(self, vec) -> float:
        """
        Get minimum distance with minimum charge to vector

        Return
        -------
        distance

        """
        return self.getVec2VecMinDistanceMinChargeDist(vec1=self, vec2=vec)

    def getMaxDistMaxCharge2VecDistance(self, vec) -> float:
        """
        Get maximum distance with maximum charge to vector

        Return
        -------
        distance


        """
        return self.getVec2VecMaxDistanceMaxChargeDist(vec1=self, vec2=vec)

    def getMaxDistMinCharge2VecDistance(self, vec) -> float:
        """
        Get maximum distance with maximum charge to vector

        Return
        -------
        distance

        """
        return self.getVec2VecMaxDistanceMinChargeDist(vec1=self, vec2=vec)

    def getMinDistMaxCharge2VecCharge(self, vec) -> float:
        """
        Get minimum distance with maximum charge to vector

        Return
        -------
        charge
        """
        return self.getVec2VecMinDistanceMaxChargeCharge(vec1=self, vec2=vec)

    def getMinDistMinCharge2VecCharge(self, vec) -> float:
        """
        Get minimum distance with minimum charge to vector

        Return
        -------
        charge

        """
        return self.getVec2VecMinDistanceMinChargeCharge(vec1=self, vec2=vec)

    def getMaxDistMaxCharge2VecCharge(self, vec) -> float:
        """
        Get maximum distance with maximum charge to vector

        Return
        -------
        charge

        """
        return self.getVec2VecMaxDistanceMaxChargeCharge(vec1=self, vec2=vec)

    def getMaxDistMinCharge2VecCharge(self, vec) -> float:
        """
        Get maximum distance with maximum charge to vector

        Return
        -------
        charge

        """
        return self.getVec2VecMaxDistanceMinChargeCharge(vec1=self, vec2=vec)

    def getMinDistMaxCharge2VecSPoint(self, vec) -> ImagePoint2D:
        """
        Get minimum distance with maximum charge to vector

        Return
        -------
        spoint: starting point which is on self
        """
        return self.getVec2VecMinDistanceMaxChargeSPoint(vec1=self, vec2=vec)

    def getMinDistMinCharge2VecSPoint(self, vec) -> ImagePoint2D:
        """
        Get minimum distance with minimum charge to vector

        Return
        -------
        spoint: starting point which is on self

        """
        return self.getVec2VecMinDistanceMinChargeSPoint(vec1=self, vec2=vec)

    def getMaxDistMaxCharge2VecSPoint(self, vec) -> ImagePoint2D:
        """
        Get maximum distance with maximum charge to vector

        Return
        -------
        spoint: starting point which is on self

        """
        return self.getVec2VecMaxDistanceMaxChargeSPoint(vec1=self, vec2=vec)

    def getMaxDistMinCharge2VecSPoint(self, vec) -> ImagePoint2D:
        """
        Get maximum distance with maximum charge to vector

        Return
        -------
        spoint: starting point which is on self

        """
        return self.getVec2VecMaxDistanceMinChargeSPoint(vec1=self, vec2=vec)

    def getMinDistMaxCharge2VecEPoint(self, vec) -> ImagePoint2D:
        """
        Get minimum distance with maximum charge to vector

        Return
        -------
        epoint: ending point which is on vec
        """
        return self.getVec2VecMinDistanceMaxChargeEPoint(vec1=self, vec2=vec)

    def getMinDistMinCharge2VecEPoint(self, vec) -> ImagePoint2D:
        """
        Get minimum distance with minimum charge to vector

        Return
        -------
        epoint: ending point which is on vec

        """
        return self.getVec2VecMinDistanceMinChargeEPoint(vec1=self, vec2=vec)

    def getMaxDistMaxCharge2VecEPoint(self, vec) -> ImagePoint2D:
        """
        Get maximum distance with maximum charge to vector

        Return
        -------
        epoint: ending point which is on vec

        """
        return self.getVec2VecMaxDistanceMaxChargeEPoint(vec1=self, vec2=vec)

    def getMaxDistMinCharge2VecEPoint(self, vec) -> ImagePoint2D:
        """
        Get maximum distance with maximum charge to vector

        Return
        -------
        epoint: ending point which is on vec

        """
        return self.getVec2VecMaxDistanceMinChargeEPoint(vec1=self, vec2=vec)

    def getMinDistMaxCharge2VecVec(self, vec) -> ImageLocatedVector2D:
        """
        Get minimum distance with maximum charge to vector

        Return
        -------
        vec: vector between the vec and self fulfilling the conditions
        """
        return self.getVec2VecMinDistanceMaxChargeVec(vec1=self, vec2=vec)

    def getMinDistMinCharge2VecVec(self, vec) -> ImageLocatedVector2D:
        """
        Get minimum distance with minimum charge to vector

        Return
        -------
        vec: vector between the vec and self fulfilling the conditions

        """
        return self.getVec2VecMinDistanceMinChargeVec(vec1=self, vec2=vec)

    def getMaxDistMaxCharge2VecVec(self, vec) -> ImageLocatedVector2D:
        """
        Get maximum distance with maximum charge to vector

        Return
        -------
        vec: vector between the vec and self fulfilling the conditions

        """
        return self.getVec2VecMaxDistanceMaxChargeVec(vec1=self, vec2=vec)

    def getMaxDistMinCharge2VecVec(self, vec) -> ImageLocatedVector2D:
        """
        Get maximum distance with maximum charge to vector

        Return
        -------
        vec: vector between the vec and self fulfilling the conditions

        """
        return self.getVec2VecMaxDistanceMinChargeVec(vec1=self, vec2=vec)

    def setLine(self) -> None:
        "Overrides base class method"
        plist = self._getStraightLine(point1=self.p_init, point2=self.p_final)
        plist = [
            ImagePoint2D(spacePoint=p, image=self.image).setPointProperties()
            for p in plist
        ]
        self.pointList = plist
        return None

    def setVecCharge(self) -> None:
        "Set vector charge"
        counter = 0
        for p in self.pointList:
            counter += p.pixel_energy
        self.charge = counter
        return None

    def setVecProperties(self) -> None:
        "Overrides base class method"
        self.setLine()
        self.setVecCharge()
        return None

    def getNearestPoint2Point(self, point) -> ImagePoint2D:
        "Overrides base class method"
        assert point.ndim == self.ndim
        npoint = self._getNearestPointOnVec(vec=self, point=point)
        impoint = ImagePoint2D(image=self.image, spacePoint=npoint)
        impoint.setPointProperties()
        return impoint

    def getFarthestPoint2Point(self, point) -> ImagePoint2D:
        "Overrides base class method to return imagepoint"
        assert point.ndim == self.ndim
        npoint = self._getFarthestPointOnVec(vec=self, point=point)
        impoint = ImagePoint2D(image=self.image, spacePoint=npoint)
        impoint.setPointProperties()
        return impoint

    def getMinVec2Vec(self, vec):  # -> imagelocatedvector2d
        "Overrides base class method to return image located vector"
        assert self.ndim == vec.ndim
        nvec = self.getVec2VecMinVec(vec1=self, vec2=vec)
        imvec = ImageLocatedVector2D(vec=nvec, image=self.image)
        imvec.setVecProperties()
        return imvec

    def getMaxVec2Vec(self, vec):  # -> ImageLocatedVector2D
        "Overrides base class method to return image located vector"
        assert self.ndim == vec.ndim
        nvec = self.getVec2VecMaxVec(vec1=self, vec2=vec)
        imvec = ImageLocatedVector2D(vec=nvec, image=self.image)
        imvec.setVecProperties()
        return imvec
