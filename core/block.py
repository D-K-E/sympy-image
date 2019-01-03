# Regroups objects with respect to point blocks
# Author: Kaan Eraslan
# Licensing: see, LICENSE

# Packages

import numpy as np
from sympy.geometry import polygon

from point import ImagePoint2D, Point2D
from vector import ImageLocatedVector2D, LocatedVector2D

# Point methods:
#     Given: a point

#     Operations:

#     - find the nearest side in block to the given point using closest point
#       on side.
#     - find the nearest side in block to the given point using farthest point
#       on side.
#     - find the point in block with least charge given the point
#     - find the nearest side in block with least charge given the point using
#       closest point on side
#     - find the nearest side in block with least charge given the point using
#       farthest point on side
#     - find the nearest side in block with least vector charge given the point
#       using closest point on side
#     - find the nearest side in block with least vector charge given the point
#       using farthest point on side
#     - find the farthest side in block with max charge given the point using
#       closest point on side
#     - find the farthest side in block with max charge given the point using
#       nearest point on side
#     - find the farthest side in block with least vector charge given the point
#       using closest point on side
#     - find the farthest side in block with least vector charge given the point
#       using farthest point on side
#     - find the farthest side in block with max vector charge given the point
#       using closest point on side
#     - find the farthest side in block with max vector charge given the point
#       using farthest point on side

#     Each of these operations can give distance, charge, vector, point, side as
#     output

#     Vector methods:

#     Given: an image located vector2d

#     Operations:

# - find the nearest point in block to the given vector
# - find the nearest point in block with least charge given the vector using
#       nearest point on vector
# - find the nearest point in block with least charge given the vector using
#   farthest point on vector
# - find the nearest point in block with least vector charge given the vector
#       using nearest point on vector
#     - find the nearest point in block with least vector charge given the vector
#       using farthest point on vector
#     - find the farthest point in block with max charge given the vector using
#       nearest point on vector
#     - find the farthest point in block with max charge given the vector using
#       farthest point on vector
#     - find the farthest point in block with least vector charge given the vector using
#       nearest point on vector
#     - find the farthest point in block with least vector charge given the vector using
#       farthest point on vector
#     - find the farthest point in block with max vector charge given the vector using
#       nearest point on vector
#     - find the farthest point in block with max vector charge given the vector using
#       farthest point on vector

#     - find the nearest side in block to the given vector
#     - find the nearest side in block with least charge given the vector using
#       nearest point on vector with nearest point in side
#     - find the nearest side in block with least charge given the vector using
#       nearest point on vector with farthest point in side
#     - find the nearest side in block with least charge given the vector using
#       farthest point on vector with nearest point in side
#     - find the nearest side in block with least charge given the vector using
#       farthest point on vector with farthest point in side
#     - find the nearest side in block with least vector charge given the vector
#       using nearest point on vector with nearest point in side
#     - find the nearest side in block with least vector charge given the vector
#       using nearest point on vector with farthest point in side
#     - find the nearest side in block with least vector charge given the vector
#       using farthest point on vector with nearest point in side
#     - find the nearest side in block with least vector charge given the vector
#       using farthest point on vector with farthest point in side

#     - find the farthest side in block with max charge given the vector using
#       nearest point on vector with nearest point in side
#     - find the farthest side in block with max charge given the vector using
#       nearest point on vector with farthest point in side
#     - find the farthest side in block with max charge given the vector using
#       farthest point on vector with nearest point in side
#     - find the farthest side in block with least vector charge given the vector using
#       nearest point on vector with nearest point in side
#     - find the farthest side in block with least vector charge given the vector using
#       farthest point on vector with nearest point in side
#     - find the farthest side in block with max vector charge given the vector using
#       nearest point on vector with nearest point in side
#     - find the farthest side in block with max vector charge given the vector using
#       farthest point on vector with nearest point in side
#     - find the farthest side in block with max charge given the vector using
#       farthest point on vector with farthest point in side
#     - find the farthest side in block with least vector charge given the vector using
# using
#       farthest point on vector with nearest point in side
#     - find the farthest side in block with max vector charge given the vector us
# ing
#       nearest point on vector with nearest point in side
#     - find the farthest side in block with max vector charge given the vector us
# ing
#       farthest point on vector with nearest point in side
#     - find the farthest side in block with max charge given the vector using
#       farthest point on vector with farthest point in side
#     - find the farthest side in block with least vector charge given the vector
# using
#       nearest point on vector with farthest point in side
#     - find the farthest side in block with least vector charge given the vector using
#       farthest point on vector with farthest point in side
#     - find the farthest side in block with max vector charge given the vector using
#       nearest point on vector with farthest point in side
#     - find the farthest side in block with max vector charge given the vector using
#       farthest point on vector with farthest point in side


class Point2DBlock(polygon.Polygon):
    """
    A grouping of a euclidean points
    """

    def __init__(
            self,
            radius=None,  # :int: of the circle that circumscribes the
            # polygon
            centerPoint=None,  # :Point2D:
            pointList=None,  # :[Point2D]:
            nb_sides=None):  # :int:
        # Initiate either a polygon or a regular polygon
        if pointList is None and nb_sides is None:
            raise ValueError(
                'please provide either a pointlist or number of sides to'
                ' instantiate')
        elif pointList is not None and nb_sides is not None:
            raise ValueError(
                'please provide either a pointlist or number of sides not both'
            )
        elif nb_sides is not None:
            if radius is None or centerPoint is None:
                raise ValueError('Please provide both radius and center point')
            else:
                super().__init__(centerPoint, radius, n=nb_sides)
        elif pointList is not None:
            super().__init__(pointList)

        self.blockpoints = []
        return None

    @staticmethod
    def _getPointsInBlock(block) -> list:  #
        "Get a list of points contained in the block"
        xmin, ymin, xmax, ymax = block.bounds  # xmin, ymin, xmax, ymax
        plist = []
        for i in range(xmin, xmax + 1):
            for k in range(ymin, ymax + 1):
                point = Point2D(i, k)
                if point in block:
                    plist.append(point)
        return plist

    def setBlockpoints(self):
        "Set block points"
        self.blockpoints = self._getPointsInBlock(block=self)

    @staticmethod
    def _getPointInBlockByDistance(block,
                                   point: Point2D,
                                   minDist: bool):
        "Get min/max distance block point to given point"
        if point in block:
            return point, 0
        if minDist is True:
            dist = float('inf')
        else:
            dist = float('-inf')
        points = block.vertices
        result = None
        for p in points:
            distance = p.distance(point)
            if minDist is True:
                condition = distance <= dist
            else:
                condition = distance > dist
            if condition:
                dist = distance
                result = p
        return result, dist

    @classmethod
    def _getNearestPointInBlock(cls, block, 
                                point: Point2D) -> Point2D:
        "Get the point in block that is closest to given point "
        return cls._getPointInBlockByDistance(block,
                                              point, minDist=True)[0]
        
    def getNearestPointInBlock(self, point) -> Point2D:
        "Wrapper method for class instance"
        return self._getNearestPointInBlock(block=self,
                                            point=point)

    @classmethod
    def _getFarthestPointInBlockWithDist(cls, block, point: Point2D):
        "Get farthest point in block to the given point"
        return cls._getPointInBlockByDistance(block, point, minDist=False)
        
    @classmethod
    def _getFarthestDistanceInBlock2Point(cls, block, point: Point2D) -> float:
        "Get farthest distance from a point in block to the given point"
        return cls._getFarthestPointInBlockWithDist(block, point)[0]

    @classmethod
    def _getFarthestPointInBlock2Point(cls, block, point) -> Point2D:
        "Get farthest distance from a point in block to the given point"
        return cls._getFarthestPointInBlockWithDist(block, point)[1]

    def getFarthestDistanceInBlock2Point(self, point: Point2D) -> float:
        "Get farthest distance from block to the given point"
        return self._getFarthestDistanceInBlock2Point(block=self, point=point)

    def getFarthestPointInBlock2Point(self, point: Point2D) -> Point2D:
        "Get farthest distance from block to the given point"
        return self._getFarthestPointInBlock2Point(block=self, point=point)

    @staticmethod
    def _getBlockSideByDistance(block, point: Point2D, isMinFuncs: bool,
                                isNear: bool) -> [float, LocatedVector2D]:
        """
        Give the block side corresponding to distance based criteria

        Possible output options are following:
        Block side with shortest distance to given point where
        the distance is calculated from the closest point on side

        Block side with shortest distance to given point where
        the distance is calculated from the farthest point on side

        Block side with farthest distance to given point where
        the distance is calculated from the closest point on side

        Block side with farthest distance to given point where
        the distance is calculated from the farthest point on side
        """
        sides = block.sides
        if isNear is True:
            dist = float('inf')
        else:
            dist = float('-inf')
        sideInBlock = None
        for side in sides:
            # side is a Segment
            vec = LocatedVector2D(segment=side)
            vec.setVecProperties()
            p, distance = vec._getPoint2VecDistancePoint(aVec=vec, 
                                                         point=point,
                                                         isMin=isMaxFuncs)
            if isNear is True:
                condition = distance <= dist
            else:
                condition = distance > dist
            if condition:
                dist = distance
                sideInBlock = side

        sideInBlock = LocatedVector2D(segment=sideInBlock)
        sideInBlock.setVecProperties()
        return dist, sideInBlock

    @classmethod
    def _getSideDistanceInBlock2Point(cls, block, point: Point2D,
                                      isNear: bool):
        """
        Get nearest side in block to the given point
        the distance is calculated from the nearest point on side
        """
        return cls._getBlockSideByDistance(block, point, 
                                           isNear=isNear, isMinFuncs=True)

    @classmethod
    def _getSideWithMaxDistanceInBlock2Point(cls, block, point: Point2D,
                                             isNear:bool):
        """
        Get nearest or farthest side in block to the given point.

        The distance is calculated from farthest point on the side
        """
        return cls._getBlockSideByDistance(block, point, isNear=isNear,
                                           isMinFuncs=False)
        
    @classmethod
    def _getCSideDistanceInBlock2Point(cls, block, point):
        "Get nearest side and distance in block to given point"
        return cls._getSideDistanceInBlock2Point(
            block, point, isNear=True)

    @classmethod
    def _getCSideInBlock2Point(cls, block, point) -> LocatedVector2D:
        "get nearest side in block to given point"
        return cls._getCSideDistanceInBlock2Point(block, point)[1]

    @classmethod
    def _getCDist2SideInBlock2Point(cls, block, point) -> float:
        "get nearest side in block to given point"
        return cls._getCSideDistanceInBlock2Point(block, point)[0]

    @classmethod
    def _getFSideDistanceInBlock2Point(cls, block, point):
        "Get nearest side and distance in block to given point"
        return cls._getSideDistanceInBlock2Point(
            block, point, isNear=False)

    @classmethod
    def _getFSideInBlock2Point(cls, block, point) -> LocatedVector2D:
        "get nearest side in block to given point"
        return cls._getFSideDistanceInBlock2Point(block, point)[1]

    @classmethod
    def _getFDist2SideInBlock2Point(cls, block, point):
        "get nearest side in block to given point"
        return cls._getFSideDistanceInBlock2Point(block, point)[0]

    @classmethod
    def _getCSideMaxDistanceInBlock2Point(cls, block, point):
        """
        Get closest side and distance in block to given point
        
        The distance is calculated from farthest point on the side
        """
        return cls._getSideWithMaxDistanceInBlock2Point(block, point,
                                                        isNear=True)

    @classmethod
    def _getCSideMDInBlock2Point(cls, block, point) -> LocatedVector2D:
        """
        get nearest side in block to given point
        
        distance computed from farthest point on side
        """
        return cls._getCSideMaxDistanceInBlock2Point(block, point)[1]

    @classmethod
    def _getCDistMD2SideInBlock2Point(cls, block, point) -> float:
        "get nearest side in block to given point"
        return cls._getCSideMaxDistanceInBlock2Point(block, point)[0]

    @classmethod
    def _getFSideDistanceMDInBlock2Point(cls, block, point):
        "Get nearest side and distance in block to given point"
        return cls._getSideWithMaxDistanceInBlock2Point(
            block, point, isNear=False)

    @classmethod
    def _getFSideInBlock2Point(cls, block, point) -> LocatedVector2D:
        "get farthest side in block to given point"
        return cls._getFSideDistanceMDInBlock2Point(block, point)[1]

    @classmethod
    def _getFDist2SideInBlock2Point(cls, block, point):
        "get nearest side in block to given point"
        return cls._getFSideDistanceMDInBlock2Point(block, point)[0]

    @classmethod
    def _getSideInBlock2VecByDistance(
            cls,
            block,
            vec: LocatedVector2D,
            isMin: bool,  # get minimum side or distance overall
            isMinFuncs: bool  # use maximum or minimum distance function
    ):
        "Get nearest side in block to the given vec"
        sides = block.sides
        sideInBlock = None

        if isMin is True:
            dist = float('inf')
        else:
            dist = float('-inf')

        for side in sides:
            # side is a Segment
            sidevec = LocatedVector2D(segment=side)

            distance = sidevec.getMinMaxDistance2Vec(isMin=isMaxFuncs,
                                                     vec=vec)

            if isMin is True:
                condition = distance < dist
            else:
                condition = distance > dist

            if condition:
                dist = distance
                sideInBlock = side

        sideInBlock = LocatedVector2D(segment=sideInBlock)
        return dist, sideInBlock

    @classmethod
    def _getClosestSideAndDistanceInBlock2VecWithMaxDist(cls, block, vec):
        "Get nearest side and distance in block to given vec"\
            " using minimum distance function"
        return cls._getSideInBlock2VecByDistance(
            block=block, vec=vec, isMin=True, isMinFuncs=True)

    @classmethod
    def _getClosestSideInBlock2VecWithMaxDist(cls, block,
                                              vec) -> LocatedVector2D:
        "Get nearest side in block to given vector using maximum distance"
        return cls._getClosestSideAndDistanceInBlock2VecWithMaxDist(
            block=block, vec=vec)[1]

    @classmethod
    def _getClosestSideDistInBlock2VecWithMaxDist(cls, block, vec) -> float:
        "Get nearest side in block to given vector using maximum distance"
        return cls._getClosestSideAndDistanceInBlock2VecWithMaxDist(
            block=block, vec=vec)[0]

    @classmethod
    def _getFarSideAndDistanceInBlock2VecWithMaxDist(cls, block, vec):
        "Get farthest side and distance in block to given vec"\
            " using maximum distance function"
        return cls._getSideInBlock2VecByDistance(
            block=block, vec=vec, isMin=False, isMaxFuncs=True)

    @classmethod
    def _getFarSideInBlock2VecWithMaxDist(cls, block, vec) -> LocatedVector2D:
        "Get farthest side in block to given vector using maximum distance"
        return cls._getFarSideAndDistanceInBlock2VecWithMaxDist(
            block=block, vec=vec)[1]

    @classmethod
    def _getFarSideDistInBlock2VecWithMaxDist(cls, block, vec) -> float:
        "Get farthest side in block to given vector using maximum distance"
        return cls._getFarSideAndDistanceInBlock2VecWithMaxDist(
            block=block, vec=vec)[0]

    @classmethod
    def _getClosestSideAndDistanceInBlock2VecWithMinDist(cls, block, vec):
        "Get nearest side and distance in block to given vec"\
            " using minimum distance function"
        return cls._getSideInBlock2VecByDistance(
            block=block, vec=vec, isMin=True, isMaxFuncs=False)

    @classmethod
    def _getClosestSideInBlock2VecWithMinDist(cls, block,
                                              vec) -> LocatedVector2D:
        "Get nearest side in block to given vector using maximum distance"
        return cls._getClosestSideAndDistanceInBlock2VecWithMinDist(
            block=block, vec=vec)[1]

    @classmethod
    def _getClosestSideDistInBlock2VecWithMinDist(cls, block, vec) -> float:
        "Get nearest side in block to given vector using maximum distance"
        return cls._getClosestSideAndDistanceInBlock2VecWithMinDist(
            block=block, vec=vec)[0]

    @classmethod
    def _getFarSideAndDistanceInBlock2VecWithMinDist(cls, block, vec):
        "Get farthest side and distance in block to given vec"\
            " using maximum distance function"
        return cls._getSideInBlock2VecByDistance(
            block=block, vec=vec, isMin=False, isMaxFuncs=False)

    @classmethod
    def _getFarSideInBlock2VecWithMinDist(cls, block, vec) -> LocatedVector2D:
        "Get farthest side in block to given vector using maximum distance"
        return cls._getFarSideAndDistanceInBlock2VecWithMinDist(
            block=block, vec=vec)[1]

    @classmethod
    def _getFarSideDistInBlock2VecWithMinDist(cls, block, vec) -> float:
        "Get farthest side in block to given vector using maximum distance"
        return cls._getFarSideAndDistanceInBlock2VecWithMinDist(
            block=block, vec=vec)[0]

    def getClosestSideOrDistanceWithDistFunc(
            self,
            other,  # segment or
            # locatedvector2d
            useMaxDistance: bool,
            justDistance: bool):
        "Regroups methods related closest distance or side"
        if useMaxDistance is True:
            if justDistance is False:
                return self._getClosestSideInBlock2VecWithMaxDist(
                    block=self, vec=other)  # LocatedVector2D
            else:
                return self._getClosestSideDistInBlock2VecWithMaxDist(
                    block=self, vec=other)
        else:
            if justDistance is False:
                return self._getClosestSideInBlock2VecWithMinDist(
                    block=self, vec=other)  # LocatedVector2D
            else:
                return self._getClosestSideDistInBlock2VecWithMinDist(
                    block=self, vec=other)

    def getFarSideOrDistanceWithDistFunc(self, other, useMaxDistance,
                                         justDistance):
        "Regroups methods related to farthest distance or side"
        if useMaxDistance is True:
            if justDistance is False:
                return self._getFarSideInBlock2VecWithMaxDist(
                    block=self, vec=other)  # LocatedVector2D
            else:
                return self._getFarSideDistInBlock2VecWithMaxDist(
                    block=self, vec=other)
        else:
            if justDistance is False:
                return self._getFarSideInBlock2VecWithMinDist(
                    block=self, vec=other)  # LocatedVector2D
            else:
                return self._getFarSideDistInBlock2VecWithMinDist(
                    block=self, vec=other)

    def getCloseFarSide2OtherDistFunc(self, other, justDistance: bool,
                                      isNearest: bool, useMaxDistance: bool):
        "Get nearest side to other"
        # other should be either a point or a segment or locatedvector2d
        if (issubclass(other, Point2D) is not True
                and issubclass(other, polygon.Segment) is not True
                and issubclass(other, LocatedVector2D) is not True):
            raise ValueError('other should be either a point2d'
                             ' or a segment or a locatedvector2d'
                             'you have provided: {}'.format(type(other)))
        #
        if isNearest is True:
            if isinstance(other, Point2D):
                if justDistance is False:
                    return self._getCSideInBlock2Point(
                        block=self, point=other)
                else:
                    return self._getCDist2SideInBlock2Point(
                        block=self, point=other)
            else:
                return self.getClosestSideOrDistanceWithDistFunc(
                    other, useMaxDistance, justDistance)
        else:
            if isinstance(other, Point2D):
                return self._getFSideInBlock2Point(block=self, point=other)
            else:
                return self.getFarSideOrDistanceWithDistFunc(
                    other, useMaxDistance, justDistance)

    def getClosestSide2OtherMaxDist(self, other) -> LocatedVector2D:
        "Get nearest side to other"
        # other should be either a point or a segment or locatedvector2d
        return self.getCloseFarSide2OtherDistFunc(
            other, isNearest=True, justDistance=False, useMaxDistance=True)

    def getClosestSide2OtherMinDist(self, other) -> LocatedVector2D:
        "Get nearest side to other with minimum distance functions"
        # other should be either a point or a segment or locatedvector2d
        return self.getCloseFarSide2OtherDistFunc(
            other, isNearest=True, justDistance=False, useMaxDistance=False)

    def getFarSide2OtherMaxDist(self, other) -> LocatedVector2D:
        "Get farthest side to other"
        # other should be either a point or a segment or locatedvector2d
        return self.getCloseFarSide2OtherDistFunc(
            other, isNearest=False, justDistance=False, useMaxDistance=True)

    def getFarSide2OtherMinDist(self, other) -> LocatedVector2D:
        "Get far side to other"
        # other should be either a point or a segment or locatedvector2d
        return self.getCloseFarSide2OtherDistFunc(
            other, isNearest=False, justDistance=False, useMaxDistance=False)

    def getClosestSideDist2OtherMaxDist(self, other):
        "Get closest side's distance to other using max distance function"
        return self.getCloseFarSide2OtherDistFunc(
            other, isNearest=True, justDistance=True, useMaxDistance=True)

    def getClosestSideDist2OtherMinDist(self, other):
        "Get closest side's distance to other using max distance function"
        return self.getCloseFarSide2OtherDistFunc(
            other, isNearest=True, justDistance=True, useMaxDistance=False)

    def getFarSideDist2OtherMinDist(self, other):
        "Get closest side's distance to other using max distance function"
        return self.getCloseFarSide2OtherDistFunc(
            other, isNearest=False, justDistance=True, useMaxDistance=False)

    def getFarSideDist2OtherMaxDist(self, other):
        "Get closest side's distance to other using max distance function"
        return self.getCloseFarSide2OtherDistFunc(
            other, isNearest=False, justDistance=True, useMaxDistance=True)

    @staticmethod
    def _checkConsecutive(block1, block2):
        "Check if 2 blocks are consecutive"
        checkval = False
        # 2 blocks are consecutive if they intersect
        if len(block1.intersect(block2)) > 0:
            checkval = True
        return checkval

    @staticmethod
    def filterContainedBlocksInBlock(block1, blocks):
        'Filter out blocks that are contained in block1'
        return [block for block in blocks if block not in block1]

    @classmethod
    def getUncontainedBlocks(cls, block, blocks):
        "Filter blocks for contained and then add containing block to list"
        uncontained = cls.filterContainedBlocksInBlock(block1=block,
                                                       blocks=blocks)
        uncontained.append(block)
        return uncontained

    @classmethod
    def filterUnconsecutiveBlocks(cls, block, blocks):
        "Filter out blocks that are not consecutive to the block"
        return [
            b for b in blocks if cls._checkConsecutive(block1=block, block2=b)
        ]

    @classmethod
    def _addBlock2Block(cls, block1, block2):
        "Add one block to another if they are consecutive"
        if cls._checkConsecutive(block1, block2) is False:
            raise ValueError(
                'Blocks need to be consecutive (intersect) for addition')
        else:
            v1 = block1.vertices
            v2 = block2.vertices
            v1.extend(v2)
            newblock = Point2DBlock(pointList=v1)
            newblock.setBlockpoints()
            return newblock

    @classmethod
    def _addBlocks2Block(cls, block, blocks):
        "Add consecutive blocks to given block"
        consecutive_blocks = cls.filterUnconsecutiveBlocks(block, blocks)
        if len(consecutive_blocks) == 0:
            return None
        for cblock in consecutive_blocks:
            block = cls._addBlock2Block(block1=block, block2=cblock)
        return block

    @classmethod
    def foldrBlocks(cls, blocks: list):
        "Merge blocks after filtering unmergeable ones"
        if len(blocks) == 1:
            return blocks[0]
        block = blocks.pop()
        block = cls._addBlocks2Block(block, blocks)
        if block is not None:
            blocks.append(block)
        cls.foldrBlocks(blocks)

    def add2Block(self, block):
        "Add block to current block"
        return self._addBlock2Block(block1=self, block2=block)

    def mergeBlocksWithSelf(self, blocks):
        "Merge blocks with instance block"
        blockscp = blocks.copy()
        blockscp.append(self)
        mergedBlock = self.foldrBlocks(blocks)
        if self in mergedBlock is not True:
            raise ValueError('List of blocks that are not mergeable'
                             ' with self block')
        else:
            return mergedBlock


class ImagePoint2DBlock(Point2DBlock):
    "Extends point block object to images"

    def __init__(
            self,
            image: np.ndarray[[np.uint8]],
            nb_sides=None,
            radius=None,  # :int: of the circle that circumscribes the
            # polygon
            centerPoint=None,  # :Point2D:
            pointList=None) -> None:
        super().__init__(
            centerPoint=centerPoint,
            pointList=pointList,
            radius=radius,
            nb_sides=nb_sides)
        self.image = image
        self.charge = 0

        return None

    # Should override the instance methods of base class
    @staticmethod
    def _getDistanceChargeVariables(isMinDist, isMinCharge):
        "Wrapper around image located vector for setting variables"
        return ImageLocatedVector2D.getDistanceChargeVariables(
            minDist=isMinDist,
            minCharge=isMinCharge)

    @staticmethod
    def _getConditionDistanceCharge(isMinDist, isMinCharge,
                                    distanceParentScope,
                                    distanceLocalScope,
                                    chargeParentScope,
                                    chargeLocalScope):
        "Wrapper around image located vector for setting condition"
        return ImageLocatedVector2D.getConditionDistanceCharge(
            isMinDist,
            isMinCharge,
            distanceParentScope,
            distanceLocalScope,
            chargeParentScope,
            chargeLocalScope)

    @staticmethod
    def _getFarNearPointInBlockWithDistMinMaxCharge(block,
                                                    point: ImagePoint2D,
                                                    isMinCharge: bool,
                                                    isPoint: bool,
                                                    isMinDist: bool):
        """
        Given a point and a block give either farthest or nearest point
        with maximum or minimum charge in block

        Note: For calculating charge we have two options, we can either look
        for the charge of the block point, or charge of the vector that
        reunites the block point to the given point
        note: isPoint: True -> point, False -> vec

        """
        bpoint = None
        bpvec = None

        bpoints = block.blockpoints
        dist, charge = block._getDistanceChargeVariables(
            isMinCharge=isMinCharge,
            isMinDist=isMinDist)
        for bp in bpoints:
            tempdist = bp.distance(point)
            if isPoint is True:
                tempcharge = bp.pixel_energy
                vec = None
            else:
                vec = ImageLocatedVector2D(initial_point=bp,
                                           image=block.image,
                                           final_point=point)
                vec.setVecProperties()
                tempcharge = vec.charge

            checkval = block._getConditionDistanceCharge(
                isMinDist=isMinDist,
                isMinCharge=isMinCharge,
                distanceParentScope=dist,
                distanceLocalScope=tempdist,
                chargeParentScope=charge,
                chargeLocalScope=tempcharge)
            if checkval:
                dist = tempdist
                charge = tempcharge
                bpoint = bp
                bpvec = vec
        return dist, charge, bpoint, bpvec

    @classmethod
    def _getNearestPointInBlock(cls, block, point) -> ImagePoint2D:
        "Overrides the base class method"
        (dist, charge,
         point, vec) = cls._getFarNearPointInBlockWithDistMinMaxCharge(
            block=block,
            point=point,
            isMinCharge=None,
            isPoint=True,
            isMinDist=True)
        point.setPointProperties()

        return point

    @classmethod
    def _getFarthestPointInBlockWithDist(cls, block,
                                         point: Point2D) -> ImagePoint2D:
        "Overrides the base class method"
        (dist, charge,
         point, vec) = cls._getFarNearPointInBlockWithDistMinMaxCharge(
            block=block,
            point=point,
            isMinCharge=None,
            isPoint=True,
            isMinDist=False)
        point.setPointProperties()

        return dist, point

    @classmethod
    def _getNearestPointVecInBlockWithDist(cls, block, point) -> ImagePoint2D:
        "Overrides the base class method"
        (dist, charge,
         point, vec) = cls._getFarNearPointInBlockWithDistMinMaxCharge(
            block=block,
            point=point,
            isMinCharge=None,
            isPoint=False,
            isMinDist=True)
        vec.setVecProperties()

        return dist, vec

    @classmethod
    def _getFarthestPointVecInBlockWithDist(cls, block,
                                            point: Point2D) -> ImagePoint2D:
        "Overrides the base class method"
        (dist, charge,
         point, vec) = cls._getFarNearPointInBlockWithDistMinMaxCharge(
            block=block,
            point=point,
            isMinCharge=None,
            isPoint=False,
            isMinDist=False)
        vec.setVecProperties()

        return dist, vec

    @classmethod
    def _getFarthestPointVecInBlock(cls,
                                    block,
                                    point: Point2D) -> ImageLocatedVector2D:
        "Get the vector between farthest point in block and given point"
        return cls._getFarthestPointVecInBlockWithDist(block, point)[1]

    @classmethod
    def _getNearestPointVecInBlock(cls,
                                   block,
                                   point: Point2D) -> ImageLocatedVector2D:
        "Get the vector between farthest point in block and given point"
        return cls._getNearestPointVecInBlockWithDist(block, point)[1]

    @classmethod
    def _getMinChargePointChargeInBlock(cls, block) -> list:
        "Get minimum charged point in block"
        p = Point2D(x=0, y=0)
        (dist, charge,
         point, vec) = cls._getFarNearPointInBlockWithDistMinMaxCharge(
            block=block,
            point=p,
            isMinCharge=True,
            isPoint=True,
            isMinDist=None)
        return point, charge

    @classmethod
    def _getMinimumPointChargeInBlock(cls, block) -> float:
        "Get minimum point charge in block"
        return cls._getMinChargePointChargeInBlock(block)[1]

    @classmethod
    def _getMinChargePointInBlock(cls, block) -> float:
        "Get minimum point charge in block"
        return cls._getMinChargePointChargeInBlock(block)[0]

    @classmethod
    def _getMaxChargePointChargeInBlock(cls, block) -> list:
        "Get point with maximum charge in block and its charge"
        p = Point2D(x=0, y=0)
        (dist, charge,
         point, vec) = cls._getFarNearPointInBlockWithDistMinMaxCharge(
            block=block,
            point=p,
            isMinCharge=False,
            isPoint=True,
            isMinDist=None)
        return point, charge

    @classmethod
    def _getMaximumPointChargeInBlock(cls, block) -> float:
        "Get minimum point charge in block"
        return cls._getMaxChargePointChargeInBlock(block)[1]

    @classmethod
    def _getMaxChargePointInBlock(cls, block) -> float:
        "Get minimum point charge in block"
        return cls._getMaxChargePointChargeInBlock(block)[0]

    @classmethod
    def _getNearestMinChargePointChargeDistInBlock(cls,
                                                   block,
                                                   point) -> list:
        """
        Get nearest point with minimum charge
        outputs charge, distance, and point
        """
        (dist, charge,
         point, vec) = cls._getFarNearPointInBlockWithDistMinMaxCharge(
            block=block,
            point=point,
            isMinCharge=True,
            isPoint=True,
            isMinDist=True)
        return point, charge, dist

    @classmethod
    def _getNearestMinChargePointDistInBlock(cls, block, point) -> float:
        "Get distance of the point with minimum charge and minimum distance"
        return cls._getNearestMinChargePointChargeDistInBlock(block, point)[2]

    @classmethod
    def _getNearestMinChargePointChargeInBlock(cls, block, point) -> float:
        "Get distance of the point with minimum charge and minimum distance"
        return cls._getNearestMinChargePointChargeDistInBlock(block, point)[1]

    @classmethod
    def _getNearestMinChargePointInBlock(cls, block, point) -> float:
        "Get distance of the point with minimum charge and minimum distance"
        return cls._getNearestMinChargePointChargeDistInBlock(block, point)[0]

    @classmethod
    def _getNearestMinChargePointVecChargeDistInBlock(cls,
                                                      block,
                                                      point) -> list:
        """
        Get nearest point with minimum charge
        outputs charge, distance, and point
        """
        (dist, charge,
         point, vec) = cls._getFarNearPointInBlockWithDistMinMaxCharge(
            block=block,
            point=point,
            isMinCharge=True,
            isPoint=False,
            isMinDist=True)
        return vec, charge, dist

    @classmethod
    def _getNearestMinChargePointVecChargeInBlock(cls, block, point) -> float:
        "Get the charge of the vector between nearest point in block and " \
            "given point"
        return cls._getNearestMinChargePointVecChargeDistInBlock(block,
                                                                 point)[1]

    @classmethod
    def _getNearestMinChargePointVecInBlock(cls,
                                            block,
                                            point) -> ImageLocatedVector2D:
        "Get the charge of the vector between nearest point in block and " \
            "given point"
        return cls._getNearestMinChargePointVecChargeDistInBlock(block,
                                                                 point)[0]

    @classmethod
    def _getNearestMaxChargePointChargeDistInBlock(cls,
                                                   block,
                                                   point) -> list:
        """
        Get nearest point with minimum charge
        outputs charge, distance, and point
        """
        (dist, charge,
         point, vec) = cls._getFarNearPointInBlockWithDistMinMaxCharge(
            block=block,
            point=point,
            isMinCharge=False,
            isPoint=True,
            isMinDist=True)
        return point, charge, dist

    @classmethod
    def _getNearestMaxChargePointDistInBlock(cls, block, point):
        "Get distance of nearest point with maximum charge in block to point"
        return cls._getNearestMaxChargePointDistInBlock(block, point)[2]

    @classmethod
    def _getNearestMaxChargePointChargeInBlock(cls, block, point):
        "Get distance of nearest point with maximum charge in block to point"
        return cls._getNearestMaxChargePointDistInBlock(block, point)[1]

    @classmethod
    def _getNearestMaxChargePointInBlock(cls, block, point):
        "Get distance of nearest point with maximum charge in block to point"
        return cls._getNearestMaxChargePointDistInBlock(block, point)[0]

    @classmethod
    def _getNearestMaxChargePointVecChargeDistInBlock(cls,
                                                      block,
                                                      point) -> list:
        """
        Get nearest point with minimum charge
        outputs charge, distance, and point
        """
        (dist, charge,
         point, vec) = cls._getFarNearPointInBlockWithDistMinMaxCharge(
            block=block,
            point=point,
            isMinCharge=False,
            isPoint=False,
            isMinDist=True)
        return vec, charge, dist

    @classmethod
    def _getNearestMaxChargePointVecChargeInBlock(cls, block, point):
        "Get charge of vec nearest point with maximum charge in block to point"
        return cls._getNearestMaxChargePointVecDistInBlock(block, point)[1]

    @classmethod
    def _getNearestMaxChargePointVecInBlock(cls, block, point):
        "Get distance of nearest point with maximum charge in block to point"
        return cls._getNearestMaxChargePointVecDistInBlock(block, point)[0]

    @classmethod
    def _getFarthestMinChargePointChargeDistInBlock(cls,
                                                    block,
                                                    point) -> list:
        """
        Get nearest point with minimum charge
        outputs charge, distance, and point
        """
        (dist, charge,
         point, vec) = cls._getFarNearPointInBlockWithDistMinMaxCharge(
            block=block,
            point=point,
            isMinCharge=True,
            isPoint=True,
            isMinDist=False)
        return point, charge, dist

    @classmethod
    def _getFarthestMinChargePointDistInBlock(cls, block, point) -> float:
        "Get distance of the point with minimum charge and minimum distance"
        return cls._getFarthestMinChargePointChargeDistInBlock(block, point)[2]

    @classmethod
    def _getFarthestMinChargePointChargeInBlock(cls, block, point) -> float:
        "Get distance of the point with minimum charge and minimum distance"
        return cls._getFarthestMinChargePointChargeDistInBlock(block, point)[1]

    @classmethod
    def _getFarthestMinChargePointInBlock(cls, block, point) -> float:
        "Get distance of the point with minimum charge and minimum distance"
        return cls._getFarthestMinChargePointChargeDistInBlock(block, point)[0]

    @classmethod
    def _getFarthestMinChargePointVecChargeDistInBlock(cls,
                                                       block,
                                                       point) -> list:
        """
        Get nearest point with minimum charge
        outputs charge, distance, and point
        """
        (dist, charge,
         point, vec) = cls._getFarNearPointInBlockWithDistMinMaxCharge(
            block=block,
            point=point,
            isMinCharge=True,
            isPoint=False,
            isMinDist=False)
        return vec, charge, dist

    @classmethod
    def _getFarthestMinChargePointVecChargeInBlock(cls, block, point) -> float:
        "Get the charge of the vector between nearest point in block and " \
            "given point"
        return cls._getFarthestMinChargePointVecChargeDistInBlock(block,
                                                                  point)[1]

    @classmethod
    def _getFarthestMinChargePointVecInBlock(cls,
                                             block,
                                             point) -> ImageLocatedVector2D:
        "Get the charge of the vector between nearest point in block and " \
            "given point"
        return cls._getFarthestMinChargePointVecChargeDistInBlock(block,
                                                                  point)[0]

    @classmethod
    def _getFarthestMaxChargePointChargeDistInBlock(cls,
                                                    block,
                                                    point) -> list:
        """
        Get nearest point with minimum charge
        outputs charge, distance, and point
        """
        (dist, charge,
         point, vec) = cls._getFarNearPointInBlockWithDistMinMaxCharge(
            block=block,
            point=point,
            isMinCharge=False,
            isPoint=True,
            isMinDist=False)
        return point, charge, dist

    @classmethod
    def _getFarthestMaxChargePointDistInBlock(cls, block, point):
        "Get distance of nearest point with maximum charge in block to point"
        return cls._getFarthestMaxChargePointDistInBlock(block, point)[2]

    @classmethod
    def _getFarthestMaxChargePointChargeInBlock(cls, block, point):
        "Get distance of nearest point with maximum charge in block to point"
        return cls._getFarthestMaxChargePointDistInBlock(block, point)[1]

    @classmethod
    def _getFarthestMaxChargePointInBlock(cls, block, point):
        "Get distance of nearest point with maximum charge in block to point"
        return cls._getFarthestMaxChargePointDistInBlock(block, point)[0]

    @classmethod
    def _getFarthestMaxChargePointVecChargeDistInBlock(cls,
                                                       block,
                                                       point) -> list:
        """
        Get nearest point with minimum charge
        outputs charge, distance, and point
        """
        (dist, charge,
         point, vec) = cls._getFarNearPointInBlockWithDistMinMaxCharge(
            block=block,
            point=point,
            isMinCharge=False,
            isPoint=False,
            isMinDist=False)
        return vec, charge, dist

    @classmethod
    def _getFarthestMaxChargePointVecChargeInBlock(cls, block, point):
        "Get charge of vec nearest point with maximum charge in block to point"
        return cls._getFarthestMaxChargePointVecDistInBlock(block, point)[1]

    @classmethod
    def _getFarthestMaxChargePointVecInBlock(cls, block, point):
        "Get distance of nearest point with maximum charge in block to point"
        return cls._getFarthestMaxChargePointVecDistInBlock(block, point)[0]

    def setBlockpoints(self):
        "Overrides the base class method"
        blockpoints = self._getPointsInBlock(block=self)
        self.blockpoints = [
            ImagePoint2D(spacePoint=p, image=self.image).setPointProperties()
            for p in blockpoints
        ]
        return None

    def setBlockCharge(self) -> None:
        "Sets the total energy charge for the block"
        counter = 0
        for p in self.blockpoints:
            counter += p.pixel_energy
        self.charge = counter
        return None

    def setBlockProperties(self) -> None:
        "Set block properties"
        self.setBlockpoints()
        self.setBlockCharge()

    def getNearestPointInBlock(self, point) -> ImagePoint2D:
        """
        Overrides base class method to return image point

        Desciption
        -----------
        Given a point the function returns the closest point to the
        given point that is inside the current block
        """
        bpoint = self._getNearestPointInBlock(block=self, point=point)
        impoint = ImagePoint2D(spacePoint=bpoint, image=self.image)
        impoint.setPointProperties()
        return impoint

    def getFarthestPointFromBlock2Point(self, point) -> ImagePoint2D:
        """
        Overrides base class method to return image point

        Desciption
        -----------
        Given a point the function returns the farthest point contained
        in the block to the given point
        """
        bpoint = self._getFarthestPointInBlock2Point(block=self, point=point)
        impoint = ImagePoint2D(spacePoint=bpoint, image=self.image)
        impoint.setPointProperties()
        return impoint

    def getClosestSide2OtherMaxDist(self, other) -> ImageLocatedVector2D:
        """
        Overrides base class method to return image locatedvector2d

        Description
        ------------
        Given a point or a segment or a located vector, function returns
        closest side contained in the block.
        Closest side is calculated using maximum distance formula, meaning that
        the distance between the side and the other is calculated using the point
        on our side that is farthest to the given other. This does not necessarily
        imply that the side itself is farthest, it is just that the distance
        is calculated using the farthest point on the side
        """
        vec = self.getCloseFarSide2OtherDistFunc(
            other, isNearest=True, justDistance=False, useMaxDistance=True)
        imvec = ImageLocatedVector2D(vec=vec, image=self.image)
        imvec.setVecProperties()
        return imvec

    def getClosestSide2OtherMinDist(self, other) -> ImageLocatedVector2D:
        """
        Overrides base class method to return image locatedvector2d

        Description
        ------------
        Given a point or a segment or a located vector, function returns
        closest side contained in the block.
        Closest side is calculated using minimum distance formula, meaning that
        the distance between the side and the other is calculated using the point
        on our side that is closest to the given other. This does not necessarily
        imply that the side itself is closest, it is just that the distance
        is calculated using the closest point on the side

        """
        vec = self.getCloseFarSide2OtherDistFunc(
            other, isNearest=True, justDistance=False, useMaxDistance=False)
        imvec = ImageLocatedVector2D(vec=vec, image=self.image)
        imvec.setVecProperties()
        return imvec

    def getFarSide2OtherMaxDist(self, other) -> ImageLocatedVector2D:
        """
        Get farthest side to other

        Description
        ------------
        Given a point or a segment or a located vector, function returns
        farthest side contained in the block.
        Farthest side is calculated using maximum distance formula, meaning that
        the distance between the side and the other is calculated using the point
        on our side that is farthest to the given other. This does not necessarily
        imply that the side itself is farthest, it is just that the distance
        is calculated using the farthest point on the side
        """
        # other should be either a point or a segment or locatedvector2d
        vec = self.getCloseFarSide2OtherDistFunc(
            other, isNearest=False, justDistance=False, useMaxDistance=True)
        imvec = ImageLocatedVector2D(vec=vec, image=self.image)
        imvec.setVecProperties()
        return imvec

    def getFarSide2OtherMinDist(self, other) -> ImageLocatedVector2D:
        """
        Get far side to other

        Description
        ------------
        Given a point or a segment or a located vector, function returns
        farthest side contained in the block.
        Farthest side is calculated using minimum distance formula, meaning that
        the distance between the side and the other is calculated using the point
        on our side that is closest to the given other. This does not necessarily
        imply that the side itself is closest, it is just that the distance
        is calculated using the closest point on the side
        """
        # other should be either a point or a segment or locatedvector2d
        vec = self.getCloseFarSide2OtherDistFunc(
            other, isNearest=False, justDistance=False, useMaxDistance=False)
        imvec = ImageLocatedVector2D(vec=vec, image=self.image)
        imvec.setVecProperties()
        return imvec

    def add2Block(self, block):
        "Overrides base class method to return image block"
        newblock = self._addBlock2Block(block1=self, block2=block)
        imblock = ImagePoint2DBlock(pointList=newblock.vertices,
                                    image=self.image)
        imblock.setBlockProperties()
        return imblock

    def foldImageBlocks(self, blocks):
        "Overrides foldrBlocks method of baseclass"
        mergeBlock = self.foldrBlocks(blocks)
        imblock = ImagePoint2DBlock(pointList=mergeBlock.vertices,
                                    image=self.image)
        imblock.setBlockProperties()
        return imblock

    def mergeBlocksWithSelf(self, blocks):
        "Overrides base class method"
        blockscp = blocks.copy()
        blockscp.append(self)
        mergedBlock = self.foldImageBlocks(blocks)
        if self in mergedBlock is not True:
            raise ValueError('List of blocks that are not mergeable'
                             ' with self block')
        else:
            return mergedBlock
