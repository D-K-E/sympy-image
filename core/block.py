# Regroups objects with respect to point blocks
# Author: Kaan Eraslan
# Licensing: see, LICENSE

# Packages

import numpy as np
from sympy.geometry import polygon

from vector import ImageLocatedVector2D, LocatedVector2D
from point import ImagePoint2D, Point2D


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
    def _getNearestPointInBlock(block, point: Point2D):
        "Get the point in block that is closest to given point "
        if point in block:
            return point
        distance = block.distance(point)
        points = block.vertices
        for p in points:
            dist = p.distance(point)
            if dist == distance:
                return p

    def getNearestPointInBlock(self, point):
        "Wrapper method for class instance"
        return self._getNearestPointInBlock(block=self, point=point)

    @staticmethod
    def _getFarthestPointInBlockWithDist(block, point: Point2D):
        "Get farthest point in block to the given point"
        dist = float('-inf')
        minpoint = None
        for p in block.vertices:
            distance = p.distance(point)
            if distance > dist:
                dist = distance
                minpoint = p

        return dist, minpoint

    @classmethod
    def _getFarthestDistanceInBlock2Point(cls, block, point: Point2D) -> int:
        "Get farthest distance from a point in block to the given point"
        return cls._getFarthestPointInBlockWithDist(block, point)[0]

    @classmethod
    def _getFarthestPointInBlock2Point(cls, block, point) -> Point2D:
        "Get farthest distance from a point in block to the given point"
        return cls._getFarthestPointInBlockWithDist(block, point)[1]

    def getFarthestDistanceFromBlock2Point(self, point: Point2D) -> float:
        "Get farthest distance from block to the given point"
        return self._getFarthestDistanceInBlock2Point(block=self, point=point)

    def getFarthestPointFromBlock2Point(self, point: Point2D) -> Point2D:
        "Get farthest distance from block to the given point"
        return self._getFarthestPointInBlock2Point(block=self, point=point)

    @staticmethod
    def _getCloseOrFarSideAndDistanceInBlock2Point(block, point: Point2D,
                                                   isNear: bool):
        "Get nearest side in block to the given point"
        sides = block.sides
        if isNear is True:
            dist = float('inf')
        else:
            dist = float('-inf')
        sideInBlock = None
        for side in sides:
            # side is a Segment
            distance = side.distance(point)
            condition = None
            if isNear is True:
                condition = distance <= dist
            else:
                condition = distance > dist
            if condition:
                dist = distance
                sideInBlock = side

        sideInBlock = LocatedVector2D(segment=sideInBlock)
        return dist, sideInBlock

    @classmethod
    def _getClosestSideAndDistanceInBlock2Point(cls, block, point):
        "Get nearest side and distance in block to given point"
        return cls._getCloseOrFarSideAndDistanceInBlock2Point(
            block, point, isNear=True)

    @classmethod
    def _getClosestSideInBlock2Point(cls, block, point) -> LocatedVector2D:
        "get nearest side in block to given point"
        return cls._getClosestSideAndDistanceInBlock2Point(block, point)[1]

    @classmethod
    def _getNearestDist2SideInBlock2Point(cls, block, point) -> float:
        "get nearest side in block to given point"
        return cls._getClosestSideAndDistanceInBlock2Point(block, point)[0]

    @classmethod
    def _getFarSideAndDistanceInBlock2Point(cls, block, point):
        "Get nearest side and distance in block to given point"
        return cls._getCloseOrFarSideAndDistanceInBlock2Point(
            block, point, isNear=False)

    @classmethod
    def _getFarSideInBlock2Point(cls, block, point) -> LocatedVector2D:
        "get nearest side in block to given point"
        return cls._getFarSideAndDistanceInBlock2Point(block, point)[1]

    @classmethod
    def _getFarthestDist2SideInBlock2Point(cls, block, point):
        "get nearest side in block to given point"
        return cls._getFarSideAndDistanceInBlock2Point(block, point)[0]

    @classmethod
    def _getCloseOrFarSideAndDistanceInBlock2Vec(
            cls,
            block,
            vec: LocatedVector2D,
            isMin: bool,  # get minimum side or distance overall
            isMaxFuncs: bool  # use maximum or minimum distance function
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
            " using maximum distance function"
        return cls._getCloseOrFarSideAndDistanceInBlock2Vec(
            block=block, vec=vec, isMin=True, isMaxFuncs=True)

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
        return cls._getCloseOrFarSideAndDistanceInBlock2Vec(
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
        return cls._getCloseOrFarSideAndDistanceInBlock2Vec(
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
        return cls._getCloseOrFarSideAndDistanceInBlock2Vec(
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
                    return self._getClosestSideInBlock2Point(
                        block=self, point=other)
                else:
                    return self._getNearestDist2SideInBlock2Point(
                        block=self, point=other)
            else:
                return self.getClosestSideOrDistanceWithDistFunc(
                    other, useMaxDistance, justDistance)
        else:
            if isinstance(other, Point2D):
                return self._getFarSideInBlock2Point(block=self, point=other)
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

    def setBlockpoints(self):
        "Overrides the base class method"
        blockpoints = self._getPointsInBlock(block=self)
        self.blockpoints = [
            ImagePoint2D(spacePoint=p, image=self.image).setPixelValueEnergy()
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
