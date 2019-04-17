# author: Kaan Eraslan
# license: see, LICENSE
# No warranties, use at your own risk

import numpy as np
import unittest
from PIL import Image, ImageDraw
from sympy.geometry import line, point
from sympy import sympify
from symage.main.point import Point2D
from symage.main.vector import LocatedVector2D
import pdb


class VectorTest(unittest.TestCase):
    "test vectors"

    def setUp(self):
        self.imwidth = 640
        self.imheight = 480
        self.blackimg = Image.new("RGB",
                                  (self.imwidth, self.imheight)
                                  )
        self.whiteimg = Image.new("RGB",
                                  (self.imwidth, self.imheight),
                                  "white")
        self.point1 = Point2D(x=300, y=150)
        self.point2 = Point2D(x=400, y=250)
        self.point3 = Point2D(x=200, y=100)
        self.point4 = Point2D(x=250, y=50)

    def compareArrays(self, arr1, arr2, message):
        "Compare arrays for equality"
        result = arr1 == arr2
        result = result.all()
        self.assertTrue(result, message)

    def drawLine2Image(self, 
                       image: Image,
                       point1: Point2D, 
                       point2: Point2D):
        "Draw line to image"
        imcopy = image.copy()
        draw = ImageDraw.Draw(imcopy)
        xy1 = (point1.x, point1.y)
        xy2 = (point2.x, point2.y)
        draw.line((xy1, xy2))
        return imcopy

    def drawLine2ImageFromVec(self, image, vec):
        sp = vec.spoint
        ep = vec.epoint
        return self.drawLine2Image(image, sp,ep)

    def drawVectors2Image(self, image, vecs: list):
        imcpy = image.copy()
        for vec in vecs:
            imcpy = self.drawLine2ImageFromVec(imcpy, vec)
        return imcpy

    def test_LocatedVector2D_segment_construct(self):
        "Test the located vector2d's segment constructor"
        seg = line.Segment(self.point1.point, self.point2.point)
        try:
            locvec = LocatedVector2D(segment=seg)
        except ValueError:
            locvec = None
        self.assertTrue(isinstance(locvec, LocatedVector2D),
                        "LocatedVector2D constructor is not working with "
                        "a single segment")

    def test_LocatedVector2D_points_constructor(self):
        "Test if located vector2d object can be instantiated with points"
        try:
            locvec = LocatedVector2D(initial_point=self.point1,
                                     final_point=self.point2)
        except TypeError:
            locvec = None
        self.assertTrue(isinstance(locvec, LocatedVector2D),
                        "LocatedVector2D constructor is not working with "
                        "a two points")

    def test_LocatedVector2D__getPoint2VecDistancePoint_isMinT_smethod(self):
        """
        test located vector 2d object's _getPoint2VecDistancePoint static
        method
        """
        isMin = True
        locvec = LocatedVector2D(initial_point=self.point1,
                                 final_point=self.point2)
        locvec.setLine()
        npoint, dist = LocatedVector2D._getPoint2VecDistancePoint(locvec,
                                                                  self.point3,
                                                                  isMin)
        self.assertTrue(self.point1 == npoint)

    def test_LocatedVector2D__getPoint2VecDistancePoint_isMinF_smethod(self):
        """
        test located vector 2d object's _getPoint2VecDistancePoint static
        method
        """
        isMin = False
        locvec = LocatedVector2D(initial_point=self.point1,
                                 final_point=self.point2)
        locvec.setLine()
        npoint, dist = LocatedVector2D._getPoint2VecDistancePoint(locvec,
                                                                  self.point3,
                                                                  isMin)
        self.assertTrue(self.point2 == npoint)

    def test_LocatedVector2D__getNearestPointOnVec(self):
        """
        test located vector 2d object's _getPoint2VecDistancePoint static
        method
        """
        locvec = LocatedVector2D(initial_point=self.point1,
                                 final_point=self.point2)
        locvec.setLine()
        npoint = LocatedVector2D._getNearestPointOnVec(locvec, self.point3)
        self.assertTrue(self.point1 == npoint)

    def test_LocatedVector2D__getFarthestPointOnVec(self):
        """
        test located vector 2d object's _getPoint2VecDistancePoint static
        method
        """
        locvec = LocatedVector2D(initial_point=self.point1,
                                 final_point=self.point2)
        locvec.setLine()
        npoint = LocatedVector2D._getFarthestPointOnVec(locvec, self.point3)
        self.assertTrue(self.point2 == npoint)

    def test_LocatedVector2D__getStraightDistance(self):
        dist = LocatedVector2D._getStraightDistance(self.point1,
                                                    self.point2)
        self.assertTrue(dist == self.point1.point.distance(self.point2.point))

    def test_LocatedVector2D__getVec2VecDistancePointVecIsMinTisMinFuncT_smethod(self):
        vec1 = LocatedVector2D(self.point1, self.point2)
        vec1.setVecProperties()
        vec2 = LocatedVector2D(self.point3, self.point4)
        vec2.setVecProperties()
        isMin = True
        (nspoint, nepoint,
         svec, dist) = LocatedVector2D._getVec2VecDistancePointVec(
            vec1, vec2, isMin, isMinFuncs=True)
        compointe = Point2D(x=225, y=75)
        compoints = Point2D(x=300, y=150)
        compvec = LocatedVector2D(initial_point=compoints, 
                                  final_point=compointe)
        compdist_expr = "75*sqrt(2)"
        compdist = sympify(compdist_expr)
        self.assertEqual(nspoint, compoints, "Starting point is not true")
        self.assertEqual(nepoint, compointe, "Ending point is not true")
        self.assertEqual(svec, compvec, "Computed vector is not true")
        self.assertEqual(dist, compdist, "Computed distance is not true")

    def test_LocatedVector2D__getVec2VecDistancePointVecIsMinFisMinFuncT_smethod(self):
        vec1 = LocatedVector2D(self.point1, self.point2)
        vec1.setVecProperties()
        vec2 = LocatedVector2D(self.point3, self.point4)
        vec2.setVecProperties()
        isMin = False
        (nspoint, nepoint,
         svec, dist) = LocatedVector2D._getVec2VecDistancePointVec(
            vec1, vec2, isMin, isMinFuncs=True)
        compointe = Point2D(x=225, y=75)
        compoints = Point2D(x=400, y=250)
        compvec = LocatedVector2D(initial_point=compoints,
                                  final_point=compointe)
        compdist_expr = "175*sqrt(2)"
        compdist = sympify(compdist_expr)
        # pdb.set_trace()
        self.assertEqual(nspoint, compoints, "Starting point is not true")
        self.assertEqual(nepoint, compointe, "Ending point is not true")
        self.assertEqual(svec, compvec, "Computed vector is not true")
        self.assertEqual(dist, compdist, "Computed distance is not true")

    def test_LocatedVector2D__getVec2VecDistancePointVecIsMinTisMinFuncF_smethod(self):
        vec1 = LocatedVector2D(self.point1, self.point2)
        vec1.setVecProperties()
        vec2 = LocatedVector2D(self.point3, self.point4)
        vec2.setVecProperties()
        isMin = True
        (nspoint, nepoint,
         svec, dist) = LocatedVector2D._getVec2VecDistancePointVec(
            vec1, vec2, isMin, isMinFuncs=False)
        compointe = Point2D(x=200, y=100)
        compoints = Point2D(x=300, y=150)
        compvec = LocatedVector2D(initial_point=compoints, 
                                  final_point=compointe)
        compdist_expr = "50*sqrt(5)"
        compdist = sympify(compdist_expr)
        # foo1 = self.drawLine2ImageFromVec(self.blackimg, vec1)
        # foo2 = self.drawLine2ImageFromVec(self.blackimg, vec2)
        # foo3 = self.drawLine2ImageFromVec(self.blackimg, svec)
        # foo4 = self.drawVectors2Image(self.blackimg, [vec1, vec2, svec])
        # pdb.set_trace()
        self.assertEqual(nspoint, compoints, "Starting point is not true")
        self.assertEqual(nepoint, compointe, "Ending point is not true")
        self.assertEqual(svec, compvec, "Computed vector is not true")
        self.assertEqual(dist, compdist, "Computed distance is not true")

    def test_LocatedVector2D__getVec2VecDistancePointVecIsMinFisMinFuncF_smethod(self):
        vec1 = LocatedVector2D(self.point1, self.point2)
        vec1.setVecProperties()
        vec2 = LocatedVector2D(self.point3, self.point4)
        vec2.setVecProperties()
        isMin = False
        (nspoint, nepoint,
         svec, dist) = LocatedVector2D._getVec2VecDistancePointVec(
            vec1, vec2, isMin, isMinFuncs=False)
        compointe = Point2D(x=200, y=100)
        compoints = Point2D(x=400, y=250)
        compvec = LocatedVector2D(initial_point=compoints,
                                  final_point=compointe)
        compdist_expr = "250"
        compdist = sympify(compdist_expr)
        # foo1 = self.drawLine2ImageFromVec(self.blackimg, vec1)
        # foo2 = self.drawLine2ImageFromVec(self.blackimg, vec2)
        # foo3 = self.drawLine2ImageFromVec(self.blackimg, svec)
        # foo4 = self.drawVectors2Image(self.blackimg, [vec1, vec2, svec])
        # pdb.set_trace()
        self.assertEqual(nspoint, compoints, "Starting point is not true")
        self.assertEqual(nepoint, compointe, "Ending point is not true")
        self.assertEqual(svec, compvec, "Computed vector is not true")
        self.assertEqual(dist, compdist, "Computed distance is not true")




if __name__ == "__main__":
    unittest.main()
