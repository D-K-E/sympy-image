# author: Kaan Eraslan
# license: see, LICENSE
# No warranties, use at your own risk

import numpy as np
import unittest
from PIL import Image, ImageDraw, ImageOps
from sympy.geometry import line, point
from sympy import sympify
from symage.main.point import Point2D, GrayImagePoint
from symage.main.vector import LocatedVector2D, GrayImageLocatedVector2D
import pdb


class VectorTest(unittest.TestCase):
    "test vectors"

    def setUp(self):
        self.imwidth = 640
        self.imheight = 480
        self.blackimg = Image.new("RGB",
                                  (self.imwidth, self.imheight)
                                  )
        self.blackimarr = np.array(self.blackimg)
        self.grayscale_img = ImageOps.grayscale(self.blackimg)
        self.grayimarr = np.array(self.grayscale_img)
        self.whiteimg = Image.new("RGB",
                                  (self.imwidth, self.imheight),
                                  "white")
        self.whiteimarr = np.array(self.whiteimg)
        self.point1 = Point2D(x=300, y=150)
        self.point2 = Point2D(x=400, y=250)
        self.point3 = Point2D(x=200, y=100)
        self.point4 = Point2D(x=250, y=50)
        self.img_point1 = GrayImagePoint(x=300, y=150, z=0)
        self.img_point1.setZvalFromImage(self.grayimarr)
        self.img_point2 = GrayImagePoint(x=400, y=250, z=0)
        self.img_point2.setZvalFromImage(self.grayimarr)
        self.img_point3 = GrayImagePoint(x=200, y=100, z=0)
        self.img_point3.setZvalFromImage(self.grayimarr)
        self.img_point4 = GrayImagePoint(x=250, y=50, z=0)
        self.img_point4.setZvalFromImage(self.grayimarr)

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

    def test_LocatedVector2D__getVec2VecDistancePointVecIsMinTisMinFuncT_smethod(
            self):
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

    def test_LocatedVector2D__getVec2VecDistancePointVecIsMinFisMinFuncT_smethod(
            self):
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

    def test_LocatedVector2D__getVec2VecDistancePointVecIsMinTisMinFuncF_smethod(
            self):
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

    def test_LocatedVector2D__getVec2VecDistancePointVecIsMinFisMinFuncF_smethod(
            self):
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

    def test_GrayImageLocatedVector2D_constructor_points(self):
        spoint = self.img_point1
        epoint = self.img_point2
        try:
            imvec = GrayImageLocatedVector2D(
                image=np.array(self.blackimg),
                initial_point=spoint,
                final_point=epoint
            )
        except TypeError:
            imvec = None
        checkval = isinstance(imvec, GrayImageLocatedVector2D)
        self.assertTrue(
            checkval,
            "GrayImageLocatedVector2D, can not be constructed from points"
        )

    def test_GrayImageLocatedVector2D_constructor_vec(self):
        vec1 = LocatedVector2D(self.point1, self.point2)
        try:
            imvec = GrayImageLocatedVector2D(
                image=np.array(self.blackimg),
                vec=vec1
            )

        except TypeError:
            imvec = None
        checkval = isinstance(imvec, GrayImageLocatedVector2D)
        self.assertTrue(
            checkval,
            "GrayImageLocatedVector2D, can not be constructed from vector"
        )

    def test_GrayImageLocatedVector2D_constructor_segment(self):
        seg = line.Segment(self.point1.point, self.point2.point)
        try:
            imvec = GrayImageLocatedVector2D(
                image=np.array(self.blackimg),
                segment=seg
            )

        except TypeError:
            imvec = None
        checkval = isinstance(imvec, GrayImageLocatedVector2D)
        self.assertTrue(
            checkval,
            "GrayImageLocatedVector2D, can not be constructed from segment"
        )

    def test_GrayImageLocatedVector2D_getDistanceChargeVariables(self):
        min_dist1 = True
        min_charge1 = None
        min_dist2 = False
        min_charge2 = None
        min_dist3 = None
        min_charge3 = True
        min_dist4 = None
        min_charge4 = False
        min_dist5 = True
        min_charge5 = True
        min_dist6 = True
        min_charge6 = False
        min_dist7 = False
        min_charge7 = True
        min_dist8 = False
        min_charge8 = False

        # compare values
        compdist1 = float('inf')
        compcharge1 = None
        compdist2 = float('-inf')
        compcharge2 = None
        compdist3 = None
        compcharge3 = float('inf')
        compdist4 = None
        compcharge4 = float('-inf')
        compdist5 = float('inf')
        compcharge5 = float('inf')
        compdist6 = float('inf')
        compcharge6 = float('-inf')
        compdist7 = float('-inf')
        compcharge7 = float('inf')
        compdist8 = float('-inf')
        compcharge8 = float('-inf')

        imvec = GrayImageLocatedVector2D(image=self.blackimarr,
                                         initial_point=self.img_point1,
                                         final_point=self.img_point2)

        dist1, charge1 = imvec.getDistanceChargeVariables(
            minDist=min_dist1,
            minCharge=min_charge1)

        dist2, charge2 = imvec.getDistanceChargeVariables(
            minDist=min_dist2,
            minCharge=min_charge2)

        dist3, charge3 = imvec.getDistanceChargeVariables(
            minDist=min_dist3,
            minCharge=min_charge3)

        dist4, charge4 = imvec.getDistanceChargeVariables(
            minDist=min_dist4,
            minCharge=min_charge4)

        dist5, charge5 = imvec.getDistanceChargeVariables(
            minDist=min_dist5,
            minCharge=min_charge5)

        dist6, charge6 = imvec.getDistanceChargeVariables(
            minDist=min_dist6,
            minCharge=min_charge6)

        dist7, charge7 = imvec.getDistanceChargeVariables(
            minDist=min_dist7,
            minCharge=min_charge7)

        dist8, charge8 = imvec.getDistanceChargeVariables(
            minDist=min_dist8,
            minCharge=min_charge8)

        self.assertEqual(dist1, compdist1)
        self.assertEqual(charge1, compcharge1)
        self.assertEqual(dist2, compdist2)
        self.assertEqual(charge2, compcharge2)
        self.assertEqual(dist3, compdist3)
        self.assertEqual(charge3, compcharge3)
        self.assertEqual(dist4, compdist4)
        self.assertEqual(charge4, compcharge4)
        self.assertEqual(dist5, compdist5)
        self.assertEqual(charge5, compcharge5)
        self.assertEqual(dist6, compdist6)
        self.assertEqual(charge6, compcharge6)
        self.assertEqual(dist7, compdist7)
        self.assertEqual(charge7, compcharge7)
        self.assertEqual(dist8, compdist8)
        self.assertEqual(charge8, compcharge8)
    
    def test_GrayImageLocatedVector2D_getConditionDistanceCharge_1(self):
        imvec = GrayImageLocatedVector2D(image=self.blackimarr,
                                         initial_point=self.img_point1,
                                         final_point=self.img_point2)
        min_dist1 = True
        min_charge1 = None
        min_dist2 = False
        min_charge2 = None
        min_dist3 = None
        min_charge3 = True
        min_dist4 = None
        min_charge4 = False
        min_dist5 = True
        min_charge5 = True
        min_dist6 = True
        min_charge6 = False
        min_dist7 = False
        min_charge7 = True
        min_dist8 = False
        min_charge8 = False

        # compare values
        parent_dist1 = float('inf')
        parent_charge1 = None
        parent_dist2 = float('-inf')
        parent_charge2 = None
        parent_dist3 = None
        parent_charge3 = float('inf')
        parent_dist4 = None
        parent_charge4 = float('-inf')
        parent_dist5 = float('inf')
        parent_charge5 = float('inf')
        parent_dist6 = float('inf')
        parent_charge6 = float('-inf')
        parent_dist7 = float('-inf')
        parent_charge7 = float('inf')
        parent_dist8 = float('-inf')
        parent_charge8 = float('-inf')

        local_distance = 0
        local_charge = 0

        checkval1 = imvec.getConditionDistanceCharge(
            minDist=min_dist1,
            minCharge=min_charge1,
            distanceParentScope=parent_dist1,
            distanceLocalScope=local_distance,
            chargeParentScope=parent_charge1,
            chargeLocalScope=local_charge
        )

        checkval2 = imvec.getConditionDistanceCharge(
            minDist=min_dist2,
            minCharge=min_charge2,
            distanceParentScope=parent_dist2,
            distanceLocalScope=local_distance,
            chargeParentScope=parent_charge2,
            chargeLocalScope=local_charge
        )

        checkval3 = imvec.getConditionDistanceCharge(
            minDist=min_dist3,
            minCharge=min_charge3,
            distanceParentScope=parent_dist3,
            distanceLocalScope=local_distance,
            chargeParentScope=parent_charge3,
            chargeLocalScope=local_charge
        )

        checkval4 = imvec.getConditionDistanceCharge(
            minDist=min_dist4,
            minCharge=min_charge4,
            distanceParentScope=parent_dist4,
            distanceLocalScope=local_distance,
            chargeParentScope=parent_charge4,
            chargeLocalScope=local_charge
        )

        checkval5 = imvec.getConditionDistanceCharge(
            minDist=min_dist5,
            minCharge=min_charge5,
            distanceParentScope=parent_dist5,
            distanceLocalScope=local_distance,
            chargeParentScope=parent_charge5,
            chargeLocalScope=local_charge
        )

        checkval6 = imvec.getConditionDistanceCharge(
            minDist=min_dist6,
            minCharge=min_charge6,
            distanceParentScope=parent_dist6,
            distanceLocalScope=local_distance,
            chargeParentScope=parent_charge6,
            chargeLocalScope=local_charge
        )

        checkval7 = imvec.getConditionDistanceCharge(
            minDist=min_dist7,
            minCharge=min_charge7,
            distanceParentScope=parent_dist7,
            distanceLocalScope=local_distance,
            chargeParentScope=parent_charge7,
            chargeLocalScope=local_charge
        )

        checkval8 = imvec.getConditionDistanceCharge(
            minDist=min_dist8,
            minCharge=min_charge8,
            distanceParentScope=parent_dist8,
            distanceLocalScope=local_distance,
            chargeParentScope=parent_charge8,
            chargeLocalScope=local_charge
        )

        self.assertTrue(checkval1)
        self.assertTrue(checkval2)
        self.assertTrue(checkval3)
        self.assertTrue(checkval4)
        self.assertTrue(checkval5)
        self.assertTrue(checkval6)
        self.assertTrue(checkval7)
        self.assertTrue(checkval8)









if __name__ == "__main__":
    unittest.main()
