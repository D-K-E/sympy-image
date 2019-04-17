import unittest
import os
import numpy as np
import sympy.geometry

from symage.main import point
from PIL import Image

# img = Image.open('Demotic_Ostraca_Medinet_Habu_73.png')
# imarr = np.array(img)


class PointTest(unittest.TestCase):
    "Tests for point class"

    def setUp(self):
        ""
        projectdir = os.getcwd()
        testdir = os.path.join(projectdir, 'tests')
        self.assetsdir = os.path.join(testdir, 'assets')
        self.imagedir = os.path.join(self.assetsdir, 'images')
        self.jsondir = os.path.join(self.assetsdir, 'jsonfiles')
        self.numpydir = os.path.join(self.assetsdir, 'numpyfiles')

    def compareArrays(self, arr1, arr2, message):
        "Compare arrays for equality"
        result = arr1 == arr2
        result = result.all()
        self.assertTrue(result, message)

    def loadImage(self):
        impath = os.path.join(self.imagedir, 
                              "Demotic_Ostraca_Medinet_Habu_73.png")
        return Image.open(impath)

    def loadImageArray(self):
        return np.array(self.loadImage())

    def test_PointND_checkPointDirectionStatic(self):
        "Test checkPointDirection method of N dimensional point object"
        a = point.PointND((1, 2))
        b = point.PointND((3, 6))
        compval = True
        result = point.PointND._checkPointDirection(a, b)
        self.assertEqual(compval, result)

    def test_PointND_checkPointDirectionInstance(self):
        "Test checkPointDirection method of N dimensional point object"
        a = point.PointND((1, 2))
        b = point.PointND((3, 6))
        compval = True
        result = a.checkPointDirection(b)
        self.assertEqual(compval, result)

    def test_Point2D_carte2polar(self):
        "Test carte2polar method of 2 dimensional point object"
        compang = 0.9273
        compdist = 5.0
        a = point.Point2D(3, 4)
        dist, ang = a.carte2polar()
        ang = round(ang, 4)
        self.assertEqual((compang, compdist), (ang, dist))

    def test_GrayImagePoint_copy(self):
        "Test copy method of 2 dimensional image point object"
        # imcp = imarr.copy()
        impoint = point.GrayImagePoint(x=500, y=700, z=250)
        imcpypoint = impoint.copy()
        self.assertEqual((impoint.x,
                          impoint.y,
                          impoint.z),
                         (imcpypoint.x, imcpypoint.y,
                          imcpypoint.z)
                         )

    def test_GrayImagePoint_getPoint2D(self):
        "test getPoint2D method of image point object"
        # imcp = imarr.copy()
        impoint = point.GrayImagePoint(x=500, y=700, z=250)
        pointinst = point.Point2D(x=500, y=700)
        impoint2d = impoint.getPoint2D()
        self.assertEqual(impoint2d, pointinst)

    def test_GrayImagePoint_getPointRowValFromImage(self):
        imarr = self.loadImageArray()
        impoint = point.GrayImagePoint(x=20, y=120, z=180)
        imrow = impoint.getPointRowValFromImage(imarr)
        comprow = imarr[120, :]
        self.compareArrays(imrow, comprow,
                           "Image rows are not equal")

    def test_GrayImagePoint_getPointColValFromImage(self):
        imarr = self.loadImageArray()
        impoint = point.GrayImagePoint(x=20, y=120, z=180)
        imcol = impoint.getPointColValFromImage(imarr)
        compcol = imarr[:, 20]
        self.compareArrays(imcol, compcol,
                           "Image columns are not equal")
    
    def test_GrayImagePoint_getPointValFromImage(self):
        imarr = self.loadImageArray()
        impoint = point.GrayImagePoint(x=20, y=120, z=180)
        imval = impoint.getPointValFromImage(imarr)
        compval = imarr[120, 20]
        self.compareArrays(imval, compval,
                           "Pixel values are not equal")


if __name__ == '__main__':
    unittest.main()
