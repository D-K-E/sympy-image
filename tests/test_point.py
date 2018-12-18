import unittest
import os
import numpy as np
import sympy.geometry

currentdir = os.getcwd()
projectdir = os.path.join(os.pardir, currentdir)
coredir = os.path.join(projectdir, 'core')
os.chdir(coredir)

from core import point

os.chdir(currentdir)

class PointTest(unittest.TestCase):
    "Tests for point class"

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
        compang = 0.6435
        compdist = 5.0
        a = point.Point2D(3,4)
        dist, ang = a.carte2polar()
        ang = round(ang, 4)
        self.assertEqual((compang, compdist), (ang, dist))
