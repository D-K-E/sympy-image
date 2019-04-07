# author: Kaan Eraslan
# license: see, LICENSE
# No warranties, use at your own risk

import numpy as np
import unittest
from PIL import Image, ImageDraw


class VectorTest(unittest.TestCase):
    "test vectors"

    def setUp(self):
        self.imwidth = 640
        self.imheight = 480
        self.blackimg = Image.new("rgb", (self.imwidth, self.imheight))
        self.whiteimg = Image.new("rgb", (self.imwidth, self.imheight),
                                  "white")
        self.point1 = (300, 150)
        self.point2 = (400, 250)
        self.point3 = (200, 100)
        self.point4 = (250, 50)

    def drawLine(self):
        "Draw "
