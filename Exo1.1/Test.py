# -*- coding: utf-8 -*-
import unittest
from funcs import trois

class TestTrois(unittest.TestCase):

    def testBasique(self):
        self.assertEqual(trois([1, 2, 3, 4, 5]), [5, 4, 3])

    def testDouble(self):
        self.assertEqual(trois([1, 2, 3, 3, 4, 4, 5]), [5, 4, 4])

    def testNegative(self):
        self.assertEqual(trois([-10, -20, -30, -40, -50]), [-10, -20, -30])

    def testMelange(self):
        self.assertEqual(trois([3, -1, 0, 10, 5]), [10, 5, 3])

    def testinferieur(self):
        self.assertEqual(trois([1, 2]), [2, 1])
        self.assertEqual(trois([7]), [7])
        self.assertEqual(trois([]), [])

    def testIdentique(self):
        self.assertEqual(trois([1, 1, 1, 1]), [1, 1, 1])

if __name__ == '__main__':
    unittest.main()
