#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from funcs import premier  

class TestIsPrime(unittest.TestCase):

    def testPremierP(self):
        self.assertTrue(premier(2))
        self.assertTrue(premier(3))
        self.assertTrue(premier(5))
        self.assertTrue(premier(7))

    def testNonPremierP(self):
        self.assertFalse(premier(4))
        self.assertFalse(premier(6))
        self.assertFalse(premier(9))

    def testPremierG(self):
        self.assertTrue(premier(101))
        self.assertTrue(premier(103))
        self.assertTrue(premier(107))
        self.assertTrue(premier(109))

    def testNonPremierG(self):
        self.assertFalse(premier(100))
        self.assertFalse(premier(102))
        self.assertFalse(premier(104))
        self.assertFalse(premier(105))

    def testNegatif(self):
        self.assertFalse(premier(-1))
        self.assertFalse(premier(-2))
        self.assertFalse(premier(-17))

    def testCasLimite(self):
        self.assertFalse(premier(0))
        self.assertFalse(premier(1))

if __name__ == '__main__':
    unittest.main()
