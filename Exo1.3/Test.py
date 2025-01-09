import unittest
from funcs import arithmetic  

class TestArithmetic(unittest.TestCase):

    def testArithmeticPositive(self):
        self.assertTrue(arithmetic([1, 3, 5, 7, 9]))  # Pas de 2
        self.assertTrue(arithmetic([10, 20, 30, 40, 50]))  # Pas de 10

    def testArithmeticNegative(self):
        self.assertTrue(arithmetic([10, 8, 6, 4, 2]))  # Pas de -2
        self.assertTrue(arithmetic([100, 50, 0, -50, -100]))  # Pas de -50

    def testArithmeticSV(self):
        self.assertTrue(arithmetic([]))  # Une liste vide est considérée comme une suite
        self.assertTrue(arithmetic([42]))  # Une seule valeur est considérée comme une suite

    def testNonArithmetic(self):
        self.assertFalse(arithmetic([1, 3, 6, 10, 15]))  # Pas variable
        self.assertFalse(arithmetic([1, 2, 4, 8, 16]))  # Suite géométrique
        self.assertFalse(arithmetic([5, 1, 3, 7]))  # Ordre non respecté

    def testArithmeticU(self):
        self.assertTrue(arithmetic([7, 7, 7, 7]))  # Pas de 0
        self.assertFalse(arithmetic([7, 7, 7, 8]))  # Pas de 0 sauf dernier élément différent

if __name__ == '__main__':
    unittest.main()
