import unittest
from fifo import FIFO  

class Testfile(unittest.TestCase):

    def test_init(self):
        file = FIFO()
        self.assertEqual(file.taille(), 0)

    def test_ajout(self):
        file = FIFO()
        file.ajout(1)
        file.ajout(2)
        file.ajout(3)
        self.assertEqual(file.taille(), 3)

    def test_remove(self):
        file = FIFO()
        file.ajout(1)
        file.ajout(2)
        file.ajout(3)
        self.assertEqual(file.remove(), 1)  
        self.assertEqual(file.remove(), 2)
        self.assertEqual(file.taille(), 1)

    def test_defile_empty_file(self):
        file = FIFO()
        with self.assertRaises(IndexError):
            file.remove()

    def test_tete(self):
        file = FIFO()
        file.ajout(10)
        file.ajout(20)
        self.assertEqual(file.tete(), 10)  
        self.assertEqual(file.taille(), 2)

    def test_clear(self):
        file = FIFO()
        file.ajout(1)
        file.ajout(2)
        file.ajout(3)
        file.clear()
        self.assertEqual(file.taille(), 0)
        with self.assertRaises(IndexError):
            file.remove()

if __name__ == '__main__':
    unittest.main()
