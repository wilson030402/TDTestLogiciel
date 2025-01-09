import unittest
from lifo import LIFOStack 

class TestLIFOStack(unittest.TestCase):

    def test_initialization(self):
        stack = LIFOStack()
        self.assertEqual(stack.size(), 0)  

    def test_push(self):
        stack = LIFOStack()
        stack.push(1)
        stack.push(2)
        stack.push(3)
        self.assertEqual(stack.size(), 3)  
        
    def test_pop(self):
        stack = LIFOStack()
        stack.push(1)
        stack.push(2)
        stack.push(3)
        self.assertEqual(stack.pop(), 3) 
        self.assertEqual(stack.pop(), 2)
        self.assertEqual(stack.size(), 1)  
        
    def test_pop_empty_stack(self):
        stack = LIFOStack()
        with self.assertRaises(IndexError):  
            stack.pop()

    def test_peek(self):
        stack = LIFOStack()
        stack.push(10)
        stack.push(20)
        self.assertEqual(stack.peek(), 20)  
        self.assertEqual(stack.size(), 2)  

    def test_clear(self):
        stack = LIFOStack()
        stack.push(1)
        stack.push(2)
        stack.push(3)
        stack.clear()
        self.assertEqual(stack.size(), 0)  
        with self.assertRaises(IndexError):
            stack.pop()

if __name__ == '__main__':
    unittest.main()
