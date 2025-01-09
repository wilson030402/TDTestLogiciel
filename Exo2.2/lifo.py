#!/usr/bin/env python3
# -*- coding: utf-8 -*-
class LIFOStack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if self.size() == 0:
            raise IndexError("Erruer")
        return self.stack.pop()

    def peek(self):
        if self.size() == 0:
            raise IndexError("Erreur")
        return self.stack[-1]

    def size(self):
        return len(self.stack)

    def clear(self):
        # Vide la pile
        self.stack = []

