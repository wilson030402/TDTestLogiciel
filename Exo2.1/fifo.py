#!/usr/bin/env python3
# -*- coding: utf-8 -*-
class FIFO:
    def __init__(self):
        self.file = []

    def ajout(self, item):
        self.file.append(item)

    def remove(self):
        if self.taille() == 0:
            raise IndexError("Erreur")
        return self.file.pop(0)

    def tete(self):
        if self.taille() == 0:
            raise IndexError("Erreur")
        return self.file[0]

    def taille(self):
        return len(self.file)

    def clear(self):
        self.file = []

