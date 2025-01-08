#!/usr/bin/env python3
# -*- coding: utf-8 -*-
def trois(nombre):
    if not nombre: 
        return []
    
    return sorted(nombre, reverse=True)[:3]

