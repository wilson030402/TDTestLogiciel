#!/usr/bin/env python3
# -*- coding: utf-8 -*-
def premier(nombre):
    if nombre <= 1:  
        return False
    if nombre <= 3:  
        return True
    if nombre % 2 == 0 or nombre % 3 == 0:  
        return False

    i = 5
    while i * i <= nombre:
        if nombre % i == 0 or nombre % (i + 2) == 0:
            return False
        i += 6
    return True
