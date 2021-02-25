# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:25:03 2020

@author: p1857809
"""
import numpy as np
import matplotlib.pyplot as plt

A =np.asarray([[('Druck', 'Mittelwert'), ('Austrag', 'Mittelwert')],
         [('Duesentemperatur', 'Mittelwert'), ('Bauraumtemperatur', 'Mittelwert')]])
B = [('Druck', 'Mittelwert'), ('Austrag', 'Mittelwert')]

print(A.shape)
for a in A:
    print(a)
    
for b in B:
    print(type(b)==tuple)