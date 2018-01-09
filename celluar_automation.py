#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-01-02 11:17:27
# @Author  : Your Name

"""a celluar automation algorithem: the evolution is determined by the initial
state, needing no inputs.

Any live cell with fewer than two live neighbours dies, as if by needs caused by underpopulation.
Any live cell with more than three live neighbours dies, as if by overcrowding.
Any live cell with two or three live neighbours lives, unchanged, to the next generation.
Any dead cell with exactly three live neighbours becomes a live cell.
"""
import numpy as np
import os
import sys

"""taking the border into account,
count the neighbours
"""

def compute_neighbours(Z):
    shape = (len(Z), len(Z[0]))
    N = [[0,]*(shape[0]) for i in range(shape[1])]
    for x in range(1, shape[0]-1):
        for y in range(1, shape[1]-1):
            N[x][y] = Z[x-1][y-1]+Z[x][y-1]+Z[x+1][y-1] \
                    + Z[x-1][y]            +Z[x+1][y]   \
                    + Z[x-1][y+1]+Z[x][y+1]+Z[x+1][y+1]

    return N
#############################################################
"""
count the number of neighbours for each internal cell"""
def iterate(Z):
    N=compute_neighbours(Z)
    shape = (len(Z), len(Z[0]))
    for x in range(1, shape[0]-1):
        for y in range(1, shape[1]):
            if Z[x][y] == 1 and (N[x][y] < 2 or N[x][y] > 3):
                Z[x][y] = 0
            elif Z[x][y] == 0 and Z[x][y]==3:
                Z[x][y] = 1
    return Z

###############################################

"""
give the value of the running times and print the output"""

def output(Z, times):
    for i in range(0, int(times)-1): print (i+1, iterate(Z))

if __name__ == "__main__":
    times=sys.argv[1]


Z = [[0,0,0,0,0,0],
     [0,0,0,1,0,0],
     [0,1,0,1,0,0],
     [0,0,1,1,0,0],
     [0,0,0,0,0,0],]

output(Z, times)




