#!/usr/bin/env python


import sys
import csv
from random import randint

# -*- coding: utf-8 -*-
"""
Create training and testing data sets by DATE


Created on Wed Jan 17 15:06:13 2018

@author: mkosmala
"""


if len(sys.argv) < 3 :
    print ("format: split_data_test_train.py <combined file> <output file>")
    exit(1)

infilename = sys.argv[1]
outfilename = sys.argv[2]

alldates = []
# get the ride data
with open(infilename,'r') as infile:
    ireader = csv.reader(infile)
    
    # header
    next(ireader)
    
    # get all the dates
    for row in ireader:
        alldates.append(row[0])
        
# now split them into training (0) and testing (1) sets
# 80%, 20%
n = len(alldates)
n_test = int(0.2*n)
testdates = []
for i in range(0,n_test):
    # grab a random date from all the (remaining) dates
    rnum = randint(0,len(alldates)-1)
    testdates.append(alldates.pop(rnum))

output = []
for d in alldates:
    output.append([d,0])
for d in testdates:
    output.append([d,1])

# output
with open(outfilename,'w') as outfile:
    owriter = csv.writer(outfile)
    owriter.writerows(sorted(output))