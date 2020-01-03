#!/usr/bin/env python

import sys
import csv

import pandas

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 19:16:06 2018

@author: mkosmala
"""


if len(sys.argv) < 3 :
    print ("format: summarize_weather.py <input file>  <output file>")
    exit(1)

infilename = sys.argv[1]
outfilename = sys.argv[2]

# get a weather vector for each date
weather = {}

with open(infilename,'r') as infile:
    ireader = csv.reader(infile)
    
    # header
    head = next(ireader)
    
    # rows
    for row in ireader:
        wdate = row[5]
        rawdata = row[6:]

        # convert to numeric
        data = []        
        for d in rawdata:
            try:
                x = float(d)
                data.append(x)
            except:
                data.append(None)
                
        if wdate not in weather:
            weather[wdate] = []
        weather[wdate].append(data)

# calculate averages for each date
averages = []
for wdate in weather:
    
    # list of lists - convert to pandas dataframe
    ldata = weather[wdate]
        
    # calculate average over each column and save to averages
    mdata = pandas.DataFrame(ldata)   
    means = mdata.mean(axis=0, skipna=True)
    averages.append([wdate] + means.tolist())
    
# output
with open(outfilename,'w') as outfile:
    owriter = csv.writer(outfile)
    
    # header
    owriter.writerow(['date'] + head[6:])
    
    # data
    owriter.writerows(sorted(averages))

        
        
        
        
        
        
        