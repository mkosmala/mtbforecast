#!/usr/bin/env python

import sys
import csv

import numpy as np
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler



# -*- coding: utf-8 -*-
"""
Takes an activity_summary file and a weather_summary file.

Created on Mon Jan 15 14:07:26 2018

@author: mkosmala
"""

# date range iterator
def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

#
# START
# 

if len(sys.argv) < 4 :
    print ("format: combine_input_data.py <activity file> <weather file> <output file>")
    exit(1)

actfilename = sys.argv[1]
weafilename = sys.argv[2]
outfilename = sys.argv[3]

# get the ride data
with open(actfilename,'r') as actfile:
    areader = csv.reader(actfile)
    
    # remove header
    next(areader)

    # save the ride info
    allrides = list(areader)
    
# for each rider, compute the first and last ride
# this will be the "active rider" time period
riders = {}
# we also want to normalize (z-score) their speed and distance
# these dictionaries are keyed by rider name and then a list of values
distances = {}
speeds = {}
for row in allrides:
    
    rider = row[0]
    mtn_bike = int(row[3])
    local = int(row[4])
    dist = float(row[5])
    spd = float(row[7])
    dt = row[8]

    # use only local mountain bike rides
    if mtn_bike==1 and local==1:
        
        # dates
        rdate = date(int(dt[0:4]),int(dt[5:7]),int(dt[8:10]))
        if rider not in riders:
            riders[rider] = [rdate,rdate]
        if rdate < riders[rider][0]:
            riders[rider][0] = rdate
        if rdate > riders[rider][1]:
            riders[rider][1] = rdate
        
        # distances and speeds
        if rider not in distances:
            distances[rider] = []
        if rider not in speeds:
            speeds[rider] = []
        distances[rider].append(dist)
        speeds[rider].append(spd)
    

# get the very first date
min_date = date(2020,1,1)
for r in riders:
    if riders[r][0] < min_date:
        min_date = riders[r][0]

# max date specified by when people started giving me data
#max_date = date(2018,1,10)

# max_date will be Dec 31, 2017 for all riders
# assumption that they all used Strava until this point
# since they shared their data with me
max_date = date(2017,12,31)

# create an empty dictionary of all the dates
# values are a pair of number of possible riders, number of actual riders
perday = {}
rdate = min_date
oneday = timedelta(days=1)
while rdate <= max_date:
    perday[rdate] = [0,0]
    rdate += oneday

# for each rider, normalize the distances and speeds
rider_distances_params = {}
rider_speeds_params = {}
for rider in distances:
    vals = distances[rider]   
    scaler = StandardScaler()
    scaler.fit(np.array(vals).reshape(-1,1))
    rider_distances_params[rider] = scaler
for rider in speeds:
    vals = speeds[rider]
    scaler = StandardScaler()
    scaler.fit(np.array(vals).reshape(-1,1))
    rider_speeds_params[rider] = scaler    
    
    
    
# count number of possible riders each day
for r in riders:
    rdate = riders[r][0]
    #while rdate <= riders[r][1] and rdate <= max_date:
    while rdate <= max_date:   # keep them all active until the end
        perday[rdate][0] += 1
        rdate += oneday

# these are keyed by date, values are a list 
dist_per_day = {}
speed_per_day = {}
# go through the ride data again and count number of riders each day
# also get the average distance and speed for that day
for row in allrides:   
    rider = row[0]
    mtn_bike = int(row[3])
    local = int(row[4])
    dist = float(row[5])
    spd = float(row[7])
    dt = row[8]

    # use only local mountain bike rides
    if mtn_bike==1 and local==1:    
        
        # date
        rdate = date(int(dt[0:4]),int(dt[5:7]),int(dt[8:10]))
        if rdate <= max_date:
            perday[rdate][1] += 1
    
        # distance
        scaler = rider_distances_params[rider]
        dist_z = scaler.transform(np.array(dist).reshape(1,-1))[0]
        if rdate not in dist_per_day:
            dist_per_day[rdate] = []
        dist_per_day[rdate].append(dist_z)
        
        # speed
        scaler = rider_speeds_params[rider]
        speed_z = scaler.transform(np.array(spd).reshape(1,-1))[0]
        if rdate not in speed_per_day:
            speed_per_day[rdate] = []
        speed_per_day[rdate].append(speed_z)


# get the weather data
weather = {}
with open(weafilename,'r') as weafile:
    ireader = csv.reader(weafile)
    
    # header
    next(ireader)
    
    for row in ireader:
        dt = row[0]
        wdate = date(int(dt[0:4]),int(dt[5:7]),int(dt[8:10]))
        precip = float(row[8])
        snowdepth = float(row[13])
        tempave = float(row[16])
        tempmax = float(row[17])
        tempmin = float(row[18])
        timesun = float(row[20])
        peakwind = float(row[27])
        
        weather[wdate] = [precip,snowdepth,tempave,tempmax,
                          tempmin,timesun,peakwind]
     
# first and last dates
weatherdates = sorted(list(weather.keys()))
firstdate = weatherdates[0]
lastdate = weatherdates[-1]
     
# calculate the previous-week stats
# - highest windspeed (indicative of recent windstorm)
# - total sunshine (indicates evaporation)
# - average temp (indicates evaporation)
# - total precipitation (indicates wet ground)
oneweek = timedelta(days=7)
for i in daterange(firstdate,lastdate-oneweek+oneday):
    highwind = 0
    sunsum = 0
    tempsum = 0
    precipsum = 0
    for j in daterange(i,i+oneweek):
        sunsum += weather[j][5]
        tempsum += weather[j][2]
        precipsum += weather[j][0]
        if weather[j][6] > highwind:
            highwind = weather[j][6]
    tempave = tempsum / 7.0
    targetday = i + oneweek
    tempdiff = weather[targetday][2] - tempave
    weather[targetday] = weather[targetday] + [highwind,sunsum,tempave,precipsum,tempdiff]


# create the per ride dataset
# keyed by rider then date
ride_dict = {}
for row in allrides:
    rider = row[0]
    mtn_bike = int(row[3])
    local = int(row[4])
    dist = float(row[5])
    spd = float(row[7])
    dt = row[8]

    # use only local mountain bike rides
    if mtn_bike==1 and local==1:    

        d = date(int(dt[0:4]),int(dt[5:7]),int(dt[8:10]))
    
        # only do 2015-2017        
        if (d.year >= 2015 and d.year <= 2017):
            n = perday[d]
            frac = 1.0*n[1]/n[0]
            doy = d.timetuple().tm_yday
            dayweek = d.weekday()
            weekend = 0
            if dayweek == 5 or dayweek == 6:
                weekend = 1
       
            if rider not in ride_dict:
                ride_dict[rider] = {}
       
            ride_dict[rider][d] = [rider,d,1,d.year,doy,weekend,n[0],n[1],frac,
                                   dist,spd]
    
# and fill in when rides didn't happen
for rider in riders:
    first,last = riders[rider]
    #for d in daterange(first,last+timedelta(1)):
    for d in daterange(first,max_date+timedelta(1)):
        if (d.year >= 2015 and d.year <= 2017):
            n = perday[d]
            frac = 1.0*n[1]/n[0]
            doy = d.timetuple().tm_yday
            dayweek = d.weekday()
            weekend = 0
            if dayweek == 5 or dayweek == 6:
                weekend = 1

            # if we don't have a ride this day, note it       
            if d not in ride_dict[rider]:                
                ride_dict[rider][d] = [rider,d,0,d.year,doy,weekend,
                                       n[0],n[1],frac,
                                       "nan","nan"]
            


# output
with open(outfilename,'w') as outfile:
    owriter = csv.writer(outfile)
    
    # header
    owriter.writerow(["rider","date","rode","year","doy","weekend",
                      "possible_riders","actual_riders","fraction",
                      "distance","speed",
                      "precip","snow_depth","ave_temp","max_temp",
                      "min_temp","sunshine","peak_wind",
                      "prev_peak_wind","prev_sunshine","prev_ave_temp",
                      "prev_precip","prev_temp_diff"])        
    
    for rider in ride_dict:
        for d in sorted(ride_dict[rider]):
            ride_info = ride_dict[rider][d]
            owriter.writerow(ride_info + weather[d])