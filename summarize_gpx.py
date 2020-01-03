#!/usr/bin/env python

import gpxpy
import gpxpy.gpx

#import datetime
#import pandas as pd

import sys
import csv
import os


# -*- coding: utf-8 -*-
"""
Basic summary of data

kosmala
"""

if len(sys.argv) < 3 :
    print ("format: summarize_gpx.py <input dir>  <output file>")
    exit(1)

indirname = sys.argv[1]
outfilename = sys.argv[2]

# keep of dictionary of rides keyed by rider name
allrides = {}

# get rider names and their rides
for x in os.walk(indirname):
    if x[1] == []:
        
        # name
        riderpath = x[0]
        ridername = riderpath.split('/')[-1]
        
        # rides
        allrides[ridername] = x[2]

# now open the rides and check them out. Collect:

# date
# length of ride
# ave speed of ride
# start locatoin of ride

total_count = 0
ride_info = []
for rider in allrides:
    print(rider)
    for ride in allrides[rider]:

        total_count += 1

        biking = 0
        mtb = 0
        local = 0

        # only use rides -- nope! everything
        if "-Ride." in ride:
            biking = 1
        
        activity = ride[16:-4]
        
        try:
            gpx = gpxpy.parse(open(os.path.join(indirname,rider,ride)))
            track = gpx.tracks[0]
            #print(track.name)
            dur = track.get_duration()*1.0/60
            dist = track.length_3d()/1609.34
            avesp = dist/(dur/60.0)
            
            firstpt = track.segments[0].points[0]
            lat = firstpt.latitude
            lon = firstpt.longitude
            dt = track.get_time_bounds().start_time   #.isoformat()
            
            # keep rides that are between 2 and 9 mph and in the
            # right geographic location
            if avesp >= 2.0 and avesp <= 9.0 and biking == 1:
                mtb = 1
            
            if (lat>= 42.032435 and lat<= 43.207649 and
                lon>= -71.881771 and lon<= -70.584011):
                local = 1
                    
            ride_info.append([rider,activity,biking,mtb,local,
                              dist,dur,avesp,dt,lat,lon])
            
            
        except:
            print ("Couldn't read: " + rider + "/" + ride)
        

print("Total activities: " + str(total_count))
        
# output data
with open(outfilename,'w') as outfile:
    owriter = csv.writer(outfile)
    
    # header
    owriter.writerow(['rider','activity','bike','mountain_bike','local','distance','duration','ave_speed',
                      'date_time','latitude','longitude'])
    
    owriter.writerows(ride_info)
    