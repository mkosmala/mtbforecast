#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
1. Download yearly historical file
2. Extract it
3. Go through and pull out the stations and data for the previous 14 days
4. Go through rolling file and update past data that are still forecasts
   and any missing information for existing past data
5. Download forecast data
6. For any days without a history data, update the forecast data up through
   the end of the forecast period
7. Copy the old file to an archived one
8. Replace old file with new one

Created on Tue Jan 30 17:16:22 2018

@author: mkosmala
"""

import requests
import csv
import gzip
from datetime import date, datetime, timedelta
from astral import Astral
from shutil import copyfile


def download_large_file(url):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter for large files
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    return local_filename


def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)


# min, max, and average temp in F each day
# Hah! Can't expect that there's a min or max temp on any given day (ARGH!!!)
def calculateTemps(dt,maxTemp,minTemp):
    
    temps = {}    
    
    for val in maxTemp["values"]:
        vdate = datetime.strptime(val["validTime"][0:10],"%Y-%m-%d").date()
        if vdate >= dt:
            farenheit = float(val["value"]) * 1.8 + 32
            temps[vdate] = [farenheit,"nan"]
    
    for val in minTemp["values"]:
        vdate = datetime.strptime(val["validTime"][0:10],"%Y-%m-%d").date()
        if vdate >= dt:
            farenheit = float(val["value"]) * 1.8 + 32
            if vdate in temps:
                temps[vdate][1] = farenheit
            else:
                temps[vdate] = ["nan",farenheit]
            
    aveTemp = {}
    minTemp = {}
    maxTemp = {}
    for vdate in temps:
        pair = temps[vdate]
        if pair[0]=="nan" or pair[1]=="nan":
            aveTemp[vdate] = "nan"
        else:
            aveTemp[vdate] = (pair[0] + pair[1])/2.0
        minTemp[vdate] = pair[1]
        maxTemp[vdate] = pair[0]
    
    return (aveTemp,minTemp,maxTemp)

# minutes of sunshine each day
def calculateSunshine(dt2,skyCover):
    
    cover = []    
    allSunshine = {}
    percSunshine = {}
    
    # load all skycover data in
    for val in skyCover["values"]:
        vtime = datetime.strptime(val["validTime"][0:13],"%Y-%m-%dT%H")
        perc = int(val["value"])
        cover.append([vtime,perc])

    a = Astral()
    city = a['Boston']
    # for each day get sunrise and sunset
    for i in daterange(dt2,dt2+timedelta(7)):
        sun = city.sun(date=i)
        sunrise = sun['sunrise'].replace(tzinfo=None)
        sunset = sun['sunset'].replace(tzinfo=None)
        
        # and caculate average sunshine
        sslist = []
        for pair in cover:            
            if pair[0] > sunrise and pair[0] < sunset:
                sslist.append(pair[1])
        # daily percentage of sunshine
        if len(sslist) > 0:
            dailyss = sum(sslist) / float(len(sslist))
        else:
            dailyss = 0
        # hours between sunrise and sunset
        tothours = int((sunset-sunrise).seconds / 60.0)       
        
        percSunshine[i] = dailyss/100.0
        allSunshine[i] = percSunshine[i] * tothours
    
    return (percSunshine,allSunshine)
        
# peak gust in miles per hour each day
def calculateMaxWinds(dt2,windGust):
    
    gusts = {}    
    
    for val in windGust["values"]:
        vdate = datetime.strptime(val["validTime"][0:10],"%Y-%m-%d").date()
        w = float(val["value"]) * (25.0 / 11)
        
        if vdate >= dt2:
            if vdate not in gusts:
                gusts[vdate] = 0
            if w > gusts[vdate]:
                gusts[vdate] = w
        
    return gusts
        
# total precip in inches per day
def calculatePrecip(dt2,precip):

    totalPrecip = {}

    for val in precip["values"]:
        vdate = datetime.strptime(val["validTime"][0:10],"%Y-%m-%d").date()
        p = float(val["value"]) / 25.4
        
        if vdate >= dt2:
            if vdate not in totalPrecip:
                totalPrecip[vdate] = 0.0
            totalPrecip[vdate] += p
        
    return totalPrecip

# this is currently just snowfall (in inches) each day
# we need to figure out how to track snow staying on the ground!
def calculateSnowfall(dt2,snow):

    totalSnow = {}

    for val in snow["values"]:
        vdate = datetime.strptime(val["validTime"][0:10],"%Y-%m-%d").date()
        p = float(val["value"]) / 25.4
        
        if vdate >= dt2:
            if vdate not in totalSnow:
                totalSnow[vdate] = 0.0
            totalSnow[vdate] += p
        
    return totalSnow


def get_forecast_data(dt):

    api = "https://api.weather.gov/gridpoints/"

    # Boston, Lowell, Nashua, Manchester
    four_cities = ["BOX/70,76","BOX/60,87","GYX/43,10","GYX/41,21"]
    
    # change this to going through all cities and averaging
    # or just take Lowell if it's too slow
    city = four_cities[1]
    
    response = requests.get(api+city)    
    data = response.json()
    
    # huh. Current day is tricky. If we ask in the morning, okay, but if
    # we ask at 6pm, what do we return?
    # for now, we'll just start with the next day, I guess
    #dt2 = dt + timedelta(1)    
    dt2 = dt    
    
    # I guess ideally we want to ping the forecaster EARLY each morning
    # and save it to have a forecast ready...    
    
    # for each day, we want to pull out:
    # precip, snow depth, ave temp, sunshine hours, peak wind
    
    # calculate ave temp as average of min and max (rather than interpolating)
    maxTemp = data["properties"]["maxTemperature"]
    minTemp = data["properties"]["minTemperature"]
    skyCover = data["properties"]["skyCover"]  # need to modify this by sunup and sundown times
    windGust = data["properties"]["windGust"]
    precip = data["properties"]["quantitativePrecipitation"]
    snow = data["properties"]["snowfallAmount"]
    
    # do temperatures
    aveTemps,minTemps,maxTemps = calculateTemps(dt2,maxTemp,minTemp)
    
    # do sunshine
    psunshine,tsunshine = calculateSunshine(dt2,skyCover)
    
    # do max wind
    maxWinds = calculateMaxWinds(dt2,windGust)
    
    # do precip
    precips = calculatePrecip(dt2,precip)
    
    # and snow...
    snowfall = calculateSnowfall(dt2,snow)
    
    
    # need defaults in case data is missing (blerg)
    # these are medians
    #default_precip = 0.01
    #default_snowpack = 0
    #default_aveTemp = 51.5
    #default_maxTemp = 61.2
    #default_minTemp = 40.5
    #default_sunshine = 412.7
    #default_maxWind = 23.6


    default_precip = "nan"
    default_snowfall = "nan"
    default_aveTemp = "nan"
    default_maxTemp = "nan"
    default_minTemp = "nan"
    default_tsunshine = "nan"
    default_psunshine = "nan"
    default_maxWind = "nan"
        
    
    # !!!
    # eek! We also need the previous week's data...
    # previous: peak wind, sunshine, ave temp, precip, (temp diff)
    # for now, use averages for previous week
    # these are means
    #prev_wind = 35.0
    #prev_sun = 2882.9
    #prev_precip = 0.78
    #prev_diff = 0.0
    

    
    # package it all up
    #header = ["AWND","PSUN","PRCP","SNOW","SNWD","TAVG","TMAX","TMIN","TSUN",
    #          "WESD","WESF","WSF5"]
    fdata = {}
    for i in daterange(dt2,dt2+timedelta(7)):

        fdata[i] = {}
        
        # fill data structure 
        # and check for missing data
        if i in precips:
            fdata[i]["PRCP"] = precips[i]
        else:
            fdata[i]["PRCP"] = default_precip
        if i in aveTemps:
            fdata[i]["TAVG"] = aveTemps[i]
        else:
            fdata[i]["TAVG"] = default_aveTemp
        if i in maxTemps:
            fdata[i]["TMAX"] = maxTemps[i]
        else:
            fdata[i]["TMAX"] = default_maxTemp
        if i in minTemps:
            fdata[i]["TMIN"] = minTemps[i]
        else:
            fdata[i]["TMIN"] = default_minTemp
        if i in tsunshine:
            fdata[i]["TSUN"] = tsunshine[i]
        else:
            fdata[i]["TSUN"] = default_tsunshine
        if i in psunshine:
            fdata[i]["PSUN"] = psunshine[i]
        else:
            fdata[i]["PSUN"] = default_psunshine
        if i in maxWinds:
            fdata[i]["WSF5"] = maxWinds[i]
        else:
            fdata[i]["WSF5"] = default_maxWind
        if i in snowfall:
            fdata[i]["SNOW"] = snowfall[i]
        else: # snow hack based on precip and temp
            if i in precips and i in maxTemps and maxTemps[i] <= 35.6:
                fdata[i]["SNOW"] = precips[i]
            else:
                fdata[i]["SNOW"] = default_snowfall
        
        # could get these in the future
        fdata[i]["AWND"] = "nan"
        fdata[i]["SNWD"] = "nan"
        fdata[i]["WESD"] = "nan"
        fdata[i]["WESF"] = "nan"
                    
    return fdata

# keep the new data unless it's "nan", then use old data
# assumes all the keys in newdata are also in olddata
def reconcile_data(newdata,olddata):
    recdata = {}
    for wtype in newdata:
        if newdata[wtype] != "nan":
            recdata[wtype] = newdata[wtype]
        else:
            recdata[wtype] = olddata[wtype]
    return recdata

def fill_in_blanks(w):
    for wtype in ["AWND","PSUN","PRCP","SNOW","SNWD","TAVG","TMAX","TMIN",
                  "TSUN","WESD","WESF","WSF5"]:
        if wtype not in w:
            w[wtype] = "nan"
    return w

# expects a gzipped file from NOAA
def read_NOAA_data(noaa_file):

    # read it and save relevant data
    # noaa_data keyed by date, then weather type; value is list of weather values
    print("reading NOAA yearly file")
    noaa_data = {}
    with gzip.open("./"+noaa_file,'rt') as nfile:
        nreader = csv.reader(nfile)
        
        # no header
        
        # we're looking only for specific stations
        # and specific data types
        # on specific dates
        for row in nreader:
            if (row[0] in stations and 
                row[1] >= sdate_str and 
                row[2] in head):
                
                dt = datetime.strptime(row[1], "%Y%m%d").date()
                if dt not in noaa_data:
                    noaa_data[dt] = {}
                
                wtype = row[2]
                if wtype not in noaa_data[dt]:
                    noaa_data[dt][wtype] = []
                
                noaa_data[dt][wtype].append(float(row[3]))
    
    # next, combine data across region for each date and type
    print("averaging across region for NOAA file")
    noaa_ave = {}
    for dt in noaa_data:
        noaa_ave[dt] = {}
        for wtype in noaa_data[dt]:
            vlist = noaa_data[dt][wtype]
            wtype_ave = sum(vlist)/float(len(vlist))

            # convert to the proper value based on weather type            
            # sinces they're all integers(!) in file
            if wtype=="AWND":  # recorded in tenths of a m/s !!!
                val = (wtype_ave/10.0) * (3600.0/1609.344)
            elif wtype=="PSUN":  # will hopefully be in percentages
                val = wtype_ave/100.0
            elif wtype=="PRCP":  # recorded in tenths of mm!!!
                val = (wtype_ave/10.0)/25.4
            elif wtype=="SNOW":  # recorded in mm
                val = wtype_ave/25.4                
            elif wtype=="SNWD":  # recorded in mm
                val = wtype_ave/25.4
            elif wtype=="TAVG":  # recorded in tenths of degree C!!!
                val = (wtype_ave/10.0) * 1.8 + 32
            elif wtype=="TMAX":  # recorded in tenths of degree C!!!
                val = (wtype_ave/10.0) * 1.8 + 32                
            elif wtype=="TMIN":  # recorded in tenths of degree C!!!
                val = (wtype_ave/10.0) * 1.8 + 32
            elif wtype=="TSUN":  # ??? should be in minutes
                val = wtype_ave
            elif wtype=="WESD":  # recorded in tenths of mm!!!
                val = (wtype_ave/10.0)/25.4
            elif wtype=="WESF":  # recorded in tenths of mm!!!
                val = (wtype_ave/10.0)/25.4
            elif wtype=="WSF5":  # recorded in tenths of a m/s !!!
                val = (wtype_ave/10.0) * (3600.0/1609.344)
                
            noaa_ave[dt][wtype] = val

        # fill in missing data with "nan"
        noaa_ave[dt] = fill_in_blanks(noaa_ave[dt])

    return noaa_ave    


# combine data from old file, new observed data, and forecast
def combine_data_sources(wdata,new_wdata,fdata):
    
    # earliest date
    diter = sorted(wdata)[0]
    
    # latest date
    end_date = sorted(fdata)[-1]
    
    odata = {}
    while diter <= end_date:
        
        source = "H"          
        oldsource = "N"
        if diter in wdata:
            oldsource = wdata[diter]["Type"] 
        
        # CASE 1: old_source = H and no new data, then use old data
        if oldsource=="H" and diter not in new_wdata:
            thisday = wdata[diter]
            
        # CASE 2: old_source = H and new data avail, then use new data before old data
        elif oldsource=="H":
            thisday = reconcile_data(new_wdata[diter],wdata[diter])
    
        # CASE 3: old_source = N and new data avail, then use new data       
        elif oldsource=="N" and diter in new_wdata:
            thisday = new_wdata[diter]
        
        # CASE 4: old_source = F and before today and new data avail, 
        #         then use new data before old forecast data
        elif oldsource=="F" and diter<date.today() and diter in new_wdata:
            thisday = reconcile_data(new_wdata[diter],wdata[diter])
        
        # CASE 5: old_source = F and before today and no new data avail,
        #         then use old forecast data
        elif oldsource=="F" and diter<date.today():
            thisday = wdata[diter]
            source = "F"
        
        # CASE 6: old_source = F and today or after, use new forecast
        #         before old forecast
        elif oldsource=="F" and diter in fdata:
            thisday = reconcile_data(fdata[diter],wdata[diter])
            source = "F"
        
        # case 7: weird gap in new forecast covered by old forecast
        elif oldsource=="F":    
            thisday = wdata[diter]
            source = "F"
        
        # case 8: new forecast
        elif oldsource=="N" and diter in fdata:
            thisday = fdata[diter]
            source = "F"
        
        # CASE 9: old_source = N and no new data
        else:
            thisday = {"AWND":"nan","PSUN":"nan","PRCP":"nan","SNOW":"nan",
                       "SNWD":"nan","TAVG":"nan","TMAX":"nan","TMIN":"nan",
                       "TSUN":"nan","WESD":"nan","WESF":"nan","WSF5":"nan"}
            source = "N"
    
        # save it
        odata[diter] = thisday
        odata[diter]["Type"] = source
    
        # increment day
        diter += timedelta(days=1)            

    return odata


def calculate_snow_depth(data):
    # start at bottom and find first historical snowdepth reading
    # latest date
    end_date = sorted(data)[-1]
    dt = end_date
    while not (data[dt]["Type"] == "H" and data[dt]["SNWD"] != "nan"):
        dt = dt - timedelta(days=1)
    base_depth = float(data[dt]["SNWD"])
    
    # now go forward, estimating the snow depth. If we have no information
    # on snowfall and temperature, change in snow depth = 0 
    dt = dt + timedelta(days=1)
    dsnow = base_depth
    while dt <= end_date:
        # add snow
        if data[dt]["SNOW"] != "nan":
            dsnow += float(data[dt]["SNOW"])
        # subtract snow
        # super simple mechanistic model ("degree-day method" from USDA)
        if data[dt]["TAVG"] != "nan":
            tavg = float(data[dt]["TAVG"])
            deltaT = tavg-32
            # only melt if above freezing!
            if deltaT > 0:
                melt = deltaT * 0.06  # 0.06 = Cm default value
                dsnow = dsnow - melt
        data[dt]["SNWD"] = dsnow
        dt += timedelta(days=1)

    return data
    

### MAIN

# local weather data file
roll_file = "./rolling_weather.csv"

# NOAA data file
year = str(date.today().year)
noaa_url = "https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/"+year+".csv.gz"

# stations we want to use
station_file = "./static/stations.csv"


stations = []
# read in the stations
with open(station_file,'r') as sfile:
    ireader = csv.reader(sfile)
    for row in ireader:
        stations.append(row[0])

# read in current rolling weather data file
print("reading current file")
head = []
wdata = {}
with open(roll_file,'r') as rfile:
    ireader = csv.reader(rfile)
    
    head = next(ireader)
    for row in ireader:
        dt = datetime.strptime(row[0], "%Y-%m-%d").date()
        wdata[dt] = {}
        for wt,val in zip(head[1:],row[1:]):
            wdata[dt][wt] = val

# go back at least two weeks before latest data to check if anything has changed        
# get data from this date forward
startdate = max(wdata) - timedelta(days=14)
sdate_str = startdate.strftime("%Y%m%d")

# get monthly file(s) from the website
print("getting NOAA yearly file")
noaa_file = download_large_file(noaa_url)    

# read and convert new weather data
new_wdata = read_NOAA_data(noaa_file)

#for dt in noaa_ave:
#    new_wdata[dt] = {}
#    for wtype in head:
#        if wtype in noaa_ave[dt]:
#            new_wdata[dt][wtype] = noaa_ave[dt][wtype]
#        elif wtype in wdata[dt]:
#            new_wdata[dt][wtype] = wdata[dt][wtype]
#        else:
#            new_wdata[dt][wtype] = "nan"

# now get the forecast data
print("getting forecast data")
fdata = get_forecast_data(date.today())

# go through all data
# for each date do:
# 1. new observational data, if available
# 2. old observational data, if available
# 3. forecast data, if available

print("combining data sources")
almostoutdata = combine_data_sources(wdata,new_wdata,fdata)

print("calculating snow depth")
# calculate the snow on the ground for forecast data
outdata = calculate_snow_depth(almostoutdata)
    
print("outputting data")

# save old weather data
yesterday = date.today()-timedelta(days=1)
dst = "./old_weather/rolling_weather_"+yesterday.strftime("%Y%m%d")+".csv"
copyfile(roll_file, dst)

# overwrite rolling weather file
with open(roll_file,'w') as ofile:
    owriter = csv.writer(ofile)
    
    owriter.writerow(head)    
    
    for dt in sorted(outdata):
        thisday = outdata[dt]
        owriter.writerow([dt.strftime("%Y-%m-%d"),thisday["Type"],
                          thisday["AWND"],thisday["PSUN"],thisday["PRCP"],
                          thisday["SNOW"],thisday["SNWD"],thisday["TAVG"],
                          thisday["TMAX"],thisday["TMIN"],thisday["TSUN"],
                          thisday["WESD"],thisday["WESF"],thisday["WSF5"]])
    

    
    