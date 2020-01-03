#!/usr/bin/env python

from csv import reader
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, validators
from datetime import date, datetime, timedelta
import pickle
import numpy as np
#import requests
#import json
from astral import Astral

#import pandas as pd

# pandas can read csvs from URL
#df = pd.read_csv('https://data.boston.gov/dataset/c8b8ef8c-dd31-4e4e-bf19-af7e4e0d7f36/resource/29e74884-a777-4242-9fcc-c30aaaf3fb10/download/economic-indicators.csv',
#                 parse_dates=[['Year', 'Month']])
#length = len(df)

hist_weather_file = './static/historical_weather.csv'
forecast_file = './rolling_weather.csv'


app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f28abc1f27567d441f2b6176a'


# work on the validators to make sure it's the right format
class ReusableForm(Form):
    name = TextField('Date:', validators=[validators.required()])


def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

# historical data
# returns a dictionary keyed by date and containing all the weather data
def get_weather_data():
    
    wdata = {}
    with open(hist_weather_file,'r') as wfile:
        wreader = reader(wfile)
        header = next(wreader) # remove header row
        for row in wreader:
            #print (row[0])
            asdt = datetime.strptime(row[0], "%Y-%m-%d").date()
            wdata[asdt] = row[1:]
        
    return (header[1:],wdata)

# OBSOLETE
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

# OBSOLETE
# minutes of sunshine each day
def calculateSunshine(dt2,skyCover):
    
    cover = []    
    allSunshine = {}
    
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
        dailyss = sum(sslist) / float(len(sslist))
        # hours between sunrise and sunset
        tothours = int((sunset-sunrise).seconds / 60.0)       
        
        allSunshine[i] = (dailyss/100.0) * tothours
    
    return allSunshine
        
# OBSOLETE
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
    
# OBSOLETE        
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

# OBSOLETE
# this is currently just snowfall (in inches) each day
# we need to figure out how to track snow staying on the ground!
def calculateSnowpack(dt2,snow):

    totalSnow = {}

    for val in snow["values"]:
        vdate = datetime.strptime(val["validTime"][0:10],"%Y-%m-%d").date()
        p = float(val["value"]) / 25.4
        
        if vdate >= dt2:
            if vdate not in totalSnow:
                totalSnow[vdate] = 0.0
            totalSnow[vdate] += p
        
    return totalSnow
    
    
    
# We now read this from a daily rolling file, instead of making an API
# call when the user clicks the button. Not only is this faster, it
# it provides more complete data.
def get_forecast_data(dt):
        
    wdata = {}
    with open(forecast_file,'r') as wfile:
        wreader = reader(wfile)
        orig_header = next(wreader)[2:] # remove header row
        for row in wreader:
            #print (row[0])
            asdt = datetime.strptime(row[0], "%Y-%m-%d").date()
            wdata[asdt] = row[2:]
    
    # convert header names
    swap = {"PRCP":"precip", "SNWD":"snow_depth", "TAVG":"ave_temp",
            "TMAX":"max_temp", "TMIN":"min_temp", "TSUN":"sunshine",
            "WSF5":"peak_wind"}     
    header = []
    for item in orig_header:
        if item in swap:
            header.append(swap[item])
        else:
            header.append(item)
    
    # calculate previous week's data for all days dt and six days after  
    # (or what's available)
    prev_header = ["prev_peak_wind","prev_sunshine","prev_ave_temp",
                   "prev_precip","prev_temp_diff"]    
    for i in daterange(dt,dt+timedelta(days=7)):
        
        max_peak_wind = 0.0 
        sum_sunshine = 0.0
        ssu = 0
        sum_prev_ave_temp = 0.0
        spat = 0
        sum_prev_precip = 0.0
        spp = 0
 
        for j in daterange(i-timedelta(days=7),i):
            if wdata[j][11] != "nan":  # 11=peak wind
                if float(wdata[j][11]) > max_peak_wind:
                    max_peak_wind = float(wdata[j][11])
            if wdata[j][8] != "nan":  # 8=sunshine
                sum_sunshine += float(wdata[j][8])
                ssu  += 1
            if wdata[j][5] != "nan": # 5=ave temp
                sum_prev_ave_temp += float(wdata[j][5])
                spat += 1
            if wdata[j][2] != "nan": # 2=precip
                sum_prev_precip += float(wdata[j][2])
                spp += 1

        # deal with missing data
        sum_sunshine = sum_sunshine/(ssu/7.0)
        sum_prev_precip = sum_prev_precip/(spp/7.0)
        
        temp_diff = float(wdata[i][5])-(sum_prev_ave_temp/spat*1.0)
        prev_data = [max_peak_wind,sum_sunshine,sum_prev_ave_temp,
                     sum_prev_precip,temp_diff]
        wdata[i] = wdata[i] + prev_data
    
    return (header+prev_header,wdata)

# future data
# returns a dictionary keyed by date and containing all the weather data   
#def get_forecast_data(dt):
#
#    api = "https://api.weather.gov/gridpoints/"
#
#    # Boston, Lowell, Nashua, Manchester
#    four_cities = ["BOX/70,76","BOX/60,87","GYX/43,10","GYX/41,21"]
#    
#    # change this to going through all cities and averaging
#    # or just take Lowell if it's too slow
#    city = four_cities[1]
#    
#    response = requests.get(api+city)    
#    data = response.json()
#    
#    # huh. Current day is tricky. If we ask in the morning, okay, but if
#    # we ask at 6pm, what do we return?
#    # for now, we'll just start with the next day, I guess
#    dt2 = dt + timedelta(1)    
#    
#    # I guess ideally we want to ping the forecaster EARLY each morning
#    # and save it to have a forecast ready...    
#   
#    # for each day, we want to pull out:
#    # precip, snow depth, ave temp, sunshine hours, peak wind
#   
#    # calculate ave temp as average of min and max (rather than interpolating)
#    maxTemp = data["properties"]["maxTemperature"]
#    minTemp = data["properties"]["minTemperature"]
#    skyCover = data["properties"]["skyCover"]  # need to modify this by sunup and sundown times
#    windGust = data["properties"]["windGust"]
#    precip = data["properties"]["quantitativePrecipitation"]
#    snow = data["properties"]["snowfallAmount"]
#    
#    # do temperatures
#    aveTemps,minTemps,maxTemps = calculateTemps(dt2,maxTemp,minTemp)
#    
#    # do sunshine
#    sunshines = calculateSunshine(dt2,skyCover)
#    
#    # do max wind
#    maxWinds = calculateMaxWinds(dt2,windGust)
#    
#    # do precip
#    precips = calculatePrecip(dt2,precip)
#    
#    # and snow...
#    snowpacks = calculateSnowpack(dt2,snow)
#    
#    
#    # need defaults in case data is missing (blerg)
#    # these are medians
#    default_precip = 0.01
#    default_snowpack = 0
#    default_aveTemp = 51.5
#    default_maxTemp = 61.2
#    default_minTemp = 40.5
#    default_sunshine = 412.7
#    default_maxWind = 23.6
#        
#    
#    # !!!
#    # eek! We also need the previous week's data...
#    # previous: peak wind, sunshine, ave temp, precip, (temp diff)
#    # for now, use averages for previous week
#    # these are means
#    prev_wind = 35.0
#    prev_sun = 2882.9
#    prev_precip = 0.78
#    prev_diff = 0.0
#    
#    
#    # package it all up
#    header = ["precip","snow_depth","ave_temp","max_temp","min_temp",
#              "sunshine","peak_wind",
#              "prev_peak_wind","prev_sunshine","prev_precip","prev_temp_diff"]
#    wdata = {}
#    for i in daterange(dt2,dt2+timedelta(7)):
#        
#        # check for missing data
#        if i in precips:
#            pr = precips[i]
#        else:
#            pr = default_precip
#        if i in aveTemps:
#            at = aveTemps[i]
#        else:
#            at = default_aveTemp
#        if i in maxTemps:
#            mxt = maxTemps[i]
#        else:
#            mxt = default_maxTemp
#        if i in minTemps:
#            mnt = minTemps[i]
#        else:
#            mnt = default_minTemp
#        if i in sunshines:
#            su = sunshines[i]
#        else:
#            su = default_sunshine
#        if i in maxWinds:
#            wd = maxWinds[i]
#        else:
#            wd = default_maxWind
#        if i in snowpacks:
#            sn = snowpacks[i]
#        else: # snow hack based on precip and temp
#            if i in precips and mxt <= 32:
#                sn = precips[i]
#            else:
#                sn = default_snowpack
#        
#        w = [pr,sn,at,mxt,mnt,su,wd,
#             prev_wind,prev_sun,prev_precip,prev_diff]
#        wdata[i] = w
#    
#    return (header,wdata)


# returns whether it's a good day to ride or not
def forecast_day(dt,weather,index_dict,scaler,model):
        
    # we need to know what day of the year and day of the week it is
#    year = dt.year
#    doy = dt.timetuple().tm_yday
#    dayweek = dt.weekday()
#    weekend = 0
#    if dayweek == 5 or dayweek == 6:
#        weekend = 1 
    weather_labels,weather_data = weather
    w = weather_data[dt]

    


    # set up the new vector for prediction
    new_vector = np.zeros(len(index_dict))

    # we can nullify the effect of year by making it the mean
    # and then a little conservative by scootching down a bit
    #new_vector[index_dict["year"]] = year
    new_vector[index_dict["year"]] = 2015
    
    # nullify doy    
    #new_vector[index_dict["doy"]] = doy
    new_vector[index_dict["doy"]] = dt.timetuple().tm_yday   
    
    # we can nullify the effect of weekend by making it the mean
    #new_vector[index_dict["weekend"]] = weekend
    new_vector[index_dict["weekend"]] = 2.0/7.0
    
    
    
    
#    weather_labels = ["precip","snow_depth","ave_temp",
#                      "sunshine","peak_wind","prev_peak_wind",
#                      "prev_precip","prev_temp_diff"]
 
         
 
    today_weather = dict(zip(weather_labels,w))
 
    for feature in index_dict:
        if feature not in ["year","doy","weekend"]:
            #print(dt)
            #print(today_weather)
            new_vector[index_dict[feature]] = today_weather[feature]

    # reshape so transform and predict will use it
    reshaped = new_vector.reshape(1,-1)

    # scale it
    transformed = scaler.transform(reshaped)

    #print(new_vector)
    #print(transformed)
    #print(model.coef_)

    # run the model
    prediction = model.predict(transformed)
    
    #print(prediction)
    
    return prediction


def get_weather_caveats(dt,weather):

    txt = ""
    icons = []

    # we need to know what day of the year and day of the week it is
    weather_labels,weather_data = weather
    w = dict(zip(weather_labels,weather_data[dt]))
    
    #print(w)    
    
    if float(w['snow_depth']) > 3.0:
        icons.append("snow_deep_small.png")
        txt += "DEEP-SNOW "
    elif float(w['snow_depth']) > 0.0:
        icons.append("snow_shallow_small.png")        
        txt += "SURFACE-SNOW "
    
    if float(w['precip']) > 0.1:
        if float(w['max_temp']) < 35:
            txt += "SNOWING "
            icons.append("snowing_small.png")
        elif float(w['precip']) > 0.5:
            txt += "RAINING "
            icons.append("raining_small.png")
        elif float(w['sunshine']) <= 660:
            txt += "DRIZZLING "
            icons.append("drizzling_small.png")
    
    if float(w['max_temp']) < 32:
        txt += "COLD "
        icons.append("cold_small.png")        
    elif float(w['max_temp']) > 85:
        txt += "HOT "
        icons.append("hot_small.png")        
        
    if float(w['peak_wind']) > 30:
        txt += "WINDY "
        icons.append("windy_small.png")        
    
    if float(w['prev_peak_wind']) > 40:
        txt += "RECENT-HIGH-WINDS "
        icons.append("recent_wind_small.png")            
    
    if float(w['prev_precip']) > 0.2:
        if float(w['max_temp']) < 40:
            txt += "ICY "
            icons.append("icy_small.png")        
        else:
            if float(w['prev_precip']) > 1:
                txt += "MUDDY "
                icons.append("muddy_small.png")        
    
    if float(w['sunshine']) > 660:
        txt += "SUNNY "
        icons.append("sunny_small.png")        
    
    return icons
    
    
@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/', methods=['GET', 'POST'])
def main():
    #image = "empty.png"
    bad_day = "bike_no_small.png"
    caution_day = "bike_caution_small.png"
    good_day = "bike_good_small.png"
    awesome_day = "bike_great_small.png"
    #image_text = ""

    # load the weather data
    hist_weather = get_weather_data()

    # load the pickles for what type of day it is
    pkl_file = open('./static/pickle/categories_riders','rb')
    index_dict = pickle.load(pkl_file)
    pkl_file = open('./static/pickle/transformations_riders','rb')
    scaler = pickle.load(pkl_file)    
    pkl_file = open('./static/pickle/logistic_regression_riders_model.pkl','rb')
    model = pickle.load(pkl_file)
    
    # load the pickles for distance
    pkl_file = open('./static/pickle/categories_distance','rb')
    index_dict_distance = pickle.load(pkl_file)
    pkl_file = open('./static/pickle/transformations_distance','rb')
    scaler_distance = pickle.load(pkl_file)    
    pkl_file = open('./static/pickle/linear_regression_distance_model.pkl','rb')
    model_distance = pickle.load(pkl_file)
    
    # load the pickles for speed
    pkl_file = open('./static/pickle/categories_speed','rb')
    index_dict_speed = pickle.load(pkl_file)
    pkl_file = open('./static/pickle/transformations_speed','rb')
    scaler_speed = pickle.load(pkl_file)    
    pkl_file = open('./static/pickle/linear_regression_speed_model.pkl','rb')
    model_speed = pickle.load(pkl_file)
    
    
    
    
        
    form = ReusableForm(request.form)
 
    date_text = []
    images = [] 
    extras = []
    daily_extras = []
    #extras2 = []
    okay = False
 
    #print (form.errors)
    if request.method == 'POST':
        
        click = request.form['button']
        
        if click=="Get the Forecast":
            firstdate = date.today()            
            weather = get_forecast_data(firstdate)
            
            # for right now, skip today
            #firstdate += timedelta(1)            
            
            okay = True            
            
        else: # "Get Conditions"
            weather = hist_weather        
        
            # this is the value in the text box, hopefully a date
            name=request.form['name']
    
            # ERROR CHECK
            # check that input is in the correct format and convert to datetime
            okay = False
            try:
                firstdate = datetime.strptime(name, "%Y-%m-%d").date()
                
                # also need to check that it's 2015-2017  !!!           
                if firstdate >= date(2015,1,1) and firstdate <= date(2017,12,31):
                    okay = True
                    
            except ValueError:
                # PUT HELPFUL ERROR MESSAGE HERE !!!
                pass

        if okay:
            # Do the forecast for a week
            images.clear()
            date_text.clear()
            extras.clear()
            daily_extras.clear()
            #extras2.clear()
            for dt in daterange(firstdate,firstdate+timedelta(7)):
                                                
                # get the text string
                txt1 = dt.strftime('%a').upper() 
                txt2 = dt.strftime('%b ').upper() + str(dt.day)
                date_text.append([txt1,txt2])                                                
                                                
                # get the day forecast and the image
                day_type = forecast_day(dt,weather,index_dict,scaler,model)
                if day_type == 0:
                    image = bad_day
                    #image_text = "Bad riding conditions"
                elif day_type == 1:
                    image = caution_day
                    #image_text = "Bike with caution"
                elif day_type == 2:
                    image = good_day
                    #image_text = "Good to ride!"
                else:
                    image = awesome_day
                    #image_text = "Awesome day to ride!"
                images.append(image)
                
                # get the predicted distance and speed
                pred_dist = forecast_day(dt,weather,
                                         index_dict_distance,scaler_distance,
                                         model_distance)
                                                        
                pred_speed = forecast_day(dt,weather,
                                         index_dict_speed,scaler_speed,
                                         model_speed)

                xtxt = ""
                if day_type > 0:
                    if pred_dist < -0.25:
                        xtxt += "SHORT "
                    elif pred_dist > 0.25:
                        xtxt += "LONG "
                    if pred_speed < -0.25:
                        xtxt += "SLOW "
                    elif pred_speed > 0.25:
                        xtxt += "FAST "
                
                #xtxt += get_weather_caveats(dt,weather)                
                #extras2.append(xtxt)

                daily_extras = get_weather_caveats(dt,weather)              
                extras.append(daily_extras)
                
                
            #flash('Hello ' + name)            
        else:
            flash('Please enter a date in format YYYY-MM-DD.')    
    
        
    #images = [image]*7
    
    return render_template('index.html', form=form, 
                           images=images, datetxt=date_text,
                           extras=extras,
                           result=okay)

# python app
if __name__ == '__main__':
    app.run(debug=True, port=5957)
