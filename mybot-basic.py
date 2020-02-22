#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic chatbot design --- for your own modifications
"""
#######################################################
# Initialise Wikipedia agent
#######################################################
"""

# THIS IS THROWING ME ERRORS, MAYBE NAME IS WRONG?

import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia('en')
wikipediaapi.log.setLevel(level=wikipediaapi.logging.ERROR)

#######################################################
# Initialise weather agent
#######################################################
import json, requests
APIkey = "" #insert your personal OpenWeathermap API key here if you have one, and want to use this feature
"""

#######################################################
#  Initialise AIML agent
#######################################################
from sklearn.feature_extraction.text import TfidfVectorizer
import aiml
import pandas as pd
import numpy as np
import operator ,os
import sys

# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)

# Use the Kernel's bootstrap() method to initialize the Kernel. The
# optional learnFiles argument is a file (or list of files) to load.
# The optional commands argument is a command (or list of commands)
# to run after the files are loaded.
# The optional brainFile argument specifies a brain file to load.
kern.bootstrap(learnFiles="mybot-basic.xml")

filepath='QA.csv'
csv_reader = pd.read_csv(filepath)
question_list = csv_reader[csv_reader.columns[0]].values.tolist()
answers_list  = csv_reader[csv_reader.columns[1]].values.tolist()
#######################################################
# Welcome user
#######################################################
print("Welcome to the holidays chat bot. You can use me to book holiday packages")
#######################################################
# Main loop
#######################################################

#TODO: Add check for whether the user has selected a destination or not
# before finishing the booking process



AIML = 'aiml'
while True:
    #get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    #pre-process user input and determine response agent (if needed)
    responseAgent = AIML
    #activate selected response agent
    if responseAgent == AIML:
        answer = kern.respond(userInput)
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break

        elif cmd == 1:
            os.subprocess.call(['train.py', "--dataset=" + params[1]])
            break;

        elif cmd == 99:
            query = userInput
            vectorizer = TfidfVectorizer(min_df=0, ngram_range=(2, 4), strip_accents='unicode', norm='l2',
                                         encoding='ISO-8859-1')
            X_train = vectorizer.fit_transform(np.array([''.join(que) for que in question_list]))
            X_query = vectorizer.transform([query])
            XX_similarity = np.dot(X_train.todense(), X_query.transpose().todense())
            XX_sim_scores = np.array(XX_similarity).flatten().tolist()
            dict_sim = dict(enumerate(XX_sim_scores))
            sorted_dict_sim = sorted(dict_sim.items(), key=operator.itemgetter(1), reverse=True)
            if sorted_dict_sim[0][1] == 0:
                print("Sorry, I did not get that please ask again")
            elif sorted_dict_sim[0][1] > 0:
                print(answers_list[sorted_dict_sim[0][0]])
    else:
        print(answer)


        """
        
                elif cmd == 1:
            wpage = wiki_wiki.page(params[1])
            if wpage.exists():
                print(wpage.summary)
                print("Learn more at", wpage.canonicalurl)
            else:
                print("Sorry, I don't know what that is.")
        elif cmd == 2:
            succeeded = False
            api_url = r"http://api.openweathermap.org/data/2.5/weather?q="
            response = requests.get(api_url + params[1] + r"&units=metric&APPID="+APIkey)
            if response.status_code == 200:
                response_json = json.loads(response.content)
                if response_json:
                    t = response_json['main']['temp']
                    tmi = response_json['main']['temp_min']
                    tma = response_json['main']['temp_max']
                    hum = response_json['main']['humidity']
                    wsp = response_json['wind']['speed']
                    wdir = response_json['wind']['deg']
                    conditions = response_json['weather'][0]['description']
                    print("The temperature is", t, "°C, varying between", tmi, "and", tma, "at the moment, humidity is", hum, "%, wind speed ", wsp, "m/s,", conditions)
                    succeeded = True
            if not succeeded:
                print("Sorry, I could not resolve the location you gave me.")
                
        """