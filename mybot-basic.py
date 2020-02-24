#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from IPython.utils.py3compat import execfile
from sklearn.feature_extraction.text import TfidfVectorizer
import aiml
import nltk
import pandas as pd
import numpy as np
import operator ,os
import sys

#######################################################
#  Initialise AIML agent
#######################################################
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


V = """ 
Eiffel_Tower => {}
Retiro_Gardens => {}
Tower_of_Pisa => {}
Roman_Colosseum => {}
Champs_Elysees => {}
Alhambra => {}
Ramblas => {}
be_in => {}
Spain => ES
Italy => IT
France => FR
"""

folval = nltk.Valuation.fromstring(V)
grammar_file = 'simple-sem.fcfg'
objectCounter = 0

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

        # is x in y ?
        elif cmd == 4:
            g = nltk.Assignment(folval.domain)
            mod = nltk.Model(folval.domain, folval)
            sent = 'some ' + params[1] + ' are_in  ' + params[2]
            results = nltk.evaluate_sents([sent], grammar_file,mod,g)[0][0]
            if results[2] == True:
                print("Yes.")
            else:
                print("No.")

        # x is in y
        elif cmd == 5:
            obj = 'obj' + str(objectCounter)
            objectCounter +=1

            folval['obj' + obj] = obj # insert constant
            # folval[params[1]] = params[0] + params[1]

            if len(folval[params[1]]) == 1:
                if('',) in folval[params[1]]:
                    folval[params[1]].clear()
                folval[params[1]].add((obj,)) # insert info
                print("Added?")
                if len(folval["be_in"]) == 1:
                    if('',) in folval["be_in"]:
                        folval["be_in"].clear()

                folval["be_in"].add((obj,folval[params[2]])) # insert location
                print("Think so")

        # Which in country
        elif cmd == 6:
            g = nltk.Assignment(folval.domain)
            mod = nltk.Model(folval.domain, folval)
            exp = nltk.Expression.fromstring(("be_in(x," + params[1] + ")"))
            sat = mod.satisfiers(exp, "x", g)
            if len(sat) == 0:
                print("None")
            else:
                # find satisfying objects in the dict
                # and print their typenames
                sol = folval.values()
                for so in sat :
                    for k, v in folval.items() :
                        if len(v) > 0 :
                            vl = list(v)
                            if len(vl[0]) == 1:
                                for i in vl:
                                    if i[0] == so:
                                        print(k)
                                        break

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
