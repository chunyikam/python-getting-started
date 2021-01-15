# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 11:34:47 2020
This is 
@author: chuny
"""



from flask import Flask, flash, redirect, render_template, request, session, abort, jsonify, send_file
import json
from skilio import *
import datetime 
import os
#from wordcloud import WordCloud 
#import matplotlib.pyplot as plt; plt.rcdefaults()
#import numpy as np
#import matplotlib.pyplot as plt
#import jinja2
import pandas as pd
#from io import BytesIO
#import os
#app = Flask(__name__,static_url_path="/static", static_folder='/graph/')
#app = Flask(__name__,static_url_path="/static", static_folder='/')
app = Flask(__name__,static_url_path="/static", static_folder='static')
#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

#from flask_appbuilder.charts.views import DirectByChartView
#from flask_appbuilder.models.sqla.interface import SQLAInterface

directory="C:/nus/FYP/dictionary/"
#dictionary="finalDict.csv"
#dictionary="finalfinaldictionary.csv"
dictionary="finalfinaldictionary_07102020v1.csv"


#modelDirectory="C:/nus/FYP/python/reference/model/"
modelDirectory="C:/nus/FYP/python/production/model/"
modelType="EnsembleNBXGLRSoft"
#modelType="XGB"

#function not in use
@app.route("/graph/")
def index():
    name="test "
    return "Flask App!" +name


@app.route('/', methods=['POST','GET'])
#calls the main page -main.html
def bar():
    #checks if  POST command(submit has been pressed)
    if request.method=="POST":
        format = "%d/%m/%Y"
        today=datetime.datetime.now().strftime(format)
        reflection=request.form["reflection"]
        text=reflection
        #calls the AI model from skilio.py 
        predicted=extractResult(directory,dictionary, modelDirectory, text, modelType) 
        json_str = json.dumps(predicted)
        resp = json.loads(json_str)
        #extract json output. return the dictionary output 
        reflection=resp[0]['reflection']
        #no of times adventurous behavior was displayed 
        Adventurous=resp[0]['noAdventurous'] 
        Analytical=resp[0]['noAnalytical']
        ConsensusBuilder=resp[0]['noConsensus Builder']
        Imaginative=resp[0]['noImaginative']
        PlanExecutor=resp[0]['noPlan Executor']
        Resilient=resp[0]['noResilient']
       

        TargetSetter=resp[0]['noTarget Setter']
        Technical=resp[0]['noTechnical']
        #is Teamwork displayed or not
        
        Teamwork=resp[0]['Teamwork']
        #no of times Teamwork is displayed
        noTeamwork=resp[0]['noTeamwork']
    
        Clarity=resp[0]['noClarity']
        Considerate=resp[0]['noConsiderate']
        Courteous=resp[0]['noCourteous']
        Credible=resp[0]['noCredible']
        
         #is Communication displayed or not
        Communication=resp[0]['Communication']
        #no of times Communication is displayed
        noCommunication=resp[0]['noCommunication']
    
        CoachabilityTeachability=resp[0]['noCoachability/ teachability']
      
        Humble=resp[0]['noHumble']
  

        IntellectuallyCurious=resp[0]['noIntellectually curious']
 

        ResponsibleForOneOwnGrowth=resp[0]["noResponsible for one's own growth"]
        
        #is Willingness to learn  displayed or not
        willingnesstolearn=resp[0]['willingness to learn']
        #no of times Teamwork is displayed
        nowillingnesstolearn=resp[0]['nowillingness to learn']
        
        
        Curiosity=resp[0]['noCuriosity']
       

        FutureOriented=resp[0]['noFuture-oriented']
        
        OpenMinded=resp[0]['noOpen-minded']
      
        Resourceful=resp[0]['noResourceful']
      
        RiskTaker=resp[0]['noRisk-Taker']
       
         #is Teamwork displayed or not
        Adaptability=resp[0]['Adaptability']
        #no of times Teamwork is displayed
        noAdaptability=resp[0]['noAdaptability']
        Empathetic=resp[0]['noEmpathetic']
      
        Empowering=resp[0]['noEmpowering']

        OneOfIntegrityAndFairness=resp[0]['One of Integrity and Fairness']
       
        Synergiser=resp[0]['noSynergiser']
        Visionary=resp[0]['noVisionary']
                #is Teamwork displayed or not
        leadership=resp[0]['leadership']      
        
        #no of times Teamwork is displayed
        noleadership=resp[0]['noleadership']
        
        
        
        neg=resp[0]['NegativeBehavior']
        #check for signs of negative behavior is being displayed
        if len(neg)==0:
            neg="No neg behavior"
        behavior=resp[0]['ReportCard'] 
        #check if behavior is being displayed
        if len(behavior)==0:
        
            behavior="No Positive Behavior"
        defines=resp[0]['Defines']
        context=resp[0]['context']
        strength=resp[0]['strength']
        areaForImprovement=resp[0]['AreaForImprovement']
        role=resp[0]['role']

        strengthImprovement=resp[0]['strengthImprovement']
        blindspot=resp[0]['blindspot']
        
        #if len(context)==0:
        #  context="No Task assigned"
        overall=resp[0]['OverallResults']
        if len(overall)==0:
            overall="unable to predict softskill"
        #Defines=resp[0]['Defines']
        #BCs=resp[0]['BCs']  
        #Softskills=resp[0]['Softskills']  
        #upon post command is detected, this will call out the graph
        #return render_template('response.html',progress=22,today=today,reflection=reflection, neg=neg,behavior=behavior,context=context,softskill=overall,dictionary=resp,Teamwork=Teamwork/8,Communication=Communication/4,Adaptability=Adaptability/5,willingnesstolearn=willingnesstolearn/4,leadership=leadership/5,Adventurous=Adventurous,Analytical=Analytical,ConsensusBuilder=ConsensusBuilder,Imaginative=Imaginative,PlanExecutor=PlanExecutor,Resilient=Resilient,TargetSetter=TargetSetter,Technical=Technical,Clarity=Clarity,Considerate=Considerate,Credible=Credible,Courteous=Courteous,CoachabilityTeachability=CoachabilityTeachability,Humble=Humble,IntellectuallyCurious=IntellectuallyCurious,ResponsibleForOneOwnGrowth=ResponsibleForOneOwnGrowth,Curiosity=Curiosity,FutureOriented=FutureOriented,OpenMinded=OpenMinded,Resourceful=Resourceful,RiskTaker=RiskTaker,Empathetic=Empathetic,Empowering=Empowering,OneofIntegrityAndFairness=OneOfIntegrityAndFairness,Synergiser=Synergiser,Visionary=Visionary)
        reportNoOfFreq=resp[0]['reportNoOfFreq']
        headerImg=os.path.join('static', 'header2.JPG')
        leadershipImg = os.path.join('static', 'leadership.jpg')
        teamworkImg = os.path.join('static', 'teamwork.jpg')
        willingnesstolearnImg = os.path.join('static', 'willingnesstolearn.jpg')
        adaptabilityImg = os.path.join('static', 'adaptability.jpg')
        communicationImg = os.path.join('static', 'communication.jpg')
        return render_template('response.html',progress=22,headerImg=headerImg,leadershipImg=leadershipImg, teamworkImg=teamworkImg, willingnesstolearnImg= willingnesstolearnImg,adaptabilityImg=adaptabilityImg, communicationImg=communicationImg,defines=defines,strength=strength, areaForImprovement=areaForImprovement,role=role,strengthImprovement=strengthImprovement,blindspot=blindspot,reportNoOfFreq=reportNoOfFreq,resp=resp,today=today,reflection=reflection, neg=neg,behavior=behavior,context=context,softskill=overall,dictionary=resp,noTeamwork=noTeamwork,noCommunication=noCommunication,noAdaptability=noAdaptability,nowillingnesstolearn=nowillingnesstolearn,noleadership=noleadership,Teamwork=Teamwork/8,Communication=Communication/4,Adaptability=Adaptability/5,willingnesstolearn=willingnesstolearn/4,leadership=leadership/5,Adventurous=Adventurous,Analytical=Analytical,ConsensusBuilder=ConsensusBuilder,Imaginative=Imaginative,PlanExecutor=PlanExecutor,Resilient=Resilient,TargetSetter=TargetSetter,Technical=Technical,Clarity=Clarity,Considerate=Considerate,Credible=Credible,Courteous=Courteous,CoachabilityTeachability=CoachabilityTeachability,Humble=Humble,IntellectuallyCurious=IntellectuallyCurious,ResponsibleForOneOwnGrowth=ResponsibleForOneOwnGrowth,Curiosity=Curiosity,FutureOriented=FutureOriented,OpenMinded=OpenMinded,Resourceful=Resourceful,RiskTaker=RiskTaker,Empathetic=Empathetic,Empowering=Empowering,OneofIntegrityAndFairness=OneOfIntegrityAndFairness,Synergiser=Synergiser,Visionary=Visionary)
    
    #this will be the default page that will be displayaed once thie function is called
    return render_template("main.html")






#loads  the server at port 80 http
if __name__ == "__main__":
    app.run(port=80)
  