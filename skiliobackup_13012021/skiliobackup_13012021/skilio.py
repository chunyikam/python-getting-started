# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 23:41:50 2020
this is the main codes for production 

@author: chuny
"""



import enchant
import pandas as pd
import nltk
import os
import re
import numpy as np
import datetime

import json
from nltk.corpus import wordnet, words
#from dateutil.parser import parse
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.util import ngrams
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from pattern.en import lemma
from nltk.corpus import wordnet as wn

import inflect
p= inflect.engine ()

pd.options.mode.chained_assignment = None

######## Behavior abstraction##########################
def isSynonyms(text): #to check  NNP is a name 
    from nltk.corpus import wordnet
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.util import ngrams
    #text='corporate with others'
    #text=str(text)
    lem = WordNetLemmatizer()
    synonyms=list()

    for syn in wordnet.synsets(text):
        for lemma in syn.lemmas():
            #synonyms.append(lemma.name())
            synonyms.append(lem.lemmatize(lemma.name(),"v"))
    synonyms=set(list(synonyms))
    return synonyms


#extract verb and noun  
def extractBcText(text): 
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    stopwords=stopwords.words('english')+["task","also"]
    text=text.lower()
    #d = enchant.Dict("en_US") 
    d = enchant.Dict("en_GB") 
    lem = WordNetLemmatizer()
    token=nltk.word_tokenize(text)
    pos=pos_tag(token)
    verb=[]
    noun=[]
    verbw=""
    nounw=""
    bc=[]
    #stopwords=["was","is","are","were"]
    #negative=["not","didn't","no", "can't"]
    for (word,tag) in pos:
     #word=en.verb.present(word)
     if word not in stopwords:
        
         #word=p.singular_noun(word)
        if tag.startswith("VB") and d.check(word)==True and len(word)>1 and word not in stopwords:
            verb.append(word)
            verbw= lem.lemmatize(word,"v")
        elif tag.startswith("NN")and d.check(word)==True and tag!="FW" and len(word)>1 and word not in stopwords:
            if not tag.startswith("NNP"):
                noun.append(word)
                nounw=lem.lemmatize(word,"v")
        if len(nounw)>1 and len(verbw)>1: 
            verbw=lemma(verbw)
            nounw=lemma(nounw)
            string=verbw+" "+nounw
            bc.append(string)
    return bc



#extract trigram
def extractBigram(text): #bigram
    from nltk.corpus import stopwords
    stopwords=stopwords.words('english')+["task","everyone","him","her","also"]
    #print(stopwords)
    lem = WordNetLemmatizer()
    d = enchant.Dict("en_US") 
    text=text.split()
    #print(sent_tokenize(str(text)))
    modal=["must","was","has","have","had","are","is","am","be","may","can","will","cannot","should","use","could"]
    text = [w for w in text if not w in stopwords]
    text=pos_tag(text)
    #print(text)
    phases=[]
    phase=" "           
    for(w1,t1),(w2,t2)in nltk.bigrams(text):
        #print(w1,t1,w2,t2)
        if((t1.startswith("NN") and t2.startswith("RB"))or(t1.startswith("VB") and t2.startswith("RB")) or (t1=="NNP" and t2.startswith("VB"))or (t1=="NNP" and t2.startswith("NN")) or (t1=="NN" and t2.startswith("NNP"))or (t1.startswith("VB") and t2.startswith("NN"))) :   
            #if((t1=="NNP" and t2.startswith("VB"))or (t1=="NNP" and t2.startswith("NN")) or (t1.startswith("VB") and t2.startswith("NN"))) :   
            if d.check(w1)==True and d.check(w2)==True and len(isSynonyms(w1))>1 and len(isSynonyms(w2))>1:
                if w1 not in stopwords and w2 not in stopwords:
                    w1=w1.strip()
                    w2=w2.strip()
                    if w1!=w2:
                        w1=lem.lemmatize(w1,"v")
                        w2=lem.lemmatize(w2,"v")
                        w1=lemma(w1)
                        w2=lemma(w2)
                        phase= w1+" "+w2
                        phases.append(phase)               
    return phases

#extract trigram 
def extractTrigram(text):
    from nltk.corpus import stopwords
    stopwords=stopwords.words('english')+["task","also"]

    lem = WordNetLemmatizer()
    d = enchant.Dict("en_US") 
    #modal=["must","was","has","have","had","are","is","am","be","may","might","can","will","cannot","should","use","could","do","does"]
    text=text.split()
    text = [w for w in text if not w in stopwords]
    text=pos_tag(text)
    phase=" "
    phases=[]
    for(w1,t1),(w2,t2),(w3,t3) in nltk.trigrams(text):
        if (t1.startswith("VB")and t2.startswith("IN") and t3.startswith("NN")) and (t1.startswith("VB")and t2.startswith("JJ") and t3.startswith("NN")) or (t1.startswith("VB")and t2.startswith("NN") and t3.startswith("RB")):  
            if len(isSynonyms(w1))>1 and len(isSynonyms(w2))>1 and len(isSynonyms(w3))>1 and d.check(w1)==True and d.check(w2)==True and d.check(w2)==True: 
               if(w1 not in stopwords) and (w2 not in stopwords) and (w3 not in stopwords):
                   w1=lem.lemmatize(w2,"v") 
                   w2=lem.lemmatize(w2,"v")
                   w3=lem.lemmatize(w3,"v")
                   if w1!=w2 and w2!=w1 and w3!=w1 and w2!=w3:
                       phase= w1+" "+w2+" "+w3 
                       phases.append(phase)
     
    return phases

#extract unigram 
def extractUnigram(text):
    from nltk.corpus import stopwords
    stopwords=stopwords.words('english')+["task"]

    d = enchant.Dict("en_US") 
    lem = WordNetLemmatizer()
    text=text.split()
    modal=["must","was","has","have","had","are","is","am","be","may","might","can","will","cannot","should","use","could","do","does"]
    text = [w for w in text if not w in stopwords]
    text=pos_tag(text)
    phase=[]
    #convert to bigram to extract the nnp and pronoun
    for (w1,t1),(w2,t2) in nltk.bigrams(text):
        #print(w1,t1)
        if (t1=="PRP" and t2.startswith("VB")) or (t1=="NNP" and t2.startswith("VB"))  :   
            if d.check(w2)==True and len(isSynonyms(w1))==0 and len(w1)>2:
                if w2 not in modal:#print(w1, w2)
                    phase.append(lem.lemmatize(w2,"v"))         
           #break;     
    for(w1, t1) in text:
        if t1.startswith("NN"):
            if d.check(w1)==True and len(isSynonyms(w1))>1 and len(w1)>2:   
                #print(isSynonyms(w1))
                if w1 not in modal:
                    phase.append(lem.lemmatize(w1,"v"))
                    break
        elif t1.startswith("VB"):
            if d.check(w1)==True and len(isSynonyms(w1))>1 and len(w1)>2:
                if w1 not in modal:
                    phase.append(lem.lemmatize(w1,"v")) 
                    break
    return phase

#function not in use
def isNegative(text):
    negative= ["nothing","negative","bad","disaster","error", "zero","no","intend","doesn't","isn't","wasn't","didn't","not", "doesn't","fail","want","wasn't"]
    lem = WordNetLemmatizer()
    text= text.split()
    marks=1
    for word in text :
        word=lem.lemmatize(word,"v")    
        if word in negative or word.endswith("?"):
            marks =marks*(-1)
        else: 
            marks =marks*(1)
    return marks 

def isNegativeNB(text):
    #directory="C:/nus/FYP/python/reference/model/"
    directory="C:/NUS/FYP/python/production/model/"
    
    filename = directory+'isNegativeNB.model'
    #filename = directory+'isNegativeXGB.model'
    #save it model directory
    bc=""
    modelType="isNegativeNB"
    #modelType="isNegativeXGB"
    filename=directory+bc+modelType+".model"
    nb_model = pickle.load(open(filename, 'rb'))
    #print(nb_model)
    count_vec=directory+bc+"count_vec"+modelType+".pickle"
    loaded_vectorizer = pickle.load(open(count_vec, 'rb'))
    tag= nb_model.predict(loaded_vectorizer.transform([text]))      
    return tag

def convertToString(text):
    text1=' '.join([text for text in text])
    return text1

    
    
def processBehavior(text): 
   # import re
    print(type(text))
    listString =["was able to","were able to", "am able to","managed to", "had to","have to","\n+","\t+","\r"]
    contrastWords=[" hence "," so ","whereas "," since "," but ","opposed to "," as "," although "," thus "," despite ","yet","on the other hand"]
    #consequencewords=["leading to",'as a result of',"consequently"]
    if("str" in str(type(text))):
        #   string="str"
        text=text.replace("[","")
        text=text.replace("]","")
        text=text.replace("\n",".")
        text=text.replace("re-","")
        text=text.replace(" - ","")
        text=text.replace(",",".")
        pattern = r'[0-9]'
        text=re.sub(pattern,"",str(text))
        text=re.sub("&","and",str(text))
        text=re.sub("/"," or ",str(text))
        text=re.sub("n't"," not",str(text))
        text=re.sub("'ve"," have",str(text))
        text=re.sub("m't"," am",str(text))
        for contrastWord in contrastWords:    
            text=re.sub(contrastWord,".",str(text))
        for reg in listString:
           # text=re.sub(reg,"",str(text))
           text=text.replace(reg,"")
        text=re.sub("  "," ",str(text))
        #text=re.sub(",""",". " ,str(text))
        text=re.sub("\"","",str(text))
        text =re.split("[(.!)]",text)
    else:
        text=text
        #text=convertToString(text)
        text =re.split("[(.!)].,",str(text))
    behavior=[]
    string2=""
    isNegativeBehavior=0
    isBehavior=0
    for text in text:
       # print(text)
        text=text.lower()
        text=text.strip()
        isNegativeBehavior=int(isNegativeNB(text))
        #if(isExhibitBehavior)==0:
        #if isBehavior>0:   
        if isNegativeBehavior>0:
            #print("Behavior",":",text,":",isNegativeBehavior)
        #isNegBehavior.append(string)
            isBehavior=1
            if isBehavior>0:   
#            if isNegativeBehavior>0:
                pattern=extractBcText(text)
                unigrams=extractUnigram(text)
                bigrams=extractBigram(text)
                trigrams=extractTrigram(text)
                if(len(pattern))>0:
                    behavior=behavior+pattern
                #behavior.append(pattern)
                if(len(bigrams))>0:
                    behavior=behavior+bigrams
                if(len(trigrams))>0:     
                    behavior=behavior+trigrams
                if(len(unigrams))>0:
                    behavior=behavior+unigrams
            
                #print("Positive behavior:",text+str(isNegativeBehavior))
                string2=string2+""
            else:
                text=text
                #print("Negative behavior:",text+str(isNegativeBehavior))
            #print("IS THIS A BEHAVIOR OR NOT:",text,":",str(isBehavior))
        else:
            string2=string2+"-"+text+"."
       
        #    print("Not a behavior:",text+str(isBehavior))
           # print("behavior not exihibited",isExhibitBehavior)
        #behavior=sum([extractUnigram(text),extractBigram(text),extractTrigram(text)],[])
        #print("Pos/negative thoughts or intention:",text+":",str(isNegativeBehavior))
        
    return list(set(behavior)),string2


def predictBehavior(modelDirectory, text, modelType,bc):
    #filename=modelName+".model"
    #directory="C:/nus/FYP/python/reference/model/"
    if bc!="Coachability/ teachability":
        filename=modelDirectory+bc+modelType+".model"
    else:
        bc="Coachability"
        filename=modelDirectory+bc+modelType+".model"
    nb_model = pickle.load(open(filename, 'rb'))
    #print(nb_model)
    count_vec=modelDirectory+bc+"count_vec"+modelType+".pickle"
    loaded_vectorizer = pickle.load(open(count_vec, 'rb'))
    
    tag= nb_model.predict(loaded_vectorizer.transform([text])) 
    #tagprobability=nb_model.predict_proba(loaded_vectorizer.transform([text]))
    #print(tag,np.average(tagprobability))
    return tag

#function not in use

def reportNeg(text):
    
    results=processBehavior(text)[1]

    return results

# function not in use
def reportProcessBehavior(text):
    results=processBehavior(text)[0]
    return results

#CAR statement
def report(text, directory,dictionary, modelDirectory, modelType):
    from collections import Counter
    #text=processBehavior(text)[0]

    #behaviors=reportProcessBehavior(text)
    behaviors=processBehavior(text)[0]
    #bcs=listOfBc(directory,dictionary)
    df=pd.read_csv(directory+dictionary,sep=",")
    behavior=[]
    string=""
    overallBehaviorStr=""
    overallbehavior=[]
    softskills=[]
    #softskills=""
    bcs=['Empathetic', 'Future-oriented', 'Humble', 'Considerate', 'Coachability/ teachability', 'Consensus Builder', 'Open-minded', 'Resourceful', 'Resilient', 'Adventurous', 'Technical', 'Courteous', 'Synergiser', 'Clarity', 'Visionary', 'One of Integrity and Fairness', 'Intellectually curious', 'Risk-Taker', 'Target Setter', 'Empowering', 'Curiosity', 'Credible', "Responsible for one's own growth", 'Analytical', 'Plan Executor', 'Imaginative']
    strSoftskills=[]
    strTags=[]
    strDefines=[]
    strSoftskills=""
    strTags=""
    strDefines=""
    tags=[]
    for behaviors in behaviors:    
        for bc in bcs:
            softskill=""
            tag= predictBehavior(modelDirectory, behaviors, modelType,bc)
            softskill=df[df.BehaviorCompetency.isin([bc])]['softskill'].unique()
            define=""
            if tag!="others":
                print(softskill)
                #string=string+"You"+" "+behaviors+" showing signs of "+str(tag)+",implying that you are likely to meet"+str(softskill)+".\n"
                #behaviors
                strSoftskill=str(softskill).replace("[","")
                strSoftskill=strSoftskill.replace("]","")
                strSoftskill=strSoftskill.replace("'","")
                strTag=str(tag).replace("[","")
                strTag=strTag.replace("]","")
                strTag=strTag.replace("'","")
                if tag=="Humble":
                    strTag=""+"Humility"+""
                
                #softskill
                #define=definition(tag)
                define=definition(strTag)[0]
                print("definition:",define)
                behaviors=convertToAdverb(behaviors)
                #string=string+"- "+str(strSoftskill).title()+": You "+str(define)+", displaying the behavior competency "+strTag+" by "+behaviors+".\n"
                string=string+"- "+str(strTag).title()+": "+str(define)+" by "+behaviors+".\n"

                #df[df['softskill']==softskill]['BehaviorCompetency']
                                #print(string)
                #print(text,":",behaviors,":", tag)
                #print(text,":",tag)
                overallBehaviorStr="- "+str(strSoftskill).title()+": "+tag
                behavior.extend(overallBehaviorStr)
                overallbehavior.extend(tag)
                #strSoftskills.extend(softskill)
                #strTags.extend(tag)
                #strDefines.extend(define)
                strSoftskills=strSoftskills+str(strSoftskill)+","
                strTags=strTags+str(strTag)+","
                #strDefines=strDefines+str(define)+","
                tags.extend(tag)
            #behavior.extend(tag) 

    print(list(Counter(tags).keys()))
    print(list(Counter(tags).values()))
    keys=list(Counter(tags).keys())
    values=list(Counter(tags).values())
    strSummary=""
    strStrength=""
    strAreaForImprovement=""
    strRole=""
    strStrengthImprovement=""
    strBlindspot=""
    for i in range(len(keys)):
        softskill=df[df.BehaviorCompetency.isin([keys[i]])]['softskill'].unique()
        strSoftskill=str(softskill).replace("[","")
        strSoftskill=strSoftskill.replace("]","")
        strSoftskill=strSoftskill.replace("'","")
        if keys[i]=="Humble":
           keys[i]="Humility" 
        #definition,strength,areaForImprovement,role,strengthImprovement,blindspot
        define=definition(keys[i])[0]
        strength=definition(keys[i])[1]
        areaForImprovement=definition(keys[i])[2]
        role=definition(keys[i])[3]
        strengthImprovement=definition(keys[i])[4]
        blindspot=definition(keys[i])[5]
        
        print(strSoftskill,":",keys[i],":",define,":",values[i])
        strSummary=strSummary+"\n"+str(strSoftskill).title()+":"+str(keys[i])+":"+str(define)+":"+str(values[i])+"."
        strDefines=strDefines+str(define)
        strStrength=strStrength+str(strength)
        strAreaForImprovement=strAreaForImprovement+str(areaForImprovement)
        strRole=strRole+str(role)
        strStrengthImprovement=strStrengthImprovement+str(strengthImprovement)
        strBlindspot= strBlindspot+str(blindspot)
    behaviorStr=""
    for behavior in set(list(behavior)):
        if len(behavior)>1:
            behaviorStr=behaviorStr+behavior+"."
    #overall="<p class=\"round1\"<BR><B>Survival softskill met:</B><BR>"+str(set(list(behavior)))+"</p><BR><BR>"
    overall=behaviorStr  
    #string=string+overall
    print(string)
    return overallbehavior,string,overall,strSoftskills,strTags,strDefines,strSummary,strStrength,strAreaForImprovement,strRole,strStrengthImprovement,strBlindspot


def reportCard(text,directory,dictionary,modelDirectory,modelType):
    results=report(text,directory,dictionary,modelDirectory,modelType)[1]
    return results
   # return dictionary

def reportInterRater(source,directory,dictionary, modelDirectory, modelType):
    ahs=pd.read_csv(source,sep=",")
    reflections=ahs["Reflection"]
    for reflection in reflections:      
        result=extractResult(directory,dictionary, modelDirectory, reflection, modelType)
        #result=extractResult(reflection,df)
    return result
                  
def listOfBc(directory,dictionary):
    df=pd.read_csv(directory+dictionary)
    df=df[["BehaviorCompetency"]]    
    df=df[df["BehaviorCompetency"]!="others"]
    bcs=df["BehaviorCompetency"].unique()
    return set(list(bcs))   

#extract entity such a location. funtion not in use
def extractEntity(text):
    import nltk
    from nltk.corpus import stopwords
    stopwords=stopwords.words('english')+["us"]
    text=text.split()
    text = [w for w in text if w not in stopwords]
    print(text)
    text=pos_tag(text)
  #  bigram=nltk.bigrams(text)  
    entity=""
    pgm=[]         
    for(w1,t1),(w2,t2) in nltk.bigrams(text):
        if(t1.startswith("NN") and t2.startswith("NN")): 
            #if len(isSynonyms(w1))==0 and len(isSynonyms(w1))==0:
            entity= w1+" "+w2 
            pgm.append(entity)                  
    return pgm

#converb the first word of behavior to present participle
def convertToAdverb(text):
  
    from pattern.en import conjugate, lemma, lexeme,PRESENT,SG
    textToList=text.lower().split()   
    print(len(textToList))
    if len(textToList)>=2:
        verb=textToList[0]
        vbg=lexeme(verb)[2]+" "+textToList[1]
    else:
        #print("error")
        vbg=lexeme(text)[2]
    return vbg
  
#convertToAdverb("ask question")

#extract event the student is doing         
def extractEvent3(text):
    import nltk
    from nltk.corpus import stopwords
    stopwords=stopwords.words('english')
    stopword =[word for word in stopwords if word not in["during","at","for","with","to","in"]]
   
    
    text=text.split()   
    text1=[]
    for text in text :
        #text=text.lower()
        if text.lower() not in stopword :
            text1.append(text)
    print(text1)
    text1=pos_tag(text1)
    string=[]
    phase=""
    for(w1,t1),(w2,t2),in nltk.bigrams(text1):
        if((t1=="IN" and t2.startswith("NNP"))):   
           w1="Activity:"
           phase=w1+" "+w2
           if len(phase)>0:
               string.append(phase)
        if((t1=="TO" and t2.startswith("NN"))):   
           w1="Task:"
           phase=w1+" "+w2
           if len(phase)>0:
               string.append(phase)
            
    for(w1,t1),(w2,t2),(w3,t3)in nltk.trigrams(text1):
        if (t1=="IN" and t2.startswith("NNP") and t3.startswith("NNP")):  
       # if (t1=="IN" and t2.startswith("NNP") and t3.startswith("NNP")):   
       #if(t1=="TO" and t2.startswith("VB") and t3.startswith("NN")):
            w1="Activity:"
            phase=w1+" "+w2+" "+w3
            if len(phase)>0:
               string.append(phase) 
   #for(w1,t1),(w2,t2),(w3,t3)in nltk.trigrams(text1):
       # phase=w1+" "+w2+" "+w3
       # print(phase)

        if (t1=="TO" and t2.startswith("VB") and t3.startswith("NN")): 
            w1="Task:"
            phase=w1+" "+w2+" "+w3
            if len(phase)>0:
               string.append(phase) 
    print(type(string))
    
    print(' '.join(string))
    return string
    #return ' '.join(string)


#extract context
def context(text):
    import re
    
    if("str" in str(type(text))):
        string="str"
        text =re.split("[-.?!,]",str(text))
    else:
        string="list"
        text =re.split("[-.?!,]",str(text))
    context=[]    
    for text in text:
        if isNegativeNB(text)>0:
            #if len(extractEntity(text))>0:
             #   context+=extractEntity(text)
        #context.append(extractEntity(text))
            if len(extractEvent3(text))>0:
                context+=extractEvent3(text)
        #context.append(extractEvent3(text))
        #print(list(set(context)))
    return ' '.join(context)
    #return list(set(context))


def definition(text):
    text=str(text)
    text=text.replace("'","")
    #text=''.join(filter(str.isalnum, text))
    text=text.lower()
    definition="" 
    strength=""
    areaForImprovement=""
    role=""
    strengthImprovement=""
    blindspot=""
    if text=="risk-taker".lower():
        definition="willingly took risks in hope of desired results"
        strength="- Willingly take risks in hope of desired results."
        
        areaForImprovement="- Willingly take risks in hope of desired results"
        role=""
        strengthImprovement=""""- Try to adopt different solutions each time to test respective effectiveness.
                               - Be comfortable with doing things different from before.
                                - Systematically eliminate risks by identifying risks which bring the worst consequences first."
                                """
        blindspot=""""- Try to adopt different solutions each time to test respective effectiveness.
                     - Be comfortable with doing things different from before.
                    - Systematically eliminate risks by identifying risks which bring the worst consequences first."
                   """
    elif text=="resourceful".lower():
        definition="thought creatively, generated ideas and identified alternatives"
        strength ="- Thought creatively, generated ideas and identified alternative." 
        areaForImprovement="""- Think creatively, generated ideas and identified alternatives."""
        role=""
        strengthImprovement="""- Expand your network of stakeholders to identify their needs and seek resources to complete you task more efficiently.
                            - Seek to understand your issue better and where you can potentially find your resources."""
        blindspot="""
        - Expand your network of stakeholders to identify their needs and seek resources to complete you task more efficiently.
        - Seek to understand your issue better and where you can potentially find your resources."""
    elif text=="future-oriented".lower(): 
        definition="have been able to anticipate and plan ahead"
        strength="- Been able to anticipate and plan ahead."
        areaForImprovement="- Been able to anticipate and plan ahead."
        role=""
        strengthImprovement="""- Keep yourself up with current affairs.
                                - Be forward looking when making decisions.
                                - Be confident when making decisions, especially when they require immediate action."
                            """
        blindspot="""- Keep yourself up with current affairs.
                        - Be forward looking when making decisions.
                         - Be confident when making decisions, especially when they require immediate action.
                   """
    elif text=="Curiosity".lower():
        definition="had the desire to know more"
        strength="- Had the desire to know more"
        areaForImprovement="- Have the desire to know more."
        role=""
        strengthImprovement="""" - Seek to learn from understanding how things work.
                        - Explore various ways of achieving the same outcome, to evaluate the best option."""
        blindspot="""- Seek to learn from understanding how things work.
                    - Explore various ways of achieving the same outcome, to evaluate the best option."""
    elif text=="open-minded".lower():
        definition="had been willing to consider novel ideas"
        strength="- Been willing to consider novel ideas"
        areaForImprovement="- Be willing to consider novel ideas."
        role=""
        strengthImprovement=""" - Encourage ideas which challenge status quo.
        - Do not use past way of doing things as benchmarks to formulate solutions"
            """
        blindspot="""- Encourage ideas which challenge status quo.
                - Do not use past way of doing things as benchmarks to formulate solutions."""
    #leadership
    elif text=="synergizer".lower():
        definition="facilitated teams to be become better than the sum of its parts."
        strength="- Facilitate teams to be become better than the sum of its parts."
        areaForImprovement="- Facilitate teams to be become better than the sum of its parts."
        role=""
        strengthImprovement="""- Encourage team members to look out for other team members.
                    - Remind team of the bigger goal that it has to achieve together."""
        blindspot="""- Encourage team members to look out for other team members.
                    - Remind team of the bigger goal that it has to achieve together."""
    elif text=="one of integrity and fairness".lower():
        definition="have been honest for the betterment of the teams"
        strength="- Been honest for the betterment of the team."
        areaForImprovement="- Be honest for the betterment of the team."
        role=""
        strengthImprovement="""- Be aware of personal biases and not let them affect decision making.
        - Seek second opinion to ensure impartiality in decision making.
        - Explain rationale behind why certain decisions are made before executing them.
        - Keep team informed of decisions made."""
        blindspot="""- Be aware of personal biases and not let them affect decision making.
        - Seek second opinion to ensure impartiality in decision making.
        - Explain rationale behind why certain decisions are made before executing them.
        - Keep team informed of decisions made."""
    elif text=="visionary".lower():
        definition="thought and planned for what a team should work towards with wisdom"
        strength="- Thought and planned for what a team should work towards with wisdom."
        areaForImprovement="- Think and plan for what a team should work towards. "
        role=""
        strengthImprovement="""- Break down goals into smaller actionable steps for members to attain.
        - Set high benchmarks for quality of work produced.
        - Set key performance indicators which are measurable."""
        blindspot="""- Break down goals into smaller actionable steps for members to attain.
                    - Set high benchmarks for quality of work produced.
                    - Set key performance indicators which are measurable."""
    elif text=="empowering".lower(): 
        definition="have given people the power and authority to do something"
        strength="- Given people the power and authority to do something."
        areaForImprovement="- Give people the power and authority to do something. "
        role=""
        strengthImprovement="""- Seek to understand what members need and provide what they need.
                                - Celebrate achievements even when they are small wins.
                                - Provide sufficient guidance for members to reach for goals that they set for themselves.
                                - Encourage members to reach for goals which are higher than what they have achieved before but attainable nonetheless.
                                - Allow members to do what they are good at for most of the time."""
        blindspot="""- Seek to understand what members need and provide what they need.
                                - Celebrate achievements even when they are small wins.
                                - Provide sufficient guidance for members to reach for goals that they set for themselves.
                                - Encourage members to reach for goals which are higher than what they have achieved before but attainable nonetheless.
                                - Allow members to do what they are good at for most of the time."""
    elif text=="Empathetic".lower(): 
        definition="had the ability to understand and share the feelings of others"
        strength="- Have the ability to understand and share the feelings of others."
        areaForImprovement="- Try to understand and share the feelings of others. "
        role=""
        strengthImprovement="""- Initiate conversations to understand members better.
                                - Do not and discourage judging or blaming people.
                                - Encourage authentic conversations and vulnerability to cultivate team trust."""
        blindspot="""- Initiate conversations to understand members better.
                                - Do not and discourage judging or blaming people.
                                - Encourage authentic conversations and vulnerability to cultivate team trust."""
   
    #communication 
    
    elif text=="courteous".lower(): 
        definition="have been polite, respectful, or considerate to others"
        strength="- Been polite, respectful, or considerate to others."
        areaForImprovement="- Be polite, respectful, or considerate to others."
        role=""
        strengthImprovement="""- Use cues to show others that you are listening actively.
        - Interact and engage with your audience when you are addressing them."""
        blindspot="""- Use cues to show others that you are listening actively.
        - Interact and engage with your audience when you are addressing them."""
    elif text=="clarity".lower():
        definition="have been easily understood by others"
        strength="- Be easily understood by others"
        areaForImprovement="- Be easily understood by others."
        role=""
        strengthImprovement="""- Structure your presentation clearly to enhance understanding.
        - Only say the things you would want to hear as part of the audience."""
        blindspot="""- Structure your presentation clearly to enhance understanding.
        - Only say the things you would want to hear as part of the audience."""
    elif text=="considerate".lower():
        definition="have shown careful thought".lower()
        strength="- Shown careful thought."
        areaForImprovement="- Show careful thought."
        role=""
        strengthImprovement="""- Do sufficient research on your audience background before preparing for your presentation.
        - Tailor your presentation according to your audience reaction.
        - Be empathetic to your audience needs and say things only if they are audience appropriate."""
        blindspot="""- Do sufficient research on your audience background before preparing for your presentation.
                    - Tailor your presentation according to your audience reaction.
                    - Be empathetic to your audience needs and say things only if they are audience appropriate."""

    elif text=="credible".lower():
        definition="came across as believable and convincing".lower()
        strength="- Came across as believable and convincing."
        areaForImprovement="- Come across as believable and convincing."
        role=""
        strengthImprovement="""- Do sufficient research and use them to justify your claims.
                                - Reduce the use of technical jargon to allow your audience to understand your presentation without background knowledge.
                                - Be mindful of the use of appropriate language and tone.
                                - Note your body language and emotion you wish to convey through your presentation."
                                """
        blindspot="""- Do sufficient research and use them to justify your claims.
                                - Reduce the use of technical jargon to allow your audience to understand your presentation without background knowledge.
                                - Be mindful of the use of appropriate language and tone.
                                - Note your body language and emotion you wish to convey through your presentation."""
    ### Teamwork
    elif text=="imaginative".lower():
        definition="have been creative and good at solving problems in unconventional ways".lower()
        strength="- Have been creative and good at solving problems in unconventional ways."
        areaForImprovement="- Solve problems in unconventional ways."
        role="- Solve problems in unconventional ways"
        strengthImprovement="""- Be unafraid to generate ideas when solving difficult problems.
            - Consider the feasibility of your ideas.
            - Think of ways to clearly communicate your ideas to others."""
        blindspot="""- Be unafraid to generate ideas when solving difficult problems.
            - Consider the feasibility of your ideas.
            - Think of ways to clearly communicate your ideas to others."""
    ### Teamwork
    elif text=="adventurous".lower(): 
        definition=" have been willing to try new methods,ideas or experiences".lower()
        strength="- Been willing to try new methods, ideas or experiences."
        areaForImprovement="- Be willing to try new methods, ideas or experiences."
        role="- Be willing to try new methods, ideas or experiences."
        strengthImprovement="""- Carry out extensive research to discover options available.
        - Embrace challenges as part and parcel of your experience.
        - See exploration as part of your learning experience.
        - Be willing to contribute and seek for new ideas when working with others."""
        blindspot="""- Carry out extensive research to discover options available.
        - Embrace challenges as part and parcel of your experience.
        - See exploration as part of your learning experience.
        - Be willing to contribute and seek for new ideas when working with others."""
    elif text=="target setter".lower(): 
        definition=" organised the team and ensure it worked effectively".lower()
        strength="- Organised the team and ensure it worked effectively."
        areaForImprovement="- Organise the team and ensure it works effectively."
        role="- Organise the team and ensure it works effectively."
        strengthImprovement="""- Develop systems and processes to delegate work among the team.
        -  Have consistent check-ins to ensure everyone is completing their tasks on time by asking and providing feedback."
        """
        blindspot="""- Develop systems and processes to delegate work among the team
        -  Have consistent check-ins to ensure everyone is completing their tasks on time by asking and providing feedback."
        """
    elif text=="Resilient".lower(): 
        definition="withstood and recovered quickly from difficult condition".lower()
        strength="- Withstood and recovered quickly from difficult conditions."
        areaForImprovement="- Withstand and recover quickly from difficult conditions."
        role="- Withstand and recover quickly from difficult conditions."
        strengthImprovement="""- See challenges as part and parcel of the learning process.
        - Deal with problems by reasoning logically one what is the next step forward.
        - Move on and do not be upset over something that cannot be changed."""
        blindspot="""- See challenges as part and parcel of the learning process.
        - Deal with problems by reasoning logically one what is the next step forward.
        - Move on and do not be upset over something that cannot be changed."""
    elif text=="Analytical".lower():
        definition="used logical reasoning to make decisions".lower() 
        strength="- Used logical reasoning to make decisions."
        areaForImprovement="- Use logical reasoning to make decisions."
        role="- Use logical reasoning to make decisions."
        strengthImprovement="""- Talk to more people to obtain input and brainstorm on what are possible ways to resolve issues as quickly as possible.
        - Make decisions based on future plans of the team.
        - Consider pros and cons of decisions in relation to goals."""
        blindspot="""- Talk to more people to obtain input and brainstorm on what are possible ways to resolve issues as quickly as possible.
        - Make decisions based on future plans of the team.
        - Consider pros and cons of decisions in relation to goals."""
    elif text=="Consensus Builder".lower(): 
        definition="allowed for agreement and understanding to be reached within a team".lower()
        strength="- Allowed for agreement and understanding to be reached within a team."
        areaForImprovement="- Allow for agreement and understanding to be reached within a team."
        role="- Allow for agreement and understanding to be reached within a team."
        strengthImprovement="""- Facilitate discussion to allow team to understand each others' perspectives before proceeding to make decisions.
        - Rationally resolve conflict by logically explaining areas for contention.
        - Putting oneself in others' shoes to understand why they are making certain stands.
        - Show your team that you are actively listening and trying to understand them.
        - Do not aim to avoid conflict, but deal with conflict rationally."""
        blindspot="""- Facilitate discussion to allow team to understand each others' perspectives before proceeding to make decisions.
        - Rationally resolve conflict by logically explaining areas for contention.
        - Putting oneself in others' shoes to understand why they are making certain stands.
        - Show your team that you are actively listening and trying to understand them.
        - Do not aim to avoid conflict, but deal with conflict rationally."""
    elif text=="Plan Executor".lower(): 
        definition="put ideas into action".lower()
        strength="- Put ideas into action."
        areaForImprovement="- Put ideas into action."
        role="- Put ideas into action."
        strengthImprovement="""- Set reasonable deadlines for team, taking into account buffer.
                                - Constantly check on progress of team and ask for feedback.
                                - Give each other feedback to maintain high quality of work produced.
                                - Set high benchmarks by producing quality work yourself.
                                """
        blindspot="""- Set reasonable deadlines for team, taking into account buffer.
                                - Constantly check on progress of team and ask for feedback.
                                - Give each other feedback to maintain high quality of work produced.
                                """
    elif text=="Technical".lower(): 
        definition="possessed the knowledge and expertise to accomplish a task".lower()
        strength="- Possessed the knowledge and expertise to accomplish a task."
        areaForImprovement="- Acquire the knowledge and expertise to accomplish a task."
        role="- Acquire the knowledge and expertise to accomplish a task."
        strengthImprovement="""- Read up and practice on skills you are already good in.
        - Be confident in proposing your ideas especially when they are in your field of expertise.
        - Be humble but not passive when sharing your knowledge with others."""
        blindspot="""- Read up and practice on skills you are already good in.
        - Be confident in proposing your ideas especially when they are in your field of expertise.
        - Be humble but not passive when sharing your knowledge with others."""
    #Willingness to Learn
    elif text=="Coachability".lower(): 
        definition="Have been easily taught and trained to do something better".lower()
        strength="- Been easily taught and trained to do something better."
        areaForImprovement="- Be willing to be taught and trained to do something better."
        role=""
        strengthImprovement="""- Be open to undertaking more responsibilities than before.
        - Take up things only when you see a true purpose for it.
        - Constantly seek ways to broaden your horizons in your area of interest.
        - Seek to discuss issues with others to gain more perspective."""
        blindspot="""- Be open to undertaking more responsibilities than before.
        - Take up things only when you see a true purpose for it.
        - Constantly seek ways to broaden your horizons in your area of interest.
        - Seek to discuss issues with others to gain more perspective."""
    elif text=="Intellectual Curiosity".lower(): 
        definition="Displayed the desire to learn and understand more".lower()
        strength="- Displayed the desire to learn and understand more."
        areaForImprovement="- Display the desire to learn and understand more."
        role=""
        strengthImprovement="""- Be willing to do things differently from before.
            - Embrace risks, but ensure you have plans to deal with them.
            - Ask questions to get more perspectives regarding a certain issue."""
        blindspot="""- Be willing to do things differently from before.
            - Embrace risks, but ensure you have plans to deal with them.
            - Ask questions to get more perspectives regarding a certain issue."""
    elif text=="Humility".lower(): 
        definition="Have not shown an excess amount of pride despite one's capabilities".lower()
        strength="- Not shown an excess amount of pride despite one's capabilities."
        areaForImprovement="- Not show an excess amount of pride despite one's capabilities."
        role=""
        strengthImprovement="""- Acknowledge the contributions of the people who have worked with you to make the experience possible.
                    - Be open to feedback and seek how to improve.
                    - Be humble to improve, but not passive in not taking action."""
        blindspot="""- Acknowledge the contributions of the people who have worked with you to make the experience possible.
                - Be open to feedback and seek how to improve.
                - Be humble to improve, but not passive in not taking action."""
    elif text=="Responsibility for One's Own Growth".lower(): 
        definition="- Have taken ownership of your improvement and growth".lower()  
        strength="- Taken ownership of your improvement and growth."
        areaForImprovement="- Take ownership of your improvement and growth."
        role=""
        strengthImprovement="""- Intentionally develop personal goals to attain excellence through actionable steps.
            - Build a network of people whom you can interact with and learn from."""
        blindspot="""- Intentionally develop personal goals to attain excellence through actionable steps.
            - Build a network of people whom you can interact with and learn from."""
    else:
        definition="Unable to locate definition".lower()  
        strength="Unable to detect strength"
        areaForImprovement="Unable to detect strength"
        role=""
        strengthImprovement="Unable to detect strength"
        blindspot="Unable to detect strength"
        definition="unable to locate definition"

    return definition,strength,areaForImprovement,role,strengthImprovement,blindspot


#Behavior to extract negative behavior
def isBehaviorNB(text):
    directory="C:/nus/FYP/python/reference/model/"
    filename = directory+'isNegativeNB.model'
    #filename = directory+'isNegativeXGB.model'
    #save it model directory
    bc=""
    modelType="isBehaviorNB"
    #modelType="isNegativeXGB"
    filename=directory+bc+modelType+".model"
    nb_model = pickle.load(open(filename, 'rb'))
    #print(nb_model)
    count_vec=directory+bc+"count_vec"+modelType+".pickle"
    loaded_vectorizer = pickle.load(open(count_vec, 'rb'))
    tag= nb_model.predict(loaded_vectorizer.transform([text]))      
    return tag

# main function to extract the result of the report

def extractResult(directory,dictionary, modelDirectory, text, modelType): 
    
    results=report(text,directory,dictionary, modelDirectory, modelType) 
    #feature extraction to return behavior text
    behavior=processBehavior(text)[0]
    #returns behavior that are negative
    neg=processBehavior(text)[1]
    #reads the dictionary to list out the softskill behavior 
    df=pd.read_csv(directory+dictionary)
    result3=results[0]
    result4=results[3]
    dfNew=pd.DataFrame()
    format = "%d/%m/%Y"
    today=str(datetime.datetime.now().strftime(format))
    dfNew["date"]=[today]
    dfNew["reflection"]=[text]
    #dfNew["behavior"]=dfNew["reflection"].apply(reportProcessBehavior)
    dfNew["behavior"]=[behavior]
    dfNew["NegativeBehavior"]=[neg]
    dfNew["context"]=dfNew["reflection"].apply(context)
    #dfNew["context"]=dfNew["reflection"].apply(extractEvent3)
    #strDefines,strSummary,strStrength,strAreaForImprovement,strRole,strStrengthImprovement,strBlindspot
    #dfNew["ReportCard"]=dfNew["reflection"].apply(reportCard,args=(directory,dictionary,modelDirectory,modelType))
    dfNew["ReportCard"]=[results[1]]
    dfNew["OverallResults"]=[results[2]]
    dfNew["BCs"]=[results[4]]
    dfNew["Softskills"]=[results[3]]
    dfNew["Defines"]=[results[5]]
    dfNew["reportNoOfFreq"]=[results[6]]
    dfNew["strength"]=[results[7]]
    dfNew["AreaForImprovement"]=[results[8]]
    dfNew["role"]=[results[9]]
    dfNew["strengthImprovement"]=[results[10]]
    dfNew["blindspot"]=[results[11]]
    softskills = df['softskill'].unique()
    behavior=[]
        
    for softskill in softskills:
        point =0
        nopoint=0
        #extract behaviior competency from the dictionary 
        behaviors = df[df['softskill']==softskill]['BehaviorCompetency'].unique()
        for behavior in behaviors:
            if behavior in result3:
                meet = 1
            else:
                meet = 0 
            print(behavior,":",meet)
            #checks if each behavior is met.
            dfNew[behavior]=[meet]
            #checks no of behavior dsplayed
            dfNew["no"+behavior]=[result3.count(behavior)]
            point = point+int(meet)
            nopoint=nopoint+int(dfNew["no"+behavior])
        #total score softskill
        
        dfNew[softskill]=[point]
        #total no of times softskill displayed
        dfNew["no"+softskill]=[nopoint]
        print(softskill,":",int(dfNew[softskill]))
        
        
    #stores data in json format
    dictionary=dfNew.to_dict('record') 
    file="record"+modelType+".csv" #report for verification purposes
    if os.path.isfile(file)==True:
         dfcsv=dfNew.to_csv(file, mode="a", index=None, header=False)
    else:
         dfcsv=dfNew.to_csv(file, index=None, header=True)
    #returns the json output
    return dictionary  
    




directory="C:/nus/FYP/dictionary/"
#dictionary="finalDict.csv"
#dictionary="finalfinaldictionary.csv"
#dictionary="finalfinalfinaldictionary.csv"
dictionary="finalfinaldictionary_07102020v1.csv"

#modelDirectory="C:/nus/FYP/python/reference/model/"
modelDirectory="C:/NUS/FYP/python/production/model/"

#modelDirectory="C:/nus/FYP/python/reference/model/balance/"
#modelDirectory="C:/nus/FYP/python/reference/model/unbalance/"



# codes to classify the AHS reflection. 

#source="C:/nus/FYP/python/production/compilation.csv"
#source="C:/nus/FYP/python/data/AHS.csv"
#source="C:/nus/FYP/python/data/skillfuture.csv"
#source="C:/nus/FYP/python/data/all_cleanup.csv"

#modelType="NB"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))

#modelType="LR"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))

#modelType="SVD"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))

#modelType="KNN"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))

#modelType="RF"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))

#modelType="XGB"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))

#modelType="NN"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))

#modelType="EnsembleNBXG"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))

#modelType="EnsembleNBXGLR"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))

#modelType="EnsembleNBXGSoft"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))

#modelType="EnsembleNBXGLRSoft"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))

#modelType="EnsembleNBXGLRKNNSoft"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))


#modelType="EnsembleStack"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))

#modelType="EnsembleStackRFLRKNNNBXGB"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))

#modelType="EnsembleStackXGBNBLRLR"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))


#modelType="EnsembleStackXGBNBLRNB" -Do not use
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))


#modelType="EnsembleStackXGBNBLRXGB"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))

#modelType="EnsembleStackAll"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))

#modelType="EnsembleStackNNSVCRFKNNXGBNBLR"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))

#modelType="EnsembleStackNNRFKNNXGBNBLR"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))

#modelType="NN_train"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))


#modelType="extraTree"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))


#modelType="AdaBoost"
#print(reportInterRater(source,directory,dictionary, modelDirectory, modelType))



# testing codes to classify single reflection
#text="""Our team was tasked with re-designing website and creating a database for Future of Work.
 #I realised that designing a website is not an easy task where I faced some challenges in doing it.
 #Some of them are user-friendly website and organising the content of the website.
 #My task was mostly re-designing the website of Future of Work (FOW). 
 #I also did some research on how to create a comprehensive database.
 #I did some research in the internet on how to create a great and user-friendly website. 
 #I also look at other website related to e-commerce which enables me to get ideas on designing the website. Moreover, I sought some opinions and ideas from my teams.I was able to design a user-friendly website, which allow users to understand who Future of Work (FOW) is."""

#modelType="EnsembleNBXGLRSoft"
#print("Model Type:",modelType)
#report(text, directory,dictionary, modelDirectory, modelType)[0]

#extractResult(directory,dictionary, modelDirectory, text, modelType) 

#modelType="KNN"
#print("Model Type:",modelType)
#report(text, directory,dictionary, modelDirectory, modelType)[0]

#modelType="RF"
#print("Model Type:",modelType)
#report(text, directory,dictionary, modelDirectory, modelType)[0]

#modelType="XGB"
#print(text)
#print("Model Type:",modelType)
#report(text, directory,dictionary, modelDirectory, modelType)[0]

#modelType="NB"
#print("Model Type:",modelType)
#report(text, directory,dictionary, modelDirectory, modelType)[0]

#modelType="NN"
#print("Model Type:",modelType)
#report(text, directory,dictionary, modelDirectory, modelType)[0]



#modelType="EnsembleNBXGLRSoft"
#print("Model Type:",modelType)
#report(text, directory,dictionary, modelDirectory, modelType)[0]
#extractResult(directory,dictionary, modelDirectory, text, modelTyrocee) 



#modelType="EnsembleNBXGLRSoft"
#report(text, directory,dictionary, modelDirectory, modelType)[0]

