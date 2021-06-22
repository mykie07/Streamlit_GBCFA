from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

from sklearn.feature_extraction.text import TfidfTransformer

vect = CountVectorizer(max_df=1.0,stop_words='english')  

import pandas as pd
import joblib
import numpy as np

import os
java_path="C:/Program Files/Java/jdk1.8.0_251/bin/java.exe"
# java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path

home = r'stanford-postagger-2018-10-16'
from nltk.tag.stanford import StanfordPOSTagger as POS_Tag
from nltk import word_tokenize
_path_to_model = home + '/models/english-bidirectional-distsim.tagger' 
_path_to_jar = home + '/stanford-postagger.jar'
stanford_tag = POS_Tag(model_filename=_path_to_model, path_to_jar=_path_to_jar)


X_train=joblib.load('absapi_X_train_sentence_list.pkl')

vect = CountVectorizer(max_df=1.0,stop_words='english')  
X_train_count = vect.fit_transform(X_train)


lbs=['HARDDRIVE', 'BATTERY', 'CHARGER', 'CONNECTIVITY', 'CPU', 'DISPLAY',
       'FAN', 'KEYBOARD', 'LAPTOP', 'MEMORY', 'MOTHERBOARD', 'MULTIMEDIA',
       'OPTICAL_DRIVE', 'OS', 'PERFORMANCE', 'SHIPPING', 'SUPPORT']



def pred_engine():
    print("absapi_aspect_class_prediction_engine.py loaded")




#Filter the word with tag- noun,adjective,verb,adverb
def filterTag(tagged_review):
    final_text_list=[]
    for text_list in tagged_review:
        final_text=[]
        for word,tag in text_list:
            if tag in ['NN','NNS','NNP','NNPS','RB','RBR','RBS','JJ','JJR','JJS','VB','VBD','VBG','VBN','VBP','VBZ']:
                final_text.append(word)
        final_text_list.append(' '.join(final_text))
    return final_text_list




#To tag using stanford pos tagger
def posTag(review):
    tagged_text_list=[]
    for text in review:
        tagged_text_list.append(stanford_tag.tag(word_tokenize(text)))
    return tagged_text_list




def get_aspects(predict_aspect):
#     x=np.where(predict_aspect[[0]]==1)
    x=np.where(predict_aspect[[0]]==1) 
    # x
    aspects=[]
    for t in x[1]:
        l = lbs[t]
        aspects.append(l)

    return aspects    # print(lbs[t])


def intersection(lst1, lst2): 
   # Use of hybrid method 
    temp = set(lst2) 
    matches = [value for value in lst1 if value in temp] 
    return matches 



def mget_predict_sentence_aspect(model,input_sentences):
    for sent in input_sentences:
        tagged_user_input = posTag([sent])
        filter_tagged_user_input = filterTag(tagged_user_input)

        user_input_series=pd.Series(filter_tagged_user_input)
        user_input_series_dtm=vect.transform(user_input_series)

#         predict_aspect= nb_classif.predict(user_input_series_dtm)
        
#         get_aspects(predict_aspect)

        predict_aspect= model.predict(user_input_series_dtm)
        print("Sentence: ", sent)
        # print(predict_aspect)
        print("******Predicted Aspects/features********")
        
#         print(predict_apsect[0])
#         predict_aspect=predict_aspect.toarray()
        print(predict_aspect[0])
        aspects = get_aspects(predict_aspect)
        # print("aspects are :", aspects)

def mainget_predict_sentence_aspect(model,sent):
    # for sent in input_sentences:
    tagged_user_input = posTag([sent])
    filter_tagged_user_input = filterTag(tagged_user_input)

    user_input_series=pd.Series(filter_tagged_user_input)
    user_input_series_dtm=vect.transform(user_input_series)

#         predict_aspect= nb_classif.predict(user_input_series_dtm)
    
#         get_aspects(predict_aspect)

    predict_aspect= model.predict(user_input_series_dtm)
    # print("Sentence: ", sent)
    # print(predict_aspect)
    # print("******Predicted Aspects/features********")
    
#         print(predict_apsect[0])
#         predict_aspect=predict_aspect.toarray()
    # print(predict_aspect[0])
    aspects = get_aspects(predict_aspect)
    # print("aspects are :", aspects)    
    return aspects
  





# ABSTAIN = -1
BATTERY=["battery","recharging","battries",
                                                     "recharge","power unit","power_unit",
                                                     "electric cell","electric_cell","cell","cells",
                                                     "electric_battery","power cell"]
CHARGER=["charger","battery charger", "battery-charger"]
CONNECTIVITY=["connectivity","rj-45","rj45","ethernet", 
                                                               "ethernet adapter",
                                                               "ethernet port","wired network connection", 
                                                                 "network connection", "bluetooth","wireless",
                                                               "wifi","wi-fi","wan","lan","usb","port","ports",
                                                               "connector","connecter","connect"]
CPU=["cpu","processing unit","processor", "central processing unit","i3","i5","i7"]
DISPLAY=["display","IPS","HD","picture quality","monitor", 
                                                     "screen","touch screen", "touchscreen", 
                                                     "pixel","lcd", "resolution","video","11.6" ,
                                                     "12.5","13.3","14","15.6", "17.3", "18.4","11.6 inches" ,
                                                     "12.5 inches","13.3 inches","14 inches",
                                                     "15.6 inches", "17.3 inches", "18.4 inches",
                                                     "11.6\"" ,"12.5\"","13.3\"","14\"","15.6\"", "17.3\"", "18.4\""]
FAN=["fan","cooling system","cpu fan","fans","heating",
                                             "cooling","heat up","cool down","cool off","temperature","temperature reduction"]
HARDDRIVE=["harddrive","hard drive",
                                                         "hard drives","disc","disk",
                                                         "hard disc","hard disc","solid state",
                                                         "solid-state","hdd","ssd","magnetic disc",
                                                         "magnetic disc"]
KEYBOARD=["keyboard","typing","keys","key boards","key board"]
LAPTOP=["laptop","chromebook","computer","pc","macbook","chromebooks","machine","product"]
MEMORY=["memory","memory chip","ram","random access memory", "memory board", "memory card","memory stick"]
MOTHERBOARD=["motherboard","mother board","mother-board","mother_board","cpu-board","cpu board","cpu_board"]
MULTIMEDIA=["multimedia","sound","audio","aux","audio jack",
                                                           "loud","speaker","headphone","earphone","headset","headphones",
                                                           "earphones","head phone","ear phone","head set",
                                                           "head phones","ear phones","mp3","mp4","mpeg","midi"]
OPTICAL_DRIVE=["optical drive","optical drives","dvd","dvd drive","dvd/cd", "cd_rom_drive","cd rom",
                                                          "cd rom drive","burner",
                                                          "videodisk","videodisc","superdrive",
                                                          "compact disk", "compact disc"]
OS=["os","operating system","operating_system","window 8","windows","linux","mac os","ubuntu","fedora","chrome os"]
PERFORMANCE=["performance","speed","fast","slow"]
SHIPPING=["shipping","delivery"]
SUPPORT=["support","tech support","tech-support","technical support","call centre","call center",
 "contact center","contact centre","answering service", "answering services"
,"helpline","hotline","troubleshooting","troubleshoo","help desk","help line","after-sale service","service"]


aspect_classes=["OPTICAL_DRIVE","DISPLAY","MEMORY","BATTERY","FAN","MOTHERBOARD","HARDDRIVE","SHIPPING","CPU ","CHARGER","SUPPORT",
 "KEYBOARD","OS","LAPTOP","PERFORMANCE","CONNECTIVITY","MULTIMEDIA"]