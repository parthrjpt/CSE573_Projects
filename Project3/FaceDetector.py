# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:56:33 2021

@author: parth
"""

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import json
import pandas as pd
import os
import sys

def read_images(imgpath):
    
    imgpath = glob.glob(imgpath)
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    return imgs

def detect_faces(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    proc_img = img.copy()
    proc_imgs=[]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5,minSize=(30,30))
    for (x,y,w,h) in faces:
        cv2.rectangle(proc_img,(x,y),(x+w,y+h),(0,0,255),2)
        proc_img = proc_img[y:y+h,x:x+w]
        proc_imgs.append(proc_img)
    return proc_imgs,faces

def json_create(lstfaces,filenames,opfilename):
    json_list=[]
    for i in range(len(lstfaces)):
        #imgname = 'img_'+str((i+1))+'.jpg'
        imgname = filenames[i]
        if (len(lstfaces[i])<=0):
            detect=[]
        else:
            for ele in lstfaces[i]:
                jsonobj = {}
                jsonobj["iname"] = imgname
                x,y,w,h=ele.astype('float32')
                jsonobj["bbox"] = [x,y,w,h]
                json_list.append(jsonobj)
    #the result json file name
    output_json = opfilename
    #dump json_list to result.json
    df =pd.DataFrame(json_list)
    jarray= df.to_json(orient ='records')
    parsed = json.loads(jarray)
    with open(output_json, 'w') as f:
        json.dump(parsed, f)
    return parsed


def get_faces(imgpath,chdirpath):
    imgs = read_images(imgpath)
    detected_imgs=[]
    boxlst=[]
    for img in imgs:
        imglst,faces = detect_faces(img)
        boxlst.append(faces)
        for ele in imglst: 
            detected_imgs.append(ele)
        n=len(detected_imgs)
    detected_imgs = [y for y in detected_imgs if 0 not in y.shape]
    old_dir=os.getcwd()
    if(type(chdirpath)==list):
        chdirpath=chdirpath[0]
    os.chdir(chdirpath)
    filenames = os.listdir()
    os.chdir(old_dir)
    opfilename="results_test.json"
    json_lst= json_create(boxlst,filenames,opfilename)
    return detected_imgs,boxlst
               
    
                
    
if __name__ == "__main__":
    arg_path= sys.argv
    if(len(arg_path)<=1):
        rel_path='./Validation folder/images'
        val_imgpath=rel_path+'/*.jpg'
    else:
        rel_path=arg_path[1]
        val_imgpath=rel_path+'/*.jpg'
    val_imgs = read_images(val_imgpath)
    detected_imgs=[]
    boxlst=[]

    for img in val_imgs:
        imglst,faces = detect_faces(img)
        boxlst.append(faces)
        for ele in imglst: 
            detected_imgs.append(ele)
    old_dir=os.getcwd()
    os.chdir(rel_path)
    filenames = os.listdir()
    os.chdir(old_dir)
    opfilename="results.json"
    json_lst= json_create(boxlst,filenames,opfilename)