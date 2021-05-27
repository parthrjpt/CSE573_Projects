# -*- coding: utf-8 -*-
"""
Created on Mon May  3 18:21:56 2021

@author: parth
"""

import FaceDetector as fd
import face_recognition
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import glob
import os 
import json

def Kmeans_learn(X,centers):
    cluster_labels=[]
    for i in range(len(X)):
        min_dist = np.sqrt((X[i]-centers[0])**2).sum()
        cluster_idx=0
        for j in range(len(centers)):
            dist = np.sqrt((X[i]-centers[j])**2).sum()
            if(dist<min_dist):
                min_dist=dist
                cluster_idx=j
        cluster_labels.append(cluster_idx)
    return cluster_labels
 
def calculate_new_cluster_centers(encoded_imgs,cluster_labels,k):
    new_centers=[]
    for i in range(k):
        clusters=[]
        clusters=encoded_imgs[cluster_labels == i]
        mean= np.array(clusters).mean(axis=0)
        new_centers.append(mean)
    return new_centers
    
def get_filenames(chdirpath):
    old_dir=os.getcwd()
    if(type(chdirpath)==list):
        chdirpath=chdirpath[0]
    os.chdir(chdirpath)
    filenames = os.listdir()
    os.chdir(old_dir)
    return filenames

def split_clusters(images,cluster_labels,k):
  clusters=[]
  i=0
  j=0
  for i in range(k):
      cluster=[]
      cdict={}
      idx =[j for j in range(len(cluster_labels)) if cluster_labels[j] == i]
      for ele in idx:
          cluster.append(images[ele])
      cdict['key'] =i
      cdict['val'] = cluster
      clusters.append(cdict)
  return clusters 

def json_create(clusters,filenames,cluster_labels,opfilename):
    json_list=[]
    for i in range(len(clusters)):
        #imgname = 'img_'+str((i+1))+'.jpg'
        jsonobj = {}
        jsonobj["cluster_no"] = i
        lstclusterfiles=[]
        for j in range(len(cluster_labels)):
            if(cluster_labels[j]==i):
                lstclusterfiles.append(filenames[j])
        jsonobj["elements"] = lstclusterfiles
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

if __name__ == "__main__":
    img_path ='./faceCluster_?/*.jpg'
    chdirpath= glob.glob('./faceCluster_?/') 
    faces,boxes=fd.get_faces(img_path,chdirpath)
    k=int(chdirpath[0].split('_')[1].split('\\')[0])
    
    encoded_imgs=[]
    for i in range(len(faces)):
       ei=face_recognition.face_encodings(faces[i])
       encoded_imgs.append(ei[0])
       
    center_idxs = np.random.choice(len(encoded_imgs),k)
    centers = []
    new_centers=[]
    for ele in center_idxs:
        centers.append(encoded_imgs[ele])
    while True:
        cluster_labels = Kmeans_learn(encoded_imgs,centers)
        new_centers=calculate_new_cluster_centers(np.array(encoded_imgs),np.array(cluster_labels),k)

        if (np.array(centers)==np.array(new_centers)).all():
            break
        centers = new_centers
    
    clusters=split_clusters(faces,cluster_labels,k)

    
    for i in range(k):
        img_i=clusters[i]['val']
        
        h_min = min(img.shape[0] 
                for img in img_i)
      
        # image resizing 
        im_list_resize = [cv2.resize(img,
                       (int(img.shape[1] * h_min / img.shape[0]),h_min)) 
                      for img in img_i]
      
        
        imgs_comb = cv2.hconcat(im_list_resize)

        img_save_path='Cluster_'+str(i)+'.jpg'
        cv2.imwrite(img_save_path, imgs_comb)
    
    lstfilenames=get_filenames(chdirpath)
    opfilename='clusters.json'
    jarray= json_create(clusters,lstfilenames,cluster_labels,opfilename)
    