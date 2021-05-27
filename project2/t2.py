# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt



def feature_extract(image):
    sift = cv2.SIFT_create()
    # sift = cv2.SIFT_create()
    #kp,des = sift.detectAndCompute(value,None)
    #orb_descriptor = cv2.ORB_create()
    #(kps, features) = orb_descriptor.detectAndCompute(image, None)
    (kps, features) = sift.detectAndCompute(image, None)
    return kps, features

def match_keypoints(kps1,kps2,features1,features2):
    matches = []
    #good_matches=[]
    for p2 in range(len(features2)):
      lst = []
      for p1 in range(len(features1)):
        lst.append(np.dot(features2[p2]-features1[p1],features2[p2]-features1[p1]))
      if(min(lst)<5000):
        matches.append([p2,lst.index(min(lst))])
        
     
    return matches

def getHomography(kps1, kps2, features1, features2, matches, reprojThresh):
    if len(matches)>4:
 
        src_pts = np.array([kps1[match[1]].pt for match in matches])
        dst_pts = np.array([kps2[match[0]].pt for match in matches])       
        #src_pts = np.float32([ kps1[m].pt for m in matches ]).reshape(-1,1,2)
        #dst_pts = np.float32([ kps2[m].pt for m in matches ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        #matchesMask = mask.ravel().tolist()
        return M,mask
    else:
        print( "Not enough matches are found - {}/{}".format(len(matches), 4) )
        #matchesMask = None
        return None


def stitch_seed(update_Image, dst_Img):
    kpsi,featuresi= feature_extract(update_Image)
    kpsi1,featuresi1= feature_extract(dst_Img)
    coordinates_to_stitch= match_keypoints(kpsi,kpsi1,featuresi,featuresi1)
    (H,status) = getHomography(kpsi,kpsi1,featuresi,featuresi1,coordinates_to_stitch,100)
    w2,h2 = dst_Img.shape[:2]
    w1,h1 = update_Image.shape[:2]
    if len(coordinates_to_stitch)!=0:
        width = w1 + w2
        height = h1 + h2
        points0 = np.array([ [0,0], [0,w1], [h1, w1], [h1,0] ], dtype=np.float32)
        points0 = points0.reshape((-1, 1, 2))
        points1 = np.array([ [0,0], [0,w2], [h2, w2], [h2,0] ], dtype=np.float32)
        points1 = points1.reshape((-1, 1, 2))
        points2 = cv2.perspectiveTransform(points1, H)
        
        points = np.concatenate((points0, points2), axis=0)
        [x_min, y_min] = np.int32(points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(points.max(axis=0).ravel() + 0.5)
        H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        H_final = H_translation.dot(H)
        result = cv2.warpPerspective(update_Image, H_final, (width, height))
        
        #plt.imshow(result)
        result[-y_min:w2-y_min, -x_min:h2-x_min] = dst_Img
        update_Image=result
        
        #plt.figure(figsize=(20,10))
        #plt.imshow(update_Image)

        #plt.axis('off')
        #plt.show()
        #cv2.imwrite('test.png',update_Image)
        
        return update_Image
    

def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    imgs_copy=imgs
    update_Image=imgs[0]
    update_image=imgs[1]
    update_image=stitch_seed(update_image, imgs[0])
    imgs=imgs[2:]
    
    for j in range(len(imgs)):
        update_image=stitch_seed(update_image, dst_Img=imgs[j])
    
    plt.figure(figsize=(20,10))
    plt.imshow(update_image)

    plt.axis('off')
    plt.show()
    cv2.imwrite(savepath,update_image)
    
    n=len(imgs_copy)
    overlap_arr=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            kpsi,featuresi= feature_extract(imgs_copy[i])
            kpsj,featuresj= feature_extract(imgs_copy[j])
            matches = match_keypoints(kpsi,kpsj,featuresi,featuresj)
            if len(matches)>50:
                overlap_arr[i][j]=1
            
    return overlap_arr

if __name__ == "__main__":
    #task2
    #overlap_arr = stitch('t2', N=4, savepath='task2.png')
    #with open('t2_overlap.txt', 'w') as outfile:
    #    json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t4', savepath='task4.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
