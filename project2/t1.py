#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import imutils

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
    for p2 in range(len(features2)):
      lst = []
      for p1 in range(len(features1)):
        lst.append(np.dot(features2[p2]-features1[p1],features2[p2]-features1[p1]))
      if(min(lst)<5000):
        matches.append([p2,lst.index(min(lst))])
    return matches

def getHomography(kps1, kps2, features1, features2, matches, reprojThresh):
    if len(matches)>10:
        #print (kps1[0].pt)
        src_pts = np.array([kps2[match[0]].pt for match in matches])
        dst_pts = np.array([kps1[match[1]].pt for match in matches])       
        #src_pts = np.float32([ kps1[m].pt for m in matches ]).reshape(-1,1,2)
        #dst_pts = np.float32([ kps2[m].pt for m in matches ]).reshape(-1,1,2)
        #print(src_pts)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        #matchesMask = mask.ravel().tolist()
        return M,mask
    else:
        print( "Not enough matches are found - {}/{}".format(len(matches), 4) )
        #matchesMask = None
        return None



def Laplacian_blending(img1,img2,mask,levels=4):
    G1 = img1.copy()
    G2 = img2.copy()
    GM = mask.copy()
    gp1 = [G1]
    gp2 = [G2]
    gpM = [GM]
    for i in range(levels):
       	G1 = cv2.pyrDown(G1)
       	G2 = cv2.pyrDown(G2)
       	GM = cv2.pyrDown(GM)
       	gp1.append(np.float32(G1))
       	gp2.append(np.float32(G2))
       	gpM.append(np.float32(GM))

   	# generate Laplacian Pyramids for A,B and masks
    lp1  = [gp1[levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lp2  = [gp2[levels-1]]
    gpMr = [gpM[levels-1]]
    for i in range(levels-1,0,-1):
       # Laplacian: subtarct upscaled version of lower level from current level
       # to get the high frequencies
       
       L1 = cv2.subtract(gp1[i-1], cv2.pyrUp(gp1[i]))
       L2= cv2.subtract(gp2[i-1], cv2.pyrUp(gp2[i]))
       lp1.append(L1)
       lp2.append(L2)
       gpMr.append(gpM[i-1]) # also reverse the masks
       # Now blend images according to mask in each level
       LS = []
    for l1,l2,gm in zip(lp1,lp2,gpMr):
       ls = l1 * gm + l2 * (1.0 - gm)
       LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1,levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    return ls_    


def foreground_removal (img1,img2,H):
    img1old=img1
    img2old=img2
    grayimg1= cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)    
    grayimg2= cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)    
    thresh1 = cv2.threshold(grayimg1,0,255,cv2.THRESH_BINARY_INV| cv2.THRESH_OTSU)[1]
    thresh2 = cv2.threshold(grayimg2,0,255,cv2.THRESH_BINARY_INV| cv2.THRESH_OTSU)[1] 
    #print(thresh)
    #contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours1 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    contours2 = cv2.findContours(thresh2, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    #Use the circumscribed rectangle method to find the geometric midpoint
    
    x, y, w, h = cv2.boundingRect(contours1[0])
    cX = x + w//2
    cY = y + h//2
    center = (cX, cY)
    mask = np.zeros(thresh1.shape, dtype="uint8")
    cv2.rectangle(mask,(x,y),(x+w,y+h),255,-1)
    img1 = cv2.seamlessClone(img1, img2old, mask, center, cv2.MIXED_CLONE)
    
    
    x, y, w, h = cv2.boundingRect(contours2[0])
    cX = x + w//2
    cY = y + h//2
    center = (cX, cY)
    mask = np.zeros(thresh2.shape, dtype="uint8")
    cv2.rectangle(mask,(x,y),(x+w,y+h),255,-1)
    img2 = cv2.seamlessClone(img2, img1old, mask, center, cv2.MIXED_CLONE)
    
    return img1,img2

def warp_image(kps1,kps2,features1,features2,coordinates_to_stitch,img1, img2):

    (H,mask) = getHomography(kps1,kps2,features1,features2,coordinates_to_stitch,200)
    
    width = img1.shape[1] + img2.shape[1]
    height = img1.shape[0] + img2.shape[0]
    
    # points0 = np.array([[0, 0], [0, img1.shape[0]], [img1.shape[1], img1.shape[0]], [img2.shape[1], 0]], dtype=np.float32)
    # points0 = points0.reshape((-1, 1, 2))
    # points1 = np.array([[0, 0], [0, img2.shape[0]], [img2.shape[1], img1.shape[0]], [img2.shape[1], 0]], dtype=np.float32)
    # points1 = points1.reshape((-1, 1, 2))
    # points2 = cv2.perspectiveTransform(points1, H)
    
    # points = np.concatenate((points0, points2), axis=0)
    # [x_min, y_min] = np.int32(points.min(axis=0).ravel() - 0.5)
    # [x_max, y_max] = np.int32(points.max(axis=0).ravel() + 0.5)
    # H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    # H_final = H_translation.dot(H)
    
    result = cv2.warpPerspective(img2,H, (width,height), dst=img1.copy(),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_TRANSPARENT)  
       
    return result

    
def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."

    (kps1, features1) = feature_extract(img1)
    (kps2, features2) = feature_extract(img2)
    
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8), constrained_layout=False)
    ax1.imshow(cv2.drawKeypoints(img1,kps1,None,color=(0,255,0)))
    ax1.set_xlabel("(a)", fontsize=14)
    ax2.imshow(cv2.drawKeypoints(img2,kps2,None,color=(0,255,0)))
    ax2.set_xlabel("(b)", fontsize=14)
    plt.show()
    
    coordinates_to_stitch= match_keypoints(kps1,kps2,features1,features2)
    
    width = img1.shape[1] + img2.shape[1]
    height = img1.shape[0] + img2.shape[0]

    result = warp_image(kps1,kps2,features1,features2,coordinates_to_stitch,img1, img2)  
    plt.figure(figsize=(20,10))
    plt.imshow(result)
    
    for i in range(0,img1.shape[0]-1):
        for j in range(0,img1.shape[1]-1):
            if np.sum(result[i][j]) > np.sum(img1[i][j]):
                  result[i][j]=result[i][j]
            else:
                  result[i][j]=img1[i][j]
                          

    plt.figure(figsize=(20,10))
    plt.imshow(result)

    plt.axis('off')
    plt.show()
    cv2.imwrite(savepath,result)
    return result



if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)


