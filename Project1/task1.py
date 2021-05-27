"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling tso detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments
    lst_charFeatures={}
    
    img_test = np.copy(test_img)
    for ele in characters:
      kp,key= enrollment(test_img, ele)
      lst_charFeatures[key]=kp

    bbox=detection(test_img)
    result=recognition(img_test,bbox,lst_charFeatures)
    
    return result


def enrollment(test_img, character):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    key,value=character
    sift = cv2.xfeatures2d.SIFT_create()
    # sift = cv2.SIFT_create()
    kp,des = sift.detectAndCompute(value,None)
    (thresh,test_img) = cv2.threshold(value, 127, 255, cv2.THRESH_BINARY)
    # kp = sift.detect(value,None)
    img=cv2.drawKeypoints(value,kp,test_img)
    #img = cv2.Canny(value,200,100)
    filename='feature_'+key+'.jpg'
    cv2.imwrite(filename, img) 
    return des,key

def detection(test_img1):
    """
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    m,n = test_img1.shape
    (thresh,test_img) = cv2.threshold(test_img1, 127, 255, cv2.THRESH_BINARY)
    pixel_visited=np.array(np.zeros([m,n]))
    conn_comp = []
    for i in range(m):
        for j in range(n):
            #Determining if the image pixel is visited or is a neighbour
            if (test_img[i][j]<0.5  and pixel_visited[i][j]!=1):
                result = []
                visited = set()
                nodes = set([(i,j)])
                while(nodes):
                    node = nodes.pop()
                    x,y = node
                    pixel_visited[x][y]=1
                    visited.add(node)
                    nodes |= neighbours(node,test_img) - visited
                    result.append(node)
                conn_comp.append(result)
    #creating bounding box around connected components identified                
    (bbox_img,bboxes) = proc_comps(conn_comp, test_img1)
    
    filename='testimg_bbox'+'.jpg'
    cv2.imwrite(filename, bbox_img) 
    
    
    return(bboxes)
               
def proc_comps(conn_comp, image):
    img = image
    result = []
    features=[]
    for data in conn_comp:
        min_y = min(data, key = lambda t: t[1])
        max_y = max(data, key = lambda t: t[1])
        min_x = min(data, key = lambda t: t[0])
        max_x = max(data, key = lambda t: t[0])
        start_p = (min_y[1], min_x[0] )
        end_p = ( max_y[1],max_x[0])
        
        # h = max_x[0] - min_x[0]
        # w = max_y[1] - min_y[1]
        # box = [min_x[0], min_y[0], w, h]
        l = []
        l.append(start_p)
        l.append(end_p)
        # print(l)
        result.append(l)
        img = cv2.rectangle(img, start_p, end_p, (0,0,0), 2) 
        
    show_image(img, delay=1000)
    return(img,result)
        
def neighbours(node,test_img):
    i,j = node
    m,n = test_img.shape
    neighbours = [(i+1,j),(i-1,j),(i,j-1),(i,j+1)]
    ans = []
    for ele in neighbours:
        x,y=ele
        if ((x in range(m)) and (y in range(n)) and test_img[x][y]<0.5):
            ans.append(ele)

    return(set(ans))


def recognition(bbox_image,bbox,lst_charFeatures):
    
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # extracts=[]
    result_json = []
    for ele in bbox:
        #extracting individual images from bboxes and test_img
        start_p,end_p=ele
        x_min,y_min=start_p
        x_max,y_max=end_p
        extract=bbox_image[y_min:y_max,x_min:x_max]
        #extracting features for comparison
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(extract,None)
        #identifying most similar char class using Brute Force
        ans = most_similar(des, lst_charFeatures)
        # show_image(extract,delay=100000)
        my_dict = {}
        my_dict["bbox"] = [x_min,y_min, x_max-x_min, y_max-y_min]
        my_dict["name"] = ans
        result_json.append(my_dict)
        # extract=cv2.drawKeypoints(extract,kp,bbox_image)
        # extracts.append(extract)
    return (result_json)

def most_similar(des, lst_charFeatures):
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True)
    lst_charFeatures.pop('dot', None)
    #print(lst_charFeatures.keys())
    #Sift is not recognizing keypoints for dot. 
    #Therefore, initially the plan was to consider all such chars as dot
    #However, dut to lower frequency of dots the number of characters misidentified increases.
    #Therefore dot is not being classified
    if(des is None):
        ans = 'UNKNOWN'
    else:
        matches = {}
        mini_dist = 1850
        ans = "UNKNOWN"
        for ele in lst_charFeatures.keys():
            match = bf.match(des, lst_charFeatures[ele])
            matches[ele] = match
            for i in match:
                if(i.distance < mini_dist):
                    #print(i.distance)
                    mini_dist = i.distance
                    ans = ele
    return(ans)


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = []
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=True)])

    test_img = read_image(args.test_img)
    
    results = ocr(test_img, characters)
    with open(os.path.join(args.rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)
    #save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
