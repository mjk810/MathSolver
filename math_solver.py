#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 11:03:23 2021

@author: marla
"""


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from PIL import Image

import cv2
import itertools


filepath = ''
filename = 'math.jpg'

   
def generateIndividualNumbers(s_contours, boxes):
    #create an image for each box; these images will be fed to classifier to predict the number or symbol
    mask = np.ones(im.shape[:2], dtype="uint8") * 255
    
    # Draw the contours on the mask
    masked = cv2.drawContours(mask, s_contours, -1, 0, cv2.FILLED)
    #crop the image to the bounding box and write to a file
    for i in range(len(boxes)):
        #add padding for additional white space 
        padding= 5
        x=boxes[i][0] - padding
        y=boxes[i][1] - padding
        w=boxes[i][2] + 2*padding
        h=boxes[i][3] + 2*padding
        print(x, y, w, h)
        cropped = masked[y:y+h,x:x+w].copy()
        cv2.imwrite('MathSolver/contour'+str(i)+'.jpg', cropped) 
        #plt.imshow(masked, cmap='gray')
        
def makeCombos(iterable, r=2):
    #get all possible 2 box combinations
    combos = itertools.combinations(iterable, r)
    
    return combos

def readImage(image_file):
    #read in the image
    im = cv2.imread(image_file)
    plt.imshow(im)
    
    return im
   
   
def binarizeImage(im):
     #convert image to grayscale
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    plt.imshow(imgray)
    
    #use a threshold to convert the image to a binary image; threshold determined by trial and error
    #with lined paper, used a lower threshold to eliminate lines
    threshhold_used,binary_image = cv2.threshold(imgray,80,255,cv2.THRESH_BINARY_INV)
    plt.imshow(binary_image, cmap = 'gray')
    
    return binary_image

def getImageContours(binary_image):
     # find the contours in the binary image
    im2, contours, hierarchy = cv2.findContours(binary_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    #create an image that is a superposition of the contours on the original image
    #arguments ar ethe image, the contours as a list, the index of the contour (-1 means all), the color, the thickness
    contour_image = cv2.drawContours(im, contours, -1, (0,255,0), 3)
    plt.imshow(contour_image)
 
    print("Number of contours found: ", len(contours))
    return contours
 
def getSortedBoundingBoxes(contours, im):
    #get the coordinates of the bounding boxes and sort them from left to right
    #box coordinates are x, y, w, h
    boxes = []
    for c in contours:
        bounding_rectangle = cv2.boundingRect(c)
        boxes.append(bounding_rectangle)
      
    #zip(*zipped item) unzips it; in this casee, unzipping the sorted zipped list;
    #sorted on x[0] which is the x coord of the bounding box
    s_boxes, s_contours = zip(*sorted(zip(boxes, contours), key=lambda x:x[0]))
   
    return list(s_boxes), s_contours

def isIntersection(box1, box2):
    #determine if there is an overlap in the x direction
    intersection = False
    if (box2[0]>=box1[0] and box2[0] < box1[0] + box1[2]) or (box2[0]<box1[0] and box2[0] + box2[2] >box1[0]):
        intersection = True
    
    return intersection
        
def redrawIntersectingBoxes(box1, box2):
    #box1 and box2 are intersecting boxes; find the coordinates of the new bounding box that includes both
    x=min(box1[0], box2[0])   
    x2coord = max(box1[0] + box1[2], box2[0] + box2[2])
    w=x2coord - x
    y=min(box1[1], box2[1])
    y2coord = max(box1[1] + box1[3], box2[1] + box2[3])
    h=y2coord-y
    
    return x,y,w,h



#driver code

im = readImage(filepath + filename)
binary_image = binarizeImage(im)   
image_contours = getImageContours(binary_image)  
#sort the contours from left to right and get the bounding_boxes and points along the contour
boxes, s_contours = getSortedBoundingBoxes(image_contours, im) 

#create all possible combinations of bounding boxes to look for intersecting boxes
combos = list(makeCombos(boxes, 2))

#iterate over the pairs and determine if they intersect; if yes, create new bounding_box
for pairs in combos:
    intersect = isIntersection(pairs[0], pairs[1])
    if intersect:
        x,y,w,h = redrawIntersectingBoxes(pairs[0], pairs[1])

        #update the list of bounding boxes; remove individual boxes and add the combined box
        boxes.remove(pairs[0])
        boxes.remove(pairs[1])
        boxes.append((x,y,w,h))

#crop each image based on the bounding_boxes and write the image to a file for classification
generateIndividualNumbers(s_contours, boxes)
    
