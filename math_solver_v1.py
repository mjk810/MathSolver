#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 19:44:53 2021

@author: marla
"""


def cvMethod(image_file):
    #read in the image
    im = cv2.imread(image_file)
    plt.imshow(im)
   
    #convert image to grayscale
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    plt.imshow(imgray)
    
    #use a threshold to convert the image to a binary image; threshold determined by trial and error
    threshhold_used,binary_image = cv2.threshold(imgray,80,255,cv2.THRESH_BINARY_INV)
    plt.imshow(binary_image, cmap = 'gray')
    
   # find the contours in the binary image
    im2, contours, hierarchy = cv2.findContours(binary_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #create an image that is a superposition of the contours on the original image
    #arguments ar ethe image, the contours as a list, the index of the contour (-1 means all), the color, the thickness
    contour_image = cv2.drawContours(im, contours, -1, (0,255,0), 3)
    plt.imshow(contour_image)
 
    print("Number of contours found: ", len(contours))
    print(hierarchy)
    
    #sort the contours from left to right
    boxes, s_contours = getSortedBoundingBoxes(contours, im)
    #create an image for each box; these images will be fed to classifier to predict the number or symbol
    for i in range(len(s_contours)):
        mask = np.ones(im.shape[:2], dtype="uint8") * 255
    
        # Draw the contours on the mask
        masked = cv2.drawContours(mask, s_contours, i, 0, cv2.FILLED)
        padding= 5
        x=boxes[i][0] - padding
        y=boxes[i][1] - padding
        w=boxes[i][2] + 2*padding
        h=boxes[i][3] + 2*padding
        print(x, y, w, h)
        cropped = masked[y:y+h,x:x+w].copy()
        
        cv2.imwrite('/MathSolver/contour'+str(i)+'.jpg', cropped) 
        #plt.imshow(masked, cmap='gray')
    
    # remove the contours from the image and show the resulting images
   # img = cv2.bitwise_and(im, im, mask=mask)
    #plt.imshow(mask)
    #plt.imshow(img)
    #cv2.waitKey(0)