
# coding: utf-8
import cv2
import numpy as np
import csv as csv
import pandas as pd
import math

# Load 'test.csv' file
test_file = open('./training.csv')
test_file_object = csv.reader(test_file)
header = next(test_file_object)

test_data = []
for row in test_file_object:
    test_data.append(row[30].split(" "))

#Just show one Picture for test
show_window = np.array(test_data[0])
show_window.shape = 96,96
np.dtype(np.uint8)
cv2.namedWindow("Picture")
cv2.imshow('Picture',show_window.astype(np.float)/255)
cv2.imwrite('Picture.jpg', show_window.astype(np.uint8))  #-->not work
cv2.waitKey(0)

#HAAR classfier
classfier=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
image = show_window.astype(np.uint8)
cv2.equalizeHist(image)
divisor=8
h, w = (96,96)
minSize=(w/divisor, h/divisor)
print(type(minSize))
faceRects = classfier.detectMultiScale(image, 1.2, 2, cv2.CASCADE_SCALE_IMAGE,math.floor(minSize))
#draw the rectangle of the face
for faceRect in faceRects:
	x, y, w, h = faceRect
	cv2.rectangle(image, (x, y), (x+w, y+h), color)
#show Picture
cv2.imshow("Recognition", image)
cv2.waitKey(0)
