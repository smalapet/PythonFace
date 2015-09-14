############################################################################################
#Copyrights 2015 Sivarat Malapet all rights reserved.
#Facial Recognition Learning Module
############################################################################################

from lib.rgbhistogram import RGBHistogram
from lib.facedetector import FaceDetector
from lib import imutils
from sklearn.preprocessing import LabelEncoder
import numpy as np
import argparse
import glob
import cv2
import pickle
import time
import urllib2

from PIL import Image, ImageFilter

ap = argparse.ArgumentParser()
ap.add_argument("-c","--cascade",help="path to where the face cascade resides",default="cascades/haarcascade_frontalface_default.xml")
ap.add_argument("-e","--eye",help="path to where the eye cascade resides",default="cascades/haarcascade_eye.xml")
ap.add_argument("-f","--faces",help="path to the image dataset",default="faces_data/")
ap.add_argument("-tf","--testface",help="path to the test image dataset",default="faces_realtime/cloony2.jpg")
args = vars(ap.parse_args())

###############################################
#Load faces classifier model from disk
pkl_file = open('models/faces_classifier.pkl', 'rb')
classifier = pickle.load(pkl_file)
pkl_file.close()
print "Load model from disk successfully" 
###############################################

###############################################
fd = FaceDetector(args["cascade"])
desc = RGBHistogram([8,8,8])
###############################################

###############################################
imagePaths = sorted(glob.glob(args["faces"]+"/*.jpg"))
target = []
for (imagePath) in imagePaths:
 image = cv2.imread(imagePath)
 features = desc.describe(image,None)
 target.append(imagePath.split("_")[-2])

targetNames = np.unique(target)
le = LabelEncoder()
target = le.fit_transform(target)
###############################################

face_file_name = args["testface"]
print "Read test file "+face_file_name

image = cv2.imread(face_file_name)

features = desc.describe(image,None)
face = le.inverse_transform(classifier.predict(features))[0]

print "Predict => "+face
txt_file = open("predict.txt","w")
txt_file.write(face)
txt_file.close()

cv2.imshow("Face", features)











