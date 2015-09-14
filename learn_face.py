############################################################################################
#Copyrights 2015 Sivarat Malapet all rights reserved.
#Facial Recognition Learning Module
############################################################################################

from lib.rgbhistogram import RGBHistogram
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import argparse
import glob
import cv2
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-f","--faces",default="faces_data/",help="path to the image dataset")
args = vars(ap.parse_args())

#Training phase#
imagePaths = sorted(glob.glob(args["faces"]+"/*.jpg"))

data = []
target = []

desc = RGBHistogram([8,8,8])

for (imagePath) in imagePaths:
 image = cv2.imread(imagePath)
 features = desc.describe(image,None)
 data.append(features)
 target.append(imagePath.split("_")[-2])

targetNames = np.unique(target)
le = LabelEncoder()
target = le.fit_transform(target)

(trainData,testData,trainTarget,testTarget) = train_test_split(data,target,test_size=0.3,random_state=42)

model = RandomForestClassifier(n_estimators=25,random_state=84)
model.fit(trainData,trainTarget)

print classification_report(testTarget,model.predict(testData),target_names=targetNames)

#Save faces classifier model to disk#
output = open("models/faces_classifier.pkl", 'wb')
pickle.dump(model, output)
output.close()
print "Save model to disk successfully"

#Testing phase#

#Load faces classifier model from disk
pkl_file = open('models/faces_classifier.pkl', 'rb')
classifier = pickle.load(pkl_file)
pkl_file.close()
print "Load model from disk successfully" 

for i in np.random.choice(np.arange(0,len(imagePaths)),20):
 imagePath = imagePaths[i]
 image = cv2.imread(imagePath)
 features = desc.describe(image,None)

 face = le.inverse_transform(classifier.predict(features))[0]
 print imagePath
 print "Nice to meet you %s" % (face.upper())
 cv2.imshow("image",image)
 cv2.waitKey(0)


 




 




 




