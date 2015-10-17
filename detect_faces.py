############################################################################################
#Copyright 2015 Sivarat Malapet all rights reserved.
############################################################################################

# import the necessary packages
from lib.facedetector import FaceDetector
from lib import imutils
import argparse
import cv2
from PIL import Image, ImageFilter

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", default="cascades/haarcascade_frontalface_default.xml",
	help = "path to where the face cascade resides")
ap.add_argument("-i", "--image", default="input_face/lena.jpg",
	help = "path to where the image file resides")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# find faces in the image
fd = FaceDetector(args["face"])
faceRects = fd.detect(gray, scaleFactor = 1.1, minNeighbors = 5,
	minSize = (100, 100))
print "I found %d face(s)" % (len(faceRects))

# loop over the faces and draw a rectangle around each
for (x, y, w, h) in faceRects:
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        sub_face = image[y:y+h, x:x+w]
        sub_face = imutils.resize(sub_face,width=200,height=200)
        sub_face_file_name = "input_face/subface.jpg"        
        cv2.imwrite(sub_face_file_name, sub_face)

detected_face_file = "input_face/detected.jpg"
print "Write detected file %s" % detected_face_file
cv2.imwrite(detected_face_file,image)

size = (100,100)
face_file_name = "input_face/desc.jpg"
im = Image.open(sub_face_file_name)
im = im.filter(ImageFilter.EDGE_ENHANCE_MORE) #BLUR CONTOUR DETAIL EDGE_ENHANCE EDGE_ENHANCE_MORE EMBOSS FIND_EDGES SMOOTH SMOOTH_MORE SHARPEN 
im.thumbnail(size)
im.save(face_file_name)



