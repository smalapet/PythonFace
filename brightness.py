############################################################################################
#Copyrights 2015 Sivarat Malapet all rights reserved.
############################################################################################

from argparse import ArgumentParser
from scipy.misc import *
from scipy.ndimage import morphology
import colorsys
import time
import numpy as np
import time
import os
import Image


def brightness_contrast(image, brightness = -100, contrast = 300):
    def vect(a):
        c   = contrast
        b   = 100 * brightness
        res = ((a - 127.5) * c + 127.5) + b
        if res < 0 :
            return 0
        if res > 255:
            return 255
        return res

    transform = np.vectorize(vect)
    data = transform(fromimage(image)).astype(np.uint8)
    return toimage(data)

im = Image.open("lena.jpg")
#Default contrast = 1, brightness = 0
#Formular new_value = (old_value - 0.5) x contrast + 0.5 + brightness
#brightness -2 to +2
#brightness_contrast(img, brightness=-1.5, contrast=1).show()
lcnt = -2;
for epoch in range(1,7):
 #Default contrast = 1, brightness = 0
 #Formular new_value = (old_value - 0.5) x contrast + 0.5 + brightness
 #brightness -2 to +2
 lcnt = lcnt+0.5
 img=brightness_contrast(im, brightness=lcnt, contrast=1)
 img.save("lena.jpg"+"-"+str(epoch)+".jpg")  

