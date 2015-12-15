from PIL import Image
import os
import skimage.filters 
import gabor
import numpy
import scipy
from skimage.filters import gaussian_filter, gabor_filter, sobel_v, hsobel, sobel
from skimage.filters import gaussian_filter, gabor_filter, sobel_v, hsobel, sobel
from sklearn.metrics import classification_report

class image:
    def __init__(self, filepath, Class):
        self._path = filepath
        self._feature = Image.open(filepath).convert('L')
        self._output = Class
        
        
    def get_feature(self, reshape):
        '''reshape is a boolean which determine whether we reshape the image into 
        one dimension or not. '''
        if reshape:
            self._feature = self._feature.resize((40,40),Image.ANTIALIAS)
            self._feature = numpy.asarray(self._feature)
            self._feature = self._normalize(self._feature)
            self._feature = numpy.reshape(self._feature, (1,1600))
            return self._feature
        else:
            self._feature = self._feature.resize((40,40),Image.ANTIALIAS)
            self._feature = numpy.asarray(self._feature)
            self._feature = self._normalize(self._feature)
            return self._feature
    
    def get_class(self):
        return self._output
    
    def _normalize(self, image):
        Image = numpy.zeros((len(image),len(image[1])))
        for i in range(len(image)):
            for j in range(len(image[1])):
                Image[i][j] = numpy.float32(1.0) - numpy.float32(float(image[i][j]) / 255)
    
        return Image
    
    def OriginalSize(self):
        '''Get the original size of the image.
        Due to my poorly written legacy code, please use this method before 
        get_feature method!
        '''
        return len(self._feature[0]), len(self._feature[0][0]) #do I need to normalize it?
    
    def getGaussianFeatures(self, reshape):
        if reshape:
            self._feature = self._feature.resize((40,40),Image.ANTIALIAS)
            self._feature = scipy.ndimage.filters.gaussian_filter(self._feature, 1)
            self._feature = numpy.asarray(self._feature)
            #self._feature = self._normalize(self._feature)
            self._feature = numpy.reshape(self._feature, (1,1600))
            return self._feature
        else:
            self._feature = self._feature.resize((40,40),Image.ANTIALIAS)
            self._feature = scipy.ndimage.filters.gaussian_filter(self._feature, 1)
            self._feature = numpy.asarray(self._feature)
            #self._feature = self._normalize(self._feature)
            return self._feature   
        
    def getGaborFeatures(self, reshape):
        if reshape:
            self._feature = self._feature.resize((40,40),Image.ANTIALIAS)
            self._feature = gabor_filter(self._feature, frequency=0.8)[0]
            self._feature = numpy.asarray(self._feature)
            #self._feature = self._normalize(self._feature)
            self._feature = numpy.reshape(self._feature, (1,1600))
            return self._feature
        else:
            self._feature = self._feature.resize((40,40),Image.ANTIALIAS)
            self._feature = gabor_filter(self._feature, frequency=0.8)[0]
            self._feature = numpy.asarray(self._feature)
            #self._feature = self._normalize(self._feature)
            return self._feature       