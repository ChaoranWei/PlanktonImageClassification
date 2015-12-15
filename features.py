import os
import sys
import numpy as np
import shutil,sys
import skimage.feature
from sklearn import preprocessing as prep
from skimage import measure
from skimage import morphology
from skimage.feature import peak_local_max
from skimage import segmentation
from skimage.morphology import watershed
import glob
from skimage.io import imread
from skimage.transform import resize
import pickle


def unit_to_float(img):
    return 1 - (img / np.float32(255))
    
    
maxPixel =40
totalPixel = maxPixel*maxPixel

directory_names = glob.glob("data/train/*")
class_names = [os.path.basename(d) for d in directory_names]
class_names.sort()
num_classes = len(class_names)

paths_train = glob.glob("data/train/*/*")
paths_train.sort()

paths = { 'train': paths_train}

    
labels_train = np.zeros(len(paths['train']), dtype='int32')
for k, path in enumerate(paths['train']):
    class_name = os.path.basename(os.path.dirname(path))
    labels_train[k] = class_names.index(class_name)

def load(subset='train'):
    images = np.empty(len(paths[subset]), dtype='object')
    for k, path in enumerate(paths[subset]):
        image = imread(path, as_grey=True)
        #image_shape = image
        #image = resize(image, (maxPixel, maxPixel)) 
        images[k] = image
    return images  

images_train=load('train')

    

def getLargestRegion(props, labelmap, imagethres):
    regionmaxprop = None
    for regionprop in props:
        # check to see if the region is at least 50% nonzero
        if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop

#Feature from kaggle tutorial website
def getMinorMajorRatio_2(Image):
    Image = Image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(Image > np.mean(Image),0.,1.0)
 
    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((4,4)))
 
    # Create the label list
    label_list = measure.label(imdilated)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)
    
    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)
        
    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    minor_axis_length = 0.0
    major_axis_length = 0.0
    area = 0.0
    convex_area = 0.0
    eccentricity = 0.0
    equivalent_diameter = 0.0
    euler_number = 0.0
    extent = 0.0
    filled_area = 0.0
    orientation = 0.0
    perimeter = 0.0
    solidity = 0.0
    centroid = [0.0,0.0]
    if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
        ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
        minor_axis_length = 0.0 if maxregion is None else maxregion.minor_axis_length 
        major_axis_length = 0.0 if maxregion is None else maxregion.major_axis_length  
        area = 0.0 if maxregion is None else maxregion.area  
        convex_area = 0.0 if maxregion is None else maxregion.convex_area  
        eccentricity = 0.0 if maxregion is None else maxregion.eccentricity  
        equivalent_diameter = 0.0 if maxregion is None else maxregion.equivalent_diameter  
        euler_number = 0.0 if maxregion is None else maxregion.euler_number  
        extent = 0.0 if maxregion is None else maxregion.extent 
        filled_area = 0.0 if maxregion is None else maxregion.filled_area  
        orientation = 0.0 if maxregion is None else maxregion.orientation 
        perimeter = 0.0 if maxregion is None else maxregion.perimeter  
        solidity = 0.0 if maxregion is None else maxregion.solidity
        centroid = [0.0,0.0] if maxregion is None else maxregion.centroid
 
    return ratio,minor_axis_length,major_axis_length,area,convex_area,eccentricity,\
           equivalent_diameter,euler_number,extent,filled_area,orientation,perimeter,solidity, centroid[0], centroid[1]

#minor/major ratio,size of image

features = {"tutorial": [], "image_size":[]}

#image_shapes = np.asarray([img.shape for img in images_train]).astype(np.float32)
#print image_shapes.shape


for i,img in enumerate(images_train):
    img_o = img.copy()
    img = unit_to_float(img)
    tut_features = np.array(getMinorMajorRatio_2(img_o))
    features["tutorial"].append(tut_features)
    features["image_size"].append(np.array(img_o.shape))
    
print features 

TARGET_PATH = 'pickle/features.pkl'
with open(TARGET_PATH, 'w') as f:
    pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)
