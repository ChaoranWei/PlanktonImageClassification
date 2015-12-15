from os import listdir
from imageclass import image
import numpy
from random import shuffle
import pickle
import skimage

def train_load():
    path = 'data/train'
    
    Dir = listdir(path)
    imagelist = []
    ListOfName = []
    count = 0
    for directory in Dir:
        if directory != '.DS_Store': #For some reason this directory keep bumping up
            subpath = listdir(path + '/' + directory)
            ListOfName.append(directory)
            print(directory)
            count = count + 1
            for subdir in subpath:
                if subdir != '.DS_Store':
                    Image = image(path + '/' + directory + '/' + subdir, count)
                    Image = Image.get_feature(reshape = True)
                    #Image = Image.getGaborFeatures(reshape = True)
                    output = count - 1
            #        output = numpy.zeros((121,1))
             #       output[count -1] = numpy.float64(1)
                    imagelist.append([Image, output])
                    
    #we need to randomly shuffle the data to retain the generality 
    shuffle(imagelist)
    with open('pickle/ListOfName.pkl','w') as f:
                pickle.dump(ListOfName,f)      
    print('got image list for train')
    return imagelist, count

def test_load():
    path = 'data/test'
    
    Dir = listdir(path)
    imagelist = []
    ListOfName = []
    for directory in Dir:
        if directory != '.DS_Store': #For some reason this directory keep bumping up
            ListOfName.append(directory)
            Image = image(path + '/' + directory, 1000) #assign a random class to test data
            Image = Image.get_feature(reshape = True)
            imagelist.append([Image, 1000])

    with open('pickle/TestImageName.pkl','w') as f:
        pickle.dump(ListOfName,f)  
    print('got image list for test')
                    
    #shuffle(imagelist)

    return imagelist
    
    
def split_data(imagelist, output, option):
    '''option is either 2 or 3 meaning split into train, test or train, validation, test. '''
    if option == 2:
        split = int(len(imagelist) * 0.8)
        Train = imagelist[0:split]
        Test = imagelist[split + 1: len(imagelist) - 1]
        return Train, Test
        
    elif option == 3:
        split1 = int(len(imagelist) * 0.6)
        split2 = int(len(imagelist) * 0.8)
        Train = imagelist[0:split1]
        Val = imagelist[split1 + 1:split2]
        Test = imagelist[split2 + 1: len(imagelist) - 1]
        return Train, Val, Test
    elif option == 4:
        split1 = int(len(imagelist))
        
    else:
        print('wrong option!')
    
def mainTrain():  
    Input, Output = train_load()
    
    for i in range(len(Input)):
        Input[i][0] = Input[i][0][0]
            
    Train, Val = split_data(Input, Output, 2)
    train = zip(*Train)
    train[0] = numpy.asarray(train[0])
    train[1] = numpy.asarray(train[1])
    val = zip(*Val)
    val[0] = numpy.asarray(val[0])
    val[1] = numpy.asarray(val[1])   
    data = train, val
    print('saving train.pkl')
    with open('pickle/train.pkl','w') as f:
        pickle.dump(data,f)  
    print('saved')
        
def mainTest():
    TestData = test_load()

    for i in range(len(TestData)):
        TestData[i][0] = TestData[i][0][0]

    test = zip(*TestData)
    test[0] = numpy.asarray(test[0])
    test[1] = numpy.asarray(test[1])   
    print('saving test.pkl')

    with open('pickle/test.pkl','w') as f:
        pickle.dump(test,f)
    print('saved')


#mainTrain()
train_load()
#mainTest()
    
#format for the data
#(array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
#       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
 #      [ 0.,  0.,  0., ...,  0.,  0.,  0.],
  #     ..., 
 #      [ 0.,  0.,  0., ...,  0.,  0.,  0.],
  #     [ 0.,  0.,  0., ...,  0.,  0.,  0.],
   #    [ 0.,  0.,  0., ...,  0.,  0.,  0.]], dtype=float32), array([5, 0, 4, ..., 8, 4, 8])) numpy.int64