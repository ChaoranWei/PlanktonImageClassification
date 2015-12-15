import pickle
import numpy as np
import csv

submission0 = pickle.load(open('pickle/submission0.pkl'))
print(len(submission0))
submission1 = pickle.load(open('pickle/submission1.pkl'))
print(len(submission1))
ListOfName = pickle.load(open('pickle/ListOfName.pkl'))
ImageName = pickle.load(open('pickle/TestImageName.pkl'))
title = ['image']

for i in ListOfName:
    title.append(i)

with open('submission.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter= ',')
    writer.writerow(title)
    
    for i in submission0:
        templist = [ImageName.pop(0)]
        templist.extend(i)
        writer.writerow(templist)
    for j in submission1:
        templist = [ImageName.pop(0)]
        templist.extend(j)
        writer.writerow(templist)
    print(len(templist))
        
    
        
    
        
