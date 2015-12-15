import pickle
import numpy as np
import csv

submission = pickle.load(open('pickle/submission.pkl'))
ListOfName = pickle.load(open('pickle/ListOfName.pkl'))
ImageName = pickle.load(open('pickle/TestImageName.pkl'))
title = ['image']

for i in ListOfName:
    title.append(i)

with open('submission.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter= ',')
    writer.writerow(title)
    count = -1
    for name in ImageName:
        count = count + 1
        print(count)
        templist = [name]
        
        for i in submission[count]:
            templist.append(i)
        writer.writerow(templist)
        
