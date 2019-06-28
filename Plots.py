import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2

import numpy as np
import pandas as pd

df = pd.read_csv('TrainTestSet.csv')
hour = 0
postwtcount_list = []
negtwtcount_list = []
while hour<24:
    poscount = 0
    negcount = 0
    for index,rows in df.iterrows():
        if rows['tweet_hour'] == hour:
            if rows['Stress'] == 1 :
                pcount +=1
            else :
                ncount+=1
    if poscount > 30 :
        poscount = 30                       # Scaling the results for better visualization
    if negcount > 30 :
        negcount = 30
    postwtcount_list.append(pcount)
    negtwtcount_list.append(ncount)
    hour+=1

ind = np.arange(24)
plt.bar(ind,postwtcount_list,0.4,label = "Stessed",color = 'r')
plt.bar(ind + 0.4,negtwtcount_list,0.4,label = "Non_Stessed",color = 'g')
plt.legend()
plt.xlabel('Tweet_Hour')
plt.ylabel('Tweets')
plt.xticks(ind+0.2,ind)
plt.show()


i=2                   # Positive and neg word count = [2,4,6,8,10] hence i+=2
pos_stress_list = []
neg_stress_list = []
pos_unstress_list = []
neg_unstress_list = []

while i <= 10 :
    pscount = 0
    nscount = 0
    pcount = 0
    ncount = 0
    for index,rows in df.iterrows():
        if rows['Positive_Words'] == i and rows['Stress'] == 1 :
            pscount+=1
        if rows['Positive_Words'] == i and rows['Stress'] == 0 :
            pcount+=1
        if rows['Negative_Words'] == i and rows['Stress'] == 1 :
            nscount+=1
        if rows['Negative_Words'] == i and rows['Stress'] == 0 :
            ncount+=1
        
    i+=2
    pos_stress_list.append(pscount)
    pos_unstress_list.append(pcount)
    neg_stress_list.append(nscount)
    neg_unstress_list.append(ncount)
    
    
ind = np.arange(5)
plt1.bar(ind,pos_stress_list,0.4,label = "Stessed",color = 'r')
plt1.bar(ind + 0.4,pos_unstress_list,0.4,label = "Non_Stessed",color = 'g')
plt1.legend()
plt1.xlabel('Number of Positive Words')
plt1.ylabel('Tweets')
plt1.xticks(ind+0.2,[2,4,6,8,10])
plt1.show()

    

plt2.bar(ind,neg_stress_list,0.4,label = "Stessed",color = 'r')
plt2.bar(ind + 0.4,neg_unstress_list,0.4,label = "Non_Stessed",color = 'g')
plt2.legend()
plt2.xlabel('Number of Negative Words')
plt2.ylabel('Tweets')
plt2.xticks(ind+0.2,[2,4,6,8,10])

