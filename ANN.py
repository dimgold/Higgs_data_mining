''' 
Data Mining Course - Final Project
Dima Goldenberg 
Ori Barkan
dimgold@gmail.com
Runtime ~3 min
'''


fileDir = '//'

import numpy as np #
import pandas as pd #
from sklearn.covariance import empirical_covariance #
import neurolab as nl
import pylab as pl
import time

def read_data(filename,fileDir):
    ''' reading file into pandas data frame'''
    filename = fileDir + filename
    data = pd.read_csv(filename) #reading with pandas
    return data

def splitframe(data,name):
    ''' returning a subset dataframe with rellevant label'''
    df = data[data.Label == name]    
    return df           

def fisher_ratio(a,b):
    '''fishers_ratio'''
    return (np.mean(a)-np.mean(b))**2 / (np.var(a) + np.var(b))
    
def accuracy(predict,classification):
    '''calculates accuracy ratio'''
    GoodPrediction  = (predict == classification)
    return GoodPrediction.value_counts()[True]*1.0 / len(GoodPrediction)
    
    
print 'Data Mining Final Project - Ori and Dima, Autumn 2015 - ANN'
print ' '
print time.ctime()
print ' '

#data load
traindata = read_data('Higgs_subTrain.csv',fileDir)
testdata = read_data('Higgs_subTest.csv',fileDir)  
Signal = splitframe(traindata,'s')
Back = splitframe(traindata,'b')

#labels
ytrain = traindata.iloc[:,-1]
ytest  = testdata.iloc[:,-1]

'''
Normalization
trainNorm = traindata.iloc[:,:-1]
testNorm = testdata.iloc[:,:-1]
X = trainNorm.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
Xtest = testNorm.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
Normalization has no effect'''

X = traindata.iloc[:,:-1]
Xtest =  testdata.iloc[:,:-1]

print 'Feature Selection - Fisher + Correlation'
Fishers = []
for Col in Signal.columns[:-1]:
    F = fisher_ratio(Signal[Col],Back[Col])
    Fishers.append((F,Col))
Fishers.sort(reverse = True)
#Fishers ratio
candidates = Fishers[:5]
candidates.sort(reverse=True)
emp_cov = empirical_covariance(X)
features = list(X.columns[:])
cols = []
for i in range(len(candidates)):
    cols.append(candidates[i][1])
    for j  in range(i+1,len(candidates)):
        ro = emp_cov[features.index(candidates[i][1]),features.index(candidates[j][1])] / np.sqrt(emp_cov[features.index(candidates[i][1]),features.index(candidates[i][1])]*emp_cov[features.index(candidates[j][1]),features.index(candidates[j][1])])
        print  candidates[i][1],", ", candidates[j][1], 'ro is:', ro

# Network Trainings

X = X[cols]
testeventid = Xtest[['EventId']]
Xtest = Xtest[cols]


# replacing labels with 0 and 1
X = np.array(X)
Xtest = np.array(Xtest)
ynum = ((ytrain == 's')*1)
yin = []
for i in ynum:
    yin.append([i])

print ''
print 'Network training'
print time.ctime()

np.random.seed(83)
#network architecture
net = nl.net.newff([[0, 999],[-1.5, 1.5],[-999, 999],[0, 999],[-999, 10000]],[50,10,5,1])
net.trainf = nl.train.train_bfgs


Acctrain, Acctest = 0,0
error = net.train(X, yin, epochs=175, show=25, goal = 200)
#training

restrain = net.sim(X)
restrain = pd.Series(restrain.round()[:,0])
ynum = pd.Series(ynum)
Acctrain = accuracy(restrain, ynum) #train accuracy

res = net.sim(Xtest)
res = pd.Series(res.round()[:,0])
ytestnum = pd.Series((ytest=='s')*1.0)
Acctest = accuracy(res, ytestnum) #test accuracy

print '5.b. ANN res_train:', Acctrain
print '5.c ANN res_test:', Acctest

# writing results to file
out = pd.concat((testeventid,res), axis = 1)
out.columns = ['EventId','Class']
out['Class'] = out['Class'].map({0:'b',1:'s'})
out.to_csv(fileDir+'results.csv', index = False)

print time.ctime()

#plot training process
pl.subplot(211)
pl.plot(error)
