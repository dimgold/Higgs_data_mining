''' 
Data Mining Course - Final Project
Dima Goldenberg
Ori Barkan
dimgold@gmail.com
'''

fileDir = '//'

from scipy import stats #
import numpy as np #
from sklearn import tree #
from sklearn.neighbors import KNeighborsClassifier #
from sklearn.covariance import empirical_covariance #
import pandas as pd #




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
    '''calculates fisher ratio'''
    return (np.mean(a)-np.mean(b))**2 / (np.var(a) + np.var(b))
    
def accuracy(predict,classification):
    '''calculates accuracy ratio'''
    GoodPrediction  = (predict == classification)
    return GoodPrediction.value_counts()[True]*1.0 / len(GoodPrediction)
        
 
print 'Data Mining Final Project - Ori and Dima, Autumn 2015'
print ' '
      
# 1.a - spliting the data    
traindata = read_data('Higgs_Train.csv', fileDir)  
Signal = splitframe(traindata,'s')
Back = splitframe(traindata,'b')

#1.b - T-Test on the features
print ('1.b - T-Test on the features')
TTests =[]
for Col in Signal.columns[:-1]:
    T = stats.ttest_ind(Signal[Col],Back[Col]) #performing T-test
    TTests.append((T[1],Col))
    if T[1] > 0.01:
        print '1.b p-val for feature named:',Col, 'is: ' ,T[1]
print ''

#1.c - Fisher on the features
print ('1.c - Fisher on the features')
Fishers = []
for Col in Signal.columns[:-1]:
    F = fisher_ratio(Signal[Col],Back[Col])
    Fishers.append((F,Col))
Fishers.sort(reverse = True)
fishprint =[]
for col in Fishers[:4]:
    fishprint.append([col[0],col[1]])
    
print '1.c Array of FDR per feature: ', fishprint
#for col in Fishers[:4]:
#    print col[1], " Fisher's Ratio: ", col[0]

#1.d TTest vs Fisher
print ''
print ('1.d - TTest vs Fisher')
Fishers.sort()
TTests.sort(reverse = True)
for i in range(len(Fishers)):
    print 'Worse #',i+1,'T-Test is', TTests[i][1],'and Fishers is', Fishers[i][1]
    


# 2 - K-Nearest
traindata = read_data('Higgs_subTrain.csv',fileDir)
testdata = read_data('Higgs_subTest.csv',fileDir)  
Ytrain =  traindata.iloc[:,-1]
Ytest = testdata.iloc[:,-1]

# Data Min-Max Normalization
trainNorm = traindata.iloc[:,:-1]
testNorm = testdata.iloc[:,:-1]
trainNorm[:] = trainNorm.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
testNorm[:] = testNorm.apply(lambda x: (x - x.min()) / (x.max() - x.min()))



# regular Euclidian distance metric
print ''
print '2 - KNN'
neigh = KNeighborsClassifier(n_neighbors=1, metric='minkowski', p=2)
neigh.fit(trainNorm,Ytrain) 


Accuracy1 = accuracy(neigh.predict(testNorm), Ytest)
print '2.b  K=1 result :', Accuracy1


print '      K=1 train result :', accuracy(neigh.predict(trainNorm), Ytrain)

print ''
neigh = KNeighborsClassifier(n_neighbors=11, metric='minkowski', p=2)
neigh.fit(trainNorm,Ytrain) 

Accuracy11 =  accuracy(neigh.predict(testNorm), Ytest)
print '2.c K=11 result :', Accuracy11

print '     K=11 train result :',accuracy(neigh.predict(trainNorm), Ytrain)



# 3 - Decision tree
print ''
print '3 - Decision trees + features selection'
traindata = read_data('Higgs_Train.csv',fileDir)
testdata = read_data('Higgs_Test.csv',fileDir) 

X = traindata.iloc[:,:-1]
ytrain = traindata.iloc[:,-1]


candidates = Fishers[-3:]
candidates.sort(reverse=True)
emp_cov = empirical_covariance(X)
features = list(X.columns[:])
cols = []
for i in range(len(candidates)): #printing covarience
    cols.append(candidates[i][1])
    for j  in range(i+1,len(candidates)):
        ro = emp_cov[features.index(candidates[i][1]),features.index(candidates[j][1])] / np.sqrt(emp_cov[features.index(candidates[i][1]),features.index(candidates[i][1])]*emp_cov[features.index(candidates[j][1]),features.index(candidates[j][1])])
        print  candidates[i][1],", ", candidates[j][1], 'ro is:', ro


Xtrain =X[cols] #selecting rellevant features
Xtest = testdata[cols]
ytest = testdata.iloc[:,-1]

clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 9 )
clf.fit(Xtrain,ytrain)
print ''
#checking tree on train
AccuracyTreetrain = accuracy(clf.predict(Xtrain), ytrain)
print '3.b Tree res_train:', AccuracyTreetrain
#checking tree on train
AccuracyTreetest = accuracy(clf.predict(Xtest), ytest)
print '3.b Tree res_test:', AccuracyTreetest


# 4 - Forest
print ''
print '4 - Forest'

#training 3 trees
clf1 = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 9 )
clf1.fit(Xtrain.iloc[:,[0,1]],ytrain)

clf2 = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 9 )
clf2.fit(Xtrain.iloc[:,[0,2]],ytrain)

clf3 = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 9 )
clf3.fit(Xtrain.iloc[:,[1,2]],ytrain)

# majority vote
#checking forest on train
res = np.column_stack((clf1.predict(Xtrain.iloc[:,[0,1]]),clf2.predict(Xtrain.iloc[:,[0,2]]),clf3.predict(Xtrain.iloc[:,[1,2]])))
ressum = stats.mode(res, axis = 1)[0] # mode of each row
ressum = ressum.transpose()[0] 

AccuracyForesttrain = accuracy(ressum, ytrain)
print '4.b Forest res_train:', AccuracyForesttrain


#checking forest on test
res = np.column_stack((clf1.predict(Xtest.iloc[:,[0,1]]),clf2.predict(Xtest.iloc[:,[0,2]]),clf3.predict(Xtest.iloc[:,[1,2]])))
ressum = stats.mode(res, axis = 1)[0]
ressum = ressum.transpose()[0]
rs  = stats.mode(res, axis = 1)[1] # vote results
AccuracyForest = accuracy(ressum, ytest)
print '4.b Forest res_test:', AccuracyForest

