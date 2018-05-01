# -*- coding: utf-8 -*-
"""
iris：鸢尾花卉数据集 150个数据集，分为3类，每类50个数据，
每个数据包含4个属性 花萼长度，花萼宽度，花瓣长度，花瓣宽度
iris数据集从UCI datasets下载的文件为iris.data
安装notepad++，用notepad++打开iris.data，将其另存为iris.csv文件 方便导入
"""

import pandas as pd
df_iris=pd.read_csv('./iris.csv',delimiter=',',header=None)#读入csv数据 逗号分隔 无表头
df_iris_x=df_iris.drop([4],axis=1)  #dataframe 删除指定列
x=df_iris_x.values  # dataframe类型转为矩阵 x相当于特征向量
#print(x)
y_word=list(df_iris[4].values)  #类别标签 这是个字符串类型的单词，需要转成数字作为类别标签
#print(y_word)

#把类别标签变成0,1,2
label=['Iris-setosa','Iris-versicolor','Iris-virginica'] 
y=[]
for i in range(0,len(y_word)):
    for j in range(0,len(label)):
        if(y_word[i]==label[j]):
            y.append(j)
    
# print(y)
'''
sklearn有导入数据集的模块。可以直接导入iris
只有一小部分常用数据集可以这样做。
'''
#利用load_iris导入iris数据集 
'''
from sklearn import datasets
iris = datasets.load_iris()
x = iris.data                       
y = iris.target
'''

#利用sklearn进行分类
from sklearn import neighbors
from sklearn.svm import SVC

#clf = neighbors.KNeighborsClassifier(3) #模型定义 k近邻分类
clf=SVC(kernel="linear", C=0.025) #模型定义
clf.fit(x, y) #模型的训练  全部数据都参与训练
acc=clf.score(x, y) # 预测的准确率 
#print(acc)

#自定义函数
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
def classify_predict(x,y):
    names = ["Nearest Neighbors", "RBF SVM", "Decision Tree",
             "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
             "Quadratic Discriminant Analysis"] #没什么大用，只是为了让人知道用的什么模型
    classifiers = [     #模型的定义 下面都是sklearn里的函数 
        KNeighborsClassifier(3),
        #SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()]
    for name, clf in zip(names, classifiers): 
        print(name) #输出模型名称
        clf.fit(x,y) #模型训练
        expected=y  # 期望值
        predicted=clf.predict(x) #预测值
        acc=clf.score(x,y)  #准确率
        print("Classification report for classifier %s:\n%s" %(name, metrics.classification_report(expected, predicted)))# 输出预测指标 
        print("acc:%s\n" %(acc)) # 输出预测准确率
    return predicted,expected

predicted,expected=classify_predict(x,y) #调用自定义函数

      
    
    

