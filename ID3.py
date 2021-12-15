import pandas as pd
import math
import numpy as np
import csv
data = pd.read_csv("traintennis.csv")
features = [feat for feat in data]
features.remove("PlayTennis")   #has only features or headers

class Node:
    def __init__(self):
        self.children = []
        self.value = ""
        self.isLeaf = False
        self.prediction = ""  

def entropy(examples):
    pos = 0.0
    neg = 0.0
    for _, row in examples.iterrows():
        if row["PlayTennis"] == "Yes":
            pos += 1
        else:
            neg += 1
    if pos == 0.0 or neg == 0.0:
        return 0.0
    else:
        p = pos / (pos + neg)
        n = neg / (pos + neg)
        return -(p * math.log(p, 2) + n * math.log(n, 2))

def info_gain(examples, attr):
    uniq = np.unique(examples[attr])
    gain = entropy(examples)
    for u in uniq:
        subdata = examples[examples[attr] == u]
        sub_mul = entropy(subdata)
        gain -= (float(len(subdata)) / float(len(examples))) * sub_mul
    return gain

def ID3(examples, attrs):
    root = Node()
    max_gain = 0
    max_feat = ""
    for feature in attrs: #find info_gain for every feature
        gain = info_gain(examples, feature)
        print(feature,":",gain)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature  #obtaining max_gain and the respective feature to be considered for classification
    print("*)Feature considered for classification :",max_feat)
    print()
    root.value = max_feat
    uniq = np.unique(examples[max_feat]) #sunny,rain,overcast
    for u in uniq:
        subdata = examples[examples[max_feat] == u]
        if entropy(subdata) == 0.0:
            newNode = Node()
            newNode.isLeaf = True
            newNode.value = u
            newNode.prediction = np.unique(subdata["PlayTennis"])
            root.children.append(newNode)
        else:
            dummyNode = Node()
            dummyNode.value = u
            new_attrs = attrs.copy()
            new_attrs.remove(max_feat)
            child = ID3(subdata, new_attrs)
            dummyNode.children.append(child)
            root.children.append(dummyNode)
    return root
def printTree(root, depth):
    for i in range(depth):
        print("\t", end="")
    print(root.value, end="")
    if root.isLeaf:
        print("--> ", root.prediction)
    print()
    for child in root.children:
        printTree(child, depth + 1)
def classify(root,x_test,features):
    if root.isLeaf==True:
        print(root.prediction)
        print("\n")
        return
    if root.value not in features:  #root is a value and not an attribute/feature i.e ex: Sunny,Rainy,Overcast so we have to go its child nodes to chk for the next attribute that is used for classification
        for val in root.children:
            pos=features.index(val.value) 
            root=val  
    else:     #root is an attribute/feature
        pos=features.index(root.value)
    for v in root.children: 
        if x_test[pos]==v.value:
            classify(v,x_test,features)
            return
root = ID3(data, features)
printTree(root,0)
lines=csv.reader(open("testtennis.csv","r"))
testdata = list(lines)
headers = testdata.pop(0)
print("Features considered:",headers,"\n")
for xtest in testdata:
    print("The test instance:",xtest)
    print("The label for test instance:",end=" ")
    classify(root,xtest,headers)
