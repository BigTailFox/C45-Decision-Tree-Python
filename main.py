# coding=utf-8

from datetime import datetime
import c45
from c45 import Node
from c45 import DecisionTree
from c45 import MetricGenerator
import numpy
import csv
import pandas as pd
from sklearn import model_selection as skms

DATA_SET = "iris_data_set/iris1.csv"
TRAINING_SET = "iris_data_set/iris_tr.csv"
TESTING_SET = "iris_data_set/iris_te.csv"

# split dataset into training set and testing set.
data_set = pd.read_csv(DATA_SET)
data_set.describe()
now = int(datetime.now().strftime("%H%M%S"))
training_set, testing_set = skms.train_test_split(
    data_set, train_size=0.8, random_state=now)
training_set.to_csv(TRAINING_SET, index=False)
testing_set.to_csv(TESTING_SET, index=False)

if __name__ == "__main__":

    # initialize decision tree, and pull entries in the TRAINING_SET into root node.
    tree = DecisionTree(TRAINING_SET)

    # we need all values's ordered set of each feature, to generate metric functions.
    sepal_length_list = []
    sepal_width_list = []
    petal_length_list = []
    petal_width_list = []
    for item in tree.root.items:
        sepal_length_list.append(item.sepal_length)
        sepal_width_list.append(item.sepal_width)
        petal_length_list.append(item.petal_length)
        petal_width_list.append(item.petal_width)

    # rigister splitting metric functions for each feature,
    # use template for contineous attributes in the MetricGenerator class.
    m1 = tree.generator.contineous_template("sepal_length", *sepal_length_list)
    m2 = tree.generator.contineous_template("sepal_width", *sepal_width_list)
    m3 = tree.generator.contineous_template("petal_length", *petal_length_list)
    m4 = tree.generator.contineous_template("petal_width", *petal_width_list)

    # min gain ratio should every splitting get,
    # min number of items supporting splitting,
    # max tree depth, considering root node.
    tree.set_super_parameters(0, 8, 6)
    # train.
    tree.train()
    print("error rate on trainning set: ",
          100 * (1 - tree.run_test(TRAINING_SET)), "%")
    print("error rate on testing set:   ",
          100 * (1 - tree.run_test(TESTING_SET)), "%")
