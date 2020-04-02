# C4.5 Decision Tree
## Introducion
An implement of the classic ML algorithm C4.5 decision tree.
## Quick Start
First to import the c45 modules, and initialise.
```
from c45 import MetricGenerator
from c45 import DecisionTree

tree = DecisionTree()
```
To apply the decision tree on your own dataset, you need to rigister metric functions into an object of ```MetricGenerator``` class.

There are some templates in ```MetricGenerator```, for nominal, ordinal, and contineous feature attributes. You can easily create a set of binary classify metrics and automatically register it. Besides, it's also possible to code your own metric functions and register them into the generator.

Here is am example:
```
feature = "color"
values_set = ["red", "blue", "green"]
metrics_set = MetricGenerator.nominal_template(feature,values_set)
tree.metric_generator.register(feature, metrics_set)
```
read data into the root node of the tree object
```
FILE_PATH = "dataset.csv"
tree.read_data(FILE_PATH)
```
train the tree
```
tree.train()
```
run on the testing set
```
TEST_FILE_PATH = "testset.csv"
precision = tree.run_test(TEST_FILE_PATH)
print("error rate on test set: ", 100 * (1 - precision), "%")
```
## Dependencies
The c45 module actually only need numpy and python3 to run.
1. numpy
2. python 3

In the example main.py, I used ```pandas``` and ```sklearn``` to split data into training set and testing set. However, they are not necessary for the algorithm. You can use native methods to process data set.
## Example
There is an example on iris data set. You can check it in main.py.
## Roadmap
- Visualise the decision tree.