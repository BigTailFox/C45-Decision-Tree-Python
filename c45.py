# coding=utf-8

# leoherz_liu@163.com 2020/03/28
# data mining course, decision tree c4.5

import numpy
import csv
import math
import collections
import itertools


class Node(object):
    """
    c4.5 decision tree's node class.
    """

    def __init__(self, item=None, left=None, right=None):
        """
        initialize a tree node.
        """
        self.items = []
        self.childs = {}
        self.categories = {}
        self.metric_function = None
        self.feature = None
        self.depth = 0
        # every splitting should have at least LEAST_GAIN's entropy decreased.
        self.LEAST_GAIN = 0
        if item is not None:
            self.add_item(item)

    def add_item(self, *items):
        """
        add an item to current node.
        """
        for item in items:
            self.items.append(item)
            if item.label in self.categories:
                self.categories[item.label] += 1
            else:
                self.categories[item.label] = 1

    def split(self, metric_func):
        """
        split current node with the given metric function.
        """
        for item in self.items:
            flag = metric_func(item)
            if flag in self.childs:
                self.childs[flag].add_item(item)
            else:
                self.childs[flag] = Node(item)
                self.childs[flag].depth = self.depth + 1

    def cal_entropy(self):
        """
        calculate current node's info-entropy.
        """
        ent = 0
        item_num = len(self.items)
        for num in self.categories.values():
            p = num / item_num
            ent -= p * math.log2(p)
        return ent

    def cal_entropy_gain(self):
        """
        calculate info-gain between current node and it's childs.
        """
        total_ent = 0
        split_ent = 0
        item_num = len(self.items)
        for node in self.childs.values():
            p = len(node.items) / item_num
            total_ent += node.cal_entropy() * p
            split_ent -= p * math.log2(p)
        if split_ent != 0:
            return (self.cal_entropy() - total_ent) / (split_ent)
        else:
            return 0  # means split entropy be 0

    def find_split_metric(self, metric_generator):
        """
        test all possible splittings given by the given metric generator,
        to find the best splitting of the current situation.
        """
        # initialise value means: splitting operation has to get as least LEAST_GAIN's benifit, or should stop.
        max_gain = self.LEAST_GAIN
        best_feature = None
        best_metric = None
        for feature in metric_generator.features:
            # print("now search: ", feature)
            metrics = metric_generator.get_metrics(feature)
            cnt = 0
            for metric in metrics:
                self.childs.clear()
                self.split(metric)
                info_gain = self.cal_entropy_gain()
                cnt += 1
                # print("try {:<2d} - split, get gain: {:>.8f}".
                #       format(cnt, info_gain))
                if info_gain == None:
                    continue
                if info_gain > max_gain:
                    best_feature = feature
                    best_metric = metric
                    max_gain = info_gain
        if best_feature is not None:
            self.feature = best_feature
            self.metric_function = best_metric


class DecisionTree(object):
    """
    c4.5 decision tree class.
    """

    def __init__(self, file_path=None):
        """
        initialize a c4.5 decision tree.
        optional data's csv file path can be given.
        """
        self.root = Node()
        self.generator = MetricGenerator()
        if file_path is not None:
            self.read_data(file_path)
        # only to apply splitting on node that has at least MIN_SPLIT_NUM items.
        self.MIN_SPLIT_NUM = 5
        # grow a tree under the depth of MAX_TREE_DEPTH.
        self.MAX_TREE_DEPTH = 100

    def set_super_parameters(self, min_gain, min_split_num, max_depth):
        self.root.LEAST_GAIN = min_gain
        self.MIN_SPLIT_NUM = min_split_num
        self.MAX_TREE_DEPTH = max_depth

    def read_data(self, file_path):
        """
        give a csv file_path, first row should be data's features.
        read data into root node of the tree.
        read features into feature set.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            f_csv = csv.reader(file)
            headers = next(f_csv)
            len_of_row = len(headers)
            Item = collections.namedtuple('Item', headers)
            for row in f_csv:
                if len(row) != len_of_row:
                    continue
                for i in range(len_of_row - 1):
                    row[i] = float(row[i])
                item = Item(*row)
                self.root.add_item(item)

    def train(self):
        """
        train decision tree.
        """
        if self.root and self.root.items:
            self.__train(self.root)

    def __train(self, node):
        """
        recursively train decision tree. greedy find a split getting best info gain.
        @this method shouldn't be called by users.
        """
        if self.__test_stop(node):
            leaf_res = {v: k for k, v in node.categories.items()}
            node.label = leaf_res[max(leaf_res.keys())]
        else:
            node.find_split_metric(self.generator)
            # when found no splitting can get gain, stop.
            if node.feature == None:
                leaf_res = {v: k for k, v in node.categories.items()}
                node.label = leaf_res[max(leaf_res.keys())]
                print("stop: no gain")
            else:
                node.childs.clear()
                node.split(node.metric_function)
                print("finally split by:", node.feature)
                for child in node.childs.values():
                    print("child: ", child.categories)
                for child in node.childs.values():
                    self.__train(child)

    def __test_stop(self, node):
        """
        test if the growth of a tree should be stopped.
        @this method shouldn't be called by users.
        """
        # when number of node's items less than MIN_SPLIT_NUM, stop
        if len(node.items) <= self.MIN_SPLIT_NUM:
            print("stop: not enough items")
            return True
        # when tree's depth reaches MAX_TREE_DEPTH, stop
        elif node.depth >= self.MAX_TREE_DEPTH:
            print("stop: depth")
            return True
        # when node's items has the same label, stop
        elif len(node.categories) == 1:
            print("stop: one class")
            return True
        else:
            return False

    def run_test(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            f_csv = csv.reader(file)
            headers = next(f_csv)
            len_of_row = len(headers)
            Item = collections.namedtuple("Item", headers)
            T = 0
            F = 0
            k = 0
            for row in f_csv:
                k += 1
                if len(row) != len_of_row:
                    continue
                for i in range(len_of_row - 1):
                    row[i] = float(row[i])
                item = Item(*row)
                predict_label = self.classify(item)
                if predict_label == item.label:
                    T += 1
                else:
                    F += 1
            return float(T / (T + F))

    def classify(self, item):
        """
        give a item, run the classifier. out put the predictive label.
        """
        current = self.root
        while(current.metric_function):
            flag = current.metric_function(item)
            current = current.childs[flag]
        return current.label

    def post_prune(self):
        #TODO: prune
        pass


class MetricGenerator(object):
    """
    metric factory class.
    @users should register metric functions of data's features first before using.
    """

    def __init__(self):
        self.metrics_dict = {}
        self.features = []

    def rigister(self, feature, maker):
        """
        register a feature-metric pair into the generator.
        @for nomimal
        @for ordinal
        @for contineous
        """
        self.metrics_dict[feature] = list(maker)
        self.features.append(feature)

    def get_metrics(self, feature):
        """
        return a metric list of given feature.
        """
        if feature in self.metrics_dict:
            return self.metrics_dict[feature]
        else:
            print("rigister metric functions for feature: ", feature, " firtst.")

# TODO: template
    def nominal_template(self, feature, *values):
        """
        given a nominal feature and it's values set. return a metric list.
        """
        def metrics_maker(feature):
            for i in range(i, len(values) - 1):
                combinations = itertools.combinations(values, i)
                for combination in combinations:
                    yield lambda item, combination=combination: getattr(item, feature) in combination
        self.rigister(feature, metrics_maker(feature))

    def ordinal_template(self, feature, *values):
        """
        given an ordinal feature and it's values. return a metric list.
        """
        values = values.sort()

        def metrics_maker(feature):
            for i in range(1, len(values)):
                yield lambda item, i=i: getattr(item, feature) < values[i]
        self.rigister(feature, metrics_maker(feature))

    def contineous_template(self, feature, *values):
        """
        given a contineous feature and it's values. return a metric list.
        """
        values = list(set(values))
        num = len(values)
        vals = numpy.sort(numpy.array(values))
        vals = numpy.repeat(vals, 2, 0)
        vals[1:-1:2] += (vals[2::2] - vals[1:-1:2]) / 2

        def metrics_maker(feature):
            for i in range(num - 1):
                yield lambda item, i=i: getattr(item, feature) < vals[i * 2 + 1]
        self.rigister(feature, metrics_maker(feature))
