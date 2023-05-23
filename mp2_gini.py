# Starter code for CS 165B MP2
# Decision Truee


import os
import sys
import json
import numpy as np
import pandas as pd

from typing import List

# define the Node structure for Decision Tree
class Node:
    def __init__(self) -> None:
        self.left = None            # left child, a Node object
        self.right = None           # right child, a Node object
        self.split_feature = None   # the feature to be split on, a string
        self.split_value = None     # the threshould value of the feature to be split on, a float
        self.is_leaf = False        # whether the node is a leaf node, a boolean
        self.prediction = None      # for leaf node, the class label, a int
        self.ig = None              # information gain for current split, a float
        self.depth = None           # depth of the node in the tree, root will be 0, a int

class DecisionTree():
    """Decision Tree Classifier."""
    def __init__(self, max_depth:int, min_samples_split:int, min_information_gain:float =1e-5) -> None:
        """
            initialize the decision tree.
        Args:
            max_depth: maximum tree depth to stop splitting. 
            min_samples_split: minimum number of data to make a split. If smaller than this, stop splitting. Typcial values: 2, 5, 10, 20, etc.
            min_information_gain: minimum ig gain to consider a split to be valid.
        """
        self.root = None                                    # the root node of the tree
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_information_gain  = min_information_gain

    def fit(self, training_data: pd.DataFrame, training_label: pd.Series):
        '''
            Fit a Decission Tree based on data
            Args:
                training_data: Data to be used to train the Decission Tree
                training_label: label of the training data
            Returns:
                root node of the Decission Tree
        '''
        self.root = self.GrowTree(training_data, training_label, counter = 0)
        return self.root
  
    def GrowTree(self, data: pd.DataFrame, label: pd.Series, counter: int=0):
        '''
            Conducts the split feature process recursively.
            Based on the given data and label, it will find the best feature and value to split the data and return the node.
            Specifically:
                1. Check the depth and sample conditions
                2. Find the best feature and value to split the data by BestSplit() function
                3. Check the IG condition
                4. Get the divided data and label based on the split feature and value, and then recursively call GrowTree() to create left and right subtree.
                5. Return the node.  
            Hint: 
                a. You could use the Node class to create a node.
                b. You should carefully deal with the leaf node case. The prediction of the leaf node should be the majority of the labels in this node.

            Args:                                   
                data: Data to be used to train the Decission Tree                           
                label: target variable column name
                counter: counter to keep track of the depth of the tree
            Returns:
                node: New node of the Decission Tree
        '''
        node = Node()
        node.depth = counter

        # Check for depth conditions
        if self.max_depth == None:
            depth_cond = True
        else:
            depth_cond = True if counter < self.max_depth else False

        # Check for sample conditions
        if self.min_samples_split == None:
            sample_cond = True
        else:
            sample_cond = True if data.shape[0] > self.min_samples_split else False

        data_with_label = data.copy()
        data_with_label["label"] = label
         #so that labels match up
        if depth_cond & sample_cond:

            split_feature, split_value, ig = self.BestSplit(data, label)
            node.ig = ig

            # Check for ig condition. If ig condition is fulfilled, make split 
            if ig is not None and ig >= self.min_information_gain:

                node.split_feature = split_feature
                node.split_value = split_value
                counter += 1

                #TODO Get the divided data and label based on the split feature and value, 
                # and then recursively call GrowTree() to create left and right subtree.
                left_sub_data = data_with_label[data_with_label[split_feature] < split_value]
                right_sub_data = data_with_label[data_with_label[split_feature] >= split_value]
                left_sub_data = left_sub_data.reset_index(drop=True)
                right_sub_data = right_sub_data.reset_index(drop=True)
                if len(left_sub_data.index) <= self.min_samples_split or len(right_sub_data.index) <= self.min_samples_split:
                    vals = data_with_label["label"].value_counts(normalize=True)
                    if len(vals) == 1:
                        node.prediction = int(vals.loc[vals.index[0]])
                    elif (vals.loc[1] > vals.loc[2]): #checking whether alive or dead are larger proportion of total
                        node.prediction = 1
                    else:
                        node.prediction = 2
                    node.is_leaf = True
                else:
                    left_sub_data.drop(columns=split_feature, inplace=True)
                    right_sub_data.drop(columns=split_feature, inplace=True)
                    node.left = self.GrowTree(left_sub_data.iloc[:, :-1], left_sub_data.iloc[:,-1], counter=counter+1)
                    node.right = self.GrowTree(right_sub_data.iloc[:, :-1], right_sub_data.iloc[:,-1], counter=counter+1)
                    if node.left.prediction == node.right.prediction and node.left.is_leaf and node.right.is_leaf:
                        node.prediction = node.left.prediction
                        node.is_leaf = True
                        node.left = None
                        node.right = None

            else:
                # TODO If it doesn't match IG condition, it is a leaf node
                vals = data_with_label["label"].value_counts(normalize=True)
                if len(vals) == 1:
                    node.prediction = int(vals.loc[vals.index[0]])
                elif (vals.loc[1] > vals.loc[2]):
                    node.prediction = 1
                else:
                    node.prediction = 2
                node.is_leaf = True
        else:
            #if data_with_label.shape[0] == 0:
                #return
            #TODO If it doesn't match depth or sample condition. It is a leaf node
            vals = data_with_label["label"].value_counts(normalize=True)
            if len(vals) == 1:
                node.prediction = int(vals.loc[vals.index[0]])
            elif (vals.loc[1] > vals.loc[2]): #checking whether alive or dead are larger proportion of total
                node.prediction = 1
            else: 
                node.prediction = 2

            node.is_leaf = True
        print(node.prediction)
        return node
    
    def BestSplit(self, data: pd.DataFrame, label: pd.Series):
        '''
            Given a data, select the best split by maximizing the information gain (maximizing the purity)
            Args:
                data: dataframe where to find the best split.
                label: label of the data.
            Returns:
                split_feature: feature to split the data. 
                split_value: value to split the data.
                split_ig: information gain of the split.
        '''
        # TODO: Implement the BestSplit function
        split_feature, split_value, split_ig = None, None, None

        #joining data and labels
        data_with_label = data.copy()
        data_with_label["label"] = label
        tot_vals = data_with_label["label"].value_counts(normalize=True)
        before_split_entropy = 0
        
        for col in data.columns: #should now handle non-boolean values
            tot_gini = 0
            vals = data_with_label[col].value_counts(normalize=True)
            #need to check whether there are 2 or more than 2 values in value_counts, if more than 2 need to cycle through each possible split value and calculate the min 
            #entropy for each split, find the value that reduces the entropy the most, then continue.
            if len(vals) > 2:
                best_split = 0
                best_gini = 1
                for items in vals.index.sort_values(ascending=True):
                    left = data_with_label[data_with_label[col] < items]
                    right = data_with_label[data_with_label[col] >= items]
                    left_freq = left["label"].value_counts(normalize=True) #labels column to get frequencies
                    right_freq = right["label"].value_counts(normalize=True) #labels column to get frequencies
                    left_gini = 1
                    right_gini = 1
                    for p in left_freq:
                        left_gini -= p**2
                    for p in right_freq:
                        right_gini -= p**2
                    curr_gini = (len(left)/len(data_with_label))*(left_gini) + (len(right)/len(data_with_label))*(right_gini) #this could be wrong calculation
                    if curr_gini < best_gini:
                        best_gini = curr_gini
                        best_split = items
                    #calculate left and right entropy to get entropy split, update best_split and best_entr if necessary
            else: #boolean case
                best_split = 2
                left = data_with_label[data_with_label[col] < 2]
                right = data_with_label[data_with_label[col] >= 2]
                left_freq = left["label"].value_counts(normalize=True) #labels column to get frequencies
                right_freq = right["label"].value_counts(normalize=True) #labels column to get frequencies
                left_gini = 1
                right_gini = 1
                for p in left_freq:
                    left_gini -= p**2
                for p in right_freq:
                    right_gini -= p**2
                best_gini = (len(left)/len(data_with_label))*(left_gini) + (len(right)/len(data_with_label))*(right_gini)

            if split_feature == None:
                split_feature = col
                split_value = best_split
                split_ig = best_gini
            else:
                if tot_gini < split_ig:
                    split_ig = best_gini
                    split_feature = col
                    split_value = best_split

        return split_feature, split_value, split_ig




    def predict(self, data: pd.DataFrame) -> List[int]:
        '''
            Given a dataset, make a prediction.
            Args:
                data: data to make a prediction.
            Returns:
                predictions: List, predictions of the data.
        '''
        predictions = []
        # TODO: Implement the predict function
        for row in data.iterrows():
            temp = self.root
            while (not temp.is_leaf):
                if row[1][temp.split_feature] < temp.split_value:
                    temp = temp.left
                else:
                    temp = temp.right
            predictions.append(temp.prediction)
        return predictions
    
    def print_tree(self):
        '''
            Prints the tree.
        '''
        self.print_tree_rec(self.root)

    def print_tree_rec(self, node):
        '''
            Prints the tree recursively.
        '''
        if node is None:
            return 
        else:
            if node.is_leaf:
                print("{}Level{} | Leaf: {}".format(' '* node.depth, node.depth, node.prediction))
                return
            else:
                print("{}Level{} | {} < {} (ig={:0.4f})".format(' '* node.depth, node.depth, node.split_feature, node.split_value, node.ig))
                self.print_tree_rec(node.left)
                self.print_tree_rec(node.right)




def run_train_test(training_data: pd.DataFrame, training_labels: pd.Series, testing_data: pd.DataFrame) -> List[int]:
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Args:
        training_data: pd.DataFrame
        training_label: pd.Series
        testing_data: pd.DataFrame
    Returns:
        testing_prediction: List[int]
    """

    #TODO implement the decision tree and return the prediction
    dt = DecisionTree(15, 3, 0.00001)
    dt.fit(training_data, training_labels)
    testing_predictions = dt.predict(testing_data)
    dt.print_tree()

    #return [1]*len(testing_data)
    return testing_predictions

######################## evaluate the accuracy #################################

def cal_accuracy(y_pred, y_real):
    '''
    Given a prediction and a real value, it calculates the accuracy.
    y_pred: prediction
    y_real: real value
    '''
    y_pred = np.array(y_pred)
    y_real = np.array(y_real)
    print(sum(y_pred == y_real))
    if len(y_pred) == len(y_real):
        return sum(y_pred == y_real)/len(y_pred)
    else:
        print('y_pred and y_real must have the same length.')

################################################################################

if __name__ == "__main__":
    training = pd.read_csv('data/train.csv')
    dev = pd.read_csv('data/dev.csv')

    training_labels = training['LABEL']
    training_data = training.drop('LABEL', axis=1)
    dev_data = dev.drop('LABEL', axis=1)

    prediction = run_train_test(training_data, training_labels, dev_data)
    accu = cal_accuracy(prediction, dev['LABEL'].to_numpy())
    print(accu)