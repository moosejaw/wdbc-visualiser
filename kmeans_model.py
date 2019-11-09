#!/usr/bin/python3
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sklearn.metrics as metrics

from os import listdir

from pyspark.sql.types import *
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.clustering import KMeans, KMeansModel

# Look into SVD!
#
# Take a sample of the dataset to train/test with and then measure
# precision/recall
#
# Plot an ROC curve

RANDOM_SEED      = 54321
CLUSTERS         = 2 # Binary classification (malignant/benign)
APP_NAME         = 'WDBC_KMeans'
DATA_FILE        = 'data/wdbc.data'
OUTPUT_FOLDER    = 'models/'
OUTPUT_MODEL     = f'{OUTPUT_FOLDER}kmeans'
CONV_DATA_FILE   = 'data/wdbc_converted.data'
GRAPH_OUTPUT     = 'output/kmeans_roc.png'
CLUSTER_OUTPUT   = 'output/clusters.txt'
DATASET_OUTPUT   = 'output/data_set.txt'
TRAIN_TEST_SPLIT = [0.7, 0.3] # Ratio of train data to test data i.e. 70%/30%

if __name__ == '__main__':
    # Set up the contexts
    sc = SparkContext(appName=APP_NAME)
    sc.setLogLevel("ERROR")
    sql = SQLContext(sc)

    # Open the file again and convert the classes into floats
    # Benign -> 0.00 and Malignant -> 1.00
    counts = {'malignant': 0, 'benign': 0}
    dataset_size = 0 # Number of lines in the file
    with open(DATA_FILE, 'r') as dataf:
        for line in dataf:
            dataset_size += 1
            if line.split(',')[1] == 'M':
                counts['malignant'] += 1
            elif line.split(',')[1] == 'B':
                counts['benign'] += 1

    print(f'There are {counts["malignant"]} malignant classes and \
    {counts["benign"]} benign classes out of a total of {dataset_size}')

    # Load the data file into a dataframe
    data_file = sql.read.csv(DATA_FILE, header=False)

    # Create a set of Vectors based on the usable features
    vector_set = data_file.rdd.map(lambda row: Vectors.dense(row[2:]))
    training_vector_set, testing_vector_set = vector_set.randomSplit(
        TRAIN_TEST_SPLIT,
        seed=RANDOM_SEED)

    # Get the size of the testing vector set
    testing_size = testing_vector_set.count()
    print(f'There are {testing_size} pieces of testing data.')

    # Now map the vectors and their true classes into a dict
    classes = {'M': [], 'B': []}
    for row in data_file.rdd.collect():
        if row[1] == 'M':
            classes['M'].append(Vectors.dense(row[2:]))
        else:
            classes['B'].append(Vectors.dense(row[2:]))

    # Train a KMeans model using the labelled set
    clusters = KMeans.train(training_vector_set, CLUSTERS, seed=RANDOM_SEED)
    print(f'The cluster centres are:\n{clusters.centers}')

    with open(CLUSTER_OUTPUT, 'w') as f:
        for i in clusters.centers:
            f.write(str(i) + '\n')

    # Save the model to the models folder
    if not listdir(OUTPUT_FOLDER):
        clusters.save(sc, OUTPUT_FOLDER)

    print('\n\nThe testing data produced the following predictions:')
    predictions = clusters.predict(testing_vector_set)
    print(predictions.collect())

    # Now we will compare the accuracy of each classification
    # We will test when 0 == 'M' and 0 == 'B' and take the one with
    # the highest number of correct classifications as the label
    classifiers = ['M', 'B']
    scores = {classifiers[0]: 0, classifiers[1]: 0}
    highest = '' # Represents the label that 0 is equal to
    zipped_data = list(zip(predictions.collect(), testing_vector_set.collect()))

    for label in classifiers:
        for row in zipped_data:
            if row[1] in classes[label] and row[0] == 0:
                scores[label] += 1

    # Get what the label actually means
    print(f'\n\nNumber of correct instances where 0 == M: {scores["M"]}')
    print(f'Number of correct instances where 0 == B: {scores["B"]}')
    if scores['M'] > scores['B']:
        highest = 'M'
        lowest = 'B'
    elif scores['M'] == scores['B']:
        highest = 'both'
        lowest = 'both'
    else:
        highest = 'B'
        lowest = 'M'

    print(f'The label means that a prediction of 0 == {highest}')

    # Now check the number of correct classifications to get the accuracy
    # And we will assume 'positive' == 'M' to get sensitivity and specificity
    #
    # classes[highest] represents whether a prediction of 0 is 'M' or 'B'
    if highest == 'both':
        print('The accuracy is the worst case scenario at 50%.')
    else:
        correct_classifications = 0
        true_positives  = 0
        true_negatives  = 0
        false_positives = 0
        false_negatives = 0
        true_array      = []
        pred_array      = []

        for row in zipped_data:
            # Populate array of predictions vs. true values
            # for sklearn to evaluate
            pred_array.append(row[0])

            # Remember... if a prediction of 0 == benign == negative
            # then they will need to be swapped if 0 == malignant == positive
            if highest == 'B':
                if row[1] in classes[highest]:
                    true_array.append(0)
                else:
                    true_array.append(1)
            else:
                if row[1] in classes[highest]:
                    true_array.append(1)
                else:
                    true_array.append(0)

            # Count the number of TPs, TNs, FPs, FNs
            if row[0] == 0 and row[1] in classes[highest]:
                correct_classifications += 1
                if highest == 'B':
                    true_negatives += 1
                else:
                    true_positives += 1
            elif row[0] == 1 and row[1] in classes[lowest]:
                correct_classifications += 1
                if highest == 'B':
                    true_positives += 1
                else:
                    true_negatives += 1
            elif row[0] == 0 and row[1] in classes[lowest]:
                if highest == 'B':
                    false_negatives += 1
                else:
                    false_positives += 1
            elif row[0] == 1 and row[1] in classes[highest]:
                if highest == 'B':
                    false_positives += 1
                else:
                    false_negatives += 1
            else:
                print('Could not classify a row!')

        # Calculate the results and print them!
        sensitivity = true_positives / (true_positives + false_negatives)
        specificity = true_negatives / (true_negatives + false_positives)
        accuracy    = correct_classifications / testing_size * 100
        print(f'\n\nDataset size: {testing_size}')
        print(f'No. of correct classifications: {correct_classifications}')
        print(f'No. of true positives: {true_positives}')
        print(f'No. of true negatives: {true_negatives}')
        print(f'No. of false positives: {false_positives}')
        print(f'No. of false negatives: {false_negatives}')
        print(f'\nAccuracy: {accuracy}%.')
        print(f'Sensitivity: {sensitivity}')
        print(f'Specificity: {specificity}')
        print(f'\n\"Positive\" indicates a classification of \"malignant\"')

        # If prediction of 0 == benign, then a positive
        # label (malignant) is 1
        #
        # Else a positive label is 0
        pos_lbl = 0
        if highest == 'B':
            pos_lbl = 1

        # Sanity check: check correct classifications matches
        # Uncomment below to run this check
        #
        #zipped_classes = list(zip(true_array, pred_array))
        #checks = 0
        #for row in zipped_classes:
        #    if row[0] == row[1]:
        #        checks += 1
        #print(f'There are {checks} correct classifications from the array vs. {correct_classifications} from the manual check.')

        # Calculate TPR, FPR and AUROC
        fpr, tpr, thresholds = metrics.roc_curve(true_array, pred_array, \
            pos_label=pos_lbl)
        auc = metrics.auc(fpr, tpr)

        # Plot the ROC curve
        plt.title("ROC Curve")
        plt.plot(fpr, tpr, 'b', color='darkorange', label=f'AUC = {auc}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Sensitivity')
        plt.xlabel('1 - Specificity')
        plt.legend(loc='lower right')
        plt.savefig(GRAPH_OUTPUT)
        print(f'\n\nSaved curve plot to {GRAPH_OUTPUT}')
