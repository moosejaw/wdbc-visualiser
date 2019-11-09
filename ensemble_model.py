import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sklearn.metrics as metrics

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel

APP_NAME         = 'WDBC_Ensemble'
DATA_FILE        = 'data/wdbc.data'
OUTPUT_FOLDER    = 'models/'
MODEL_FOLDER     = f'{OUTPUT_FOLDER}ensemble'
GRAPH_OUTPUT     = 'output/ensemble_roc.png'
RANDOM_SEED      = 12345
NUM_CLASSES      = 2 # Binary classification - malignant/benign
TRAIN_TEST_SPLIT = [0.7, 0.3]

if __name__ == '__main__':
    # Set up the contexts
    sc = SparkContext(appName=APP_NAME)
    sc.setLogLevel('ERROR')
    sql = SQLContext(sc)

    # Load the dataset into an RDD of LabeledPoints
    # and split into training and testing data
    data_set = sql.read.csv(DATA_FILE, header=False).rdd.map( \
        lambda row: LabeledPoint(0.0 if row[1] == 'B' else 1.0, Vectors.dense( \
            row[2:])))
    training_data_set, testing_data_set = data_set.randomSplit( \
        TRAIN_TEST_SPLIT, seed=RANDOM_SEED)

    # Train a RandomForest classifier based on the training data
    model = RandomForest.trainClassifier(training_data_set,
        numClasses=NUM_CLASSES,
        categoricalFeaturesInfo={},
        numTrees=3,
        featureSubsetStrategy='auto',
        impurity='gini',
        maxDepth=4,
        maxBins=32,
        seed=RANDOM_SEED)

    # Train some predictions and compare them with their true classes
    predictions = model.predict(testing_data_set.map(lambda row: row.features))
    classes_and_predictions = list(zip(testing_data_set.map(lambda row: \
        row.label).collect(), predictions.collect()))

    # Gather some statistics and print them
    testing_data_set_size = testing_data_set.count()
    correct_classifications = 0
    true_positives  = 0
    true_negatives  = 0
    false_positives = 0
    false_negatives = 0
    true_classes    = [i[0] for i in classes_and_predictions]
    pred_classes    = [i[1] for i in classes_and_predictions]

    for row in classes_and_predictions:
        if row[0] == 0.0 and row[1] == 0.0:
            correct_classifications += 1
            true_negatives += 1
        elif row[0] == 1.0 and row[1] == 1.0:
            correct_classifications += 1
            true_positives +=1
        elif row[0] == 0.0 and row[1] == 1.0:
            false_positives += 1
        elif row[0] == 1.0 and row[1] == 0.0:
            false_negatives += 1

    accuracy = correct_classifications / testing_data_set_size * 100
    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)
    print(f'\n\nTesting dataset size: {testing_data_set_size}')
    print(f'No. of correct classifications: {correct_classifications}')
    print(f'No. of true positives: {true_positives}')
    print(f'No. of true negatives: {true_negatives}')
    print(f'No. of false positives: {false_positives}')
    print(f'No. of false negatives: {false_negatives}')
    print(f'\nAccuracy of the model: {accuracy}')
    print(f'Sensitivity of the model: {sensitivity}')
    print(f'Specificity of the model: {specificity}')

    # Calculate TPR, FPR and AUROC
    fpr, tpr, thresholds = metrics.roc_curve(true_classes, pred_classes, \
        pos_label=1.0)
    auc = metrics.auc(fpr, tpr)

    # Plot the ROC curve
    plt.title("ROC curve for RandomForest classifier")
    plt.plot(fpr, tpr, 'b', color='darkorange', label=f'AUC = {auc}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Sensitivity')
    plt.xlabel('1 - Specificity')
    plt.legend(loc='lower right')
    plt.savefig(GRAPH_OUTPUT)
    print(f'\nSaved curve plot to {GRAPH_OUTPUT}')
