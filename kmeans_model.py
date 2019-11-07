#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.types import *

RANDOM_SEED   = 1
CLUSTERS      = 2 # Binary classification (malignant/benign)
APP_NAME      = 'WDBC_KMeans'
DATA_FILE     = 'data/wdbc.data'
OUTPUT_FOLDER = 'models/kmeans'

if __name__ == '__main__':
    # Set up the contexts
    sc = SparkContext(appName=APP_NAME)
    sql = SQLContext(sc)

    # Load the data file and cast the headers
    # into a dataframe
    data_file = sql.read.csv(DATA_FILE, header=False)

    # Convert all usable values (except the label) into floats
    data_file_converted = data_file.select(*(data_file[c].cast('float').alias(\
        c) for c in data_file.columns[2:]))

    # Create a set of LabeledPoints consisting
    # of <class, (rest of rows)>
    labeled_set = data_file.rdd.map(lambda row: LabeledPoint(row[1], row[2:]))

    # Train a KMeans model using the labelled set
    clusters = KMeans.train(labeled_set, CLUSTERS, seed=RANDOM_SEED)
    print(f'The cluster centres are:\n{clusters.centers}')

    # Save the model to the models folder
    clusters.save(sc, OUTPUT_FOLDER)
