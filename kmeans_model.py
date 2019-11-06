#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.clustering import KMeans
from pyspark.sql.types import *

RANDOM_SEED = 1
APP_NAME    = 'WDBC_KMeans'
DATA_FILE   = 'data/wdbc.data'

if __name__ == '__main__':
    # Set up the contexts
    sc = SparkContext(appName=APP_NAME)
    sql = SQLContext(sc)

    # Load the data file and cast the headers
    # into a dataframe
    data_file = sql.read.csv('data/wdbc.data', header=False)

    # Convert all usable values (except the label) into floats
    data_file_converted = data_file.select(*(data_file[c].cast('float').alias(\
        c) for c in data_file.columns[2:]))

    # Create a set of LabeledPoints consisting
    # of <class, (rest of rows)>
    labeled_set = data_file.rdd.map(lambda row: LabeledPoint(row[1], row[2:]))
