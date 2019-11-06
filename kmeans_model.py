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
DATA_SCHEMA = StructType([
    StructField('diagnosis', StringType(), True),
    StructField('radius', FloatType(), True),
    StructField('texture', FloatType(), True),
    StructField('perimeter', FloatType(), True),
    StructField('area', FloatType(), True),
    StructField('smoothness', FloatType(), True),
    StructField('compactness', FloatType(), True),
    StructField('concavity', FloatType(), True),
    StructField('concavepts', FloatType(), True),
    StructField('symmetry', FloatType(), True),
    StructField('fractaldim', FloatType(), True)
])

if __name__ == '__main__':
    # Set up the contexts
    sc = SparkContext(appName=APP_NAME)
    sql = SQLContext(sc)

    # Load the data file and cast the headers
    # into a dataframe
    data_file = sql.read.csv(
        header=False,
        inferSchema=False).map(
        # THIS WON'T WORK!!!!
        # Need to strip out the first comma and keep the rest of line intact
        lambda row: row.split(',')[1] # Strip out the 'id' row
        )
    df = data_file.createDataFrame(
        data_file,
        schema=DATA_SCHEMA
    )

    # Create a set of LabeledPoints consisting
    # of <class, (rest of rows)>
    labeled_set = df.rdd.map(
        lambda row: LabeledPoint(row[0], row[])
    )
