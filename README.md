# wdbc-visualiser
A KMeans and Ensemble Learning model for visualising the Wisconsin Diagnostic Breast Cancer dataset.

The dataset is available [here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29). You should place `wdbc.data` into a folder named `data` located in the root folder of this repository. If not, you can modify the `DATA_FILE` variable in the script to point to another location.

You need to have Apache Spark installed to run this script, along with the requirements specified in the Python script. You can `pip install` from the `requirements.txt` file in this repository if you want to.

You can run this script by running:
```bash
spark-submit model.py
```

The saved models are output to files in a folder named `models` which will be created if it does not exist. It is located in the root folder of this repository.
