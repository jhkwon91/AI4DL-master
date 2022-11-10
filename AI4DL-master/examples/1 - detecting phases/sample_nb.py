import inspect
import json
import pickle

import random
import numpy as np 
import pandas as pd
import numexpr as ne
import sklearn
from sklearn import preprocessing
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
import copy

from matplotlib import pyplot as plt 

from joblib import dump, load

import os,sys,inspect

sys.path.append('../../package')

import AI4DL
from AI4DL import AI4DL

#training_file = "training.csv"
testing_file = "sample.csv"

selected_features = ["cpu.usage", "cpu.usage.cores", "cpu.usage.pct", "cpu.usage.pct.container.requested", "mem.current", "mem.limit", "mem.usage.pct", "mem.working.set"]
container_id = "pod.id"

n_hidden  = 10
n_history  = 3
learning_rate = 0.001
n_epochs = 100
n_clusters = 5

crbm_save = "ai4dl_crbm1"
kmeans_save = "ai4dl_kmeans1.joblib"
scaler_save = "ai4dl_scaler1.joblib"

palette = {0:'grey', 1:'green', 2:'blue', 3:'red', 4:'orange'}
cpu_idx = 0 # cpu.usage
mem_idx = 4 # mem.current

ai4dl1 = AI4DL.AI4DL()
#ai4dl1.LoadTrainingDataset(training_file, selected_features, container_id)

sample = pd.read_csv(testing_file)
sample.head()

ai4dl1.LoadModel(crbm_save, kmeans_save, scaler_save)
ai4dl1.exec_id = container_id
ai4dl1.features = selected_features
ai4dl1.n_features = len(selected_features)

list_of_timeseries_ts = ai4dl1.TransformData(testing_file)
predicted_seq_phases_ts = ai4dl1.Predict(list_of_timeseries_ts)

selected_exec = 0
ai4dl1.PrintTrace(list_of_timeseries_ts, predicted_seq_phases_ts, selected_exec, cpu_idx, mem_idx, palette, col_names = ["CPU usage", "MEM usage"], f_name = "trace_11.png")

ai4dl1.PrintVarAnalysis(list_of_timeseries_ts, predicted_seq_phases_ts, cpu_idx, mem_idx, palette, col_names = ["CPU variation", "MEM variation"], f_name = "variance.png")
