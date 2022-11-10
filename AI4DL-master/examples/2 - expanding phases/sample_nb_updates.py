
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

selected_features = ["cpu.usage", "cpu.usage.cores", "cpu.usage.pct", "cpu.usage.pct.container.requested", "mem.current", "mem.limit", "mem.usage.pct", "mem.working.set"]
container_id = "pod.id"

n_clusters = 5

## Models from DLaaS
crbm_save = "ai4dl_crbm1"
kmeans_save = "ai4dl_kmeans1.joblib"
scaler_save = "ai4dl_scaler1.joblib"
sample_dlaas = "sample-dlaas.csv"

kmeans_new_save = "ai4dl_kmeans2.joblib"
sample_vr = "sample-vr.csv"

palette = {0:'grey', 1:'green', 2:'blue', 3:'red', 4:'orange', 5:'fuchsia', 6:'tan', 7:'lightsteelblue', 8:'black'}
cpu_idx = 0 # cpu.usage
mem_idx = 4 # mem.current

ai4dl1 = AI4DL.AI4DL()

ai4dl1.LoadModel(crbm_save, kmeans_save, scaler_save)
ai4dl1.exec_id = container_id
ai4dl1.features = selected_features
ai4dl1.n_features = len(selected_features)

list_of_timeseries_dlaas = ai4dl1.TransformData(sample_dlaas)
pred_seq_phases_dlaas = ai4dl1.Predict(list_of_timeseries_dlaas)

selected_exec = 0
ai4dl1.PrintTrace(list_of_timeseries_dlaas, pred_seq_phases_dlaas, selected_exec, cpu_idx, mem_idx, palette, col_names = ["CPU usage", "MEM usage"], f_name = "trace_dlaas.png")

ai4dl1.PrintVarAnalysis(list_of_timeseries_dlaas, pred_seq_phases_dlaas, cpu_idx, mem_idx, palette, col_names = ["CPU variation", "MEM variation"], f_name = "variance_dlaas.png")


sample  = pd.read_csv(sample_vr)
sample.head()


list_of_timeseries_vr = ai4dl1.TransformData(sample_vr)
predicted_seq_phases_vr = ai4dl1.Predict(list_of_timeseries_vr)


selected_exec = 0
ai4dl1.PrintTrace(list_of_timeseries_vr, predicted_seq_phases_vr, selected_exec, cpu_idx, mem_idx, palette, col_names = ["CPU usage", "MEM usage"], f_name = "trace_vr_pre.png")


ai4dl1.PrintVarAnalysis(list_of_timeseries_vr, predicted_seq_phases_vr, cpu_idx, mem_idx, palette, col_names = ["CPU variation", "MEM variation"], f_name = "variance_vr_pre.png")


target_cluster = 2


list_of_activations = []
for x in list_of_timeseries_vr:
    step_output = ai4dl1.pipeline.getStep(0).predict(x)
    list_of_activations.append(step_output)
X_activations = np.vstack(list_of_activations)



selected_activations = []
count = 0
for x in range(0, len(predicted_seq_phases_vr)):
    for y in range(0, len(predicted_seq_phases_vr[x])):
        if predicted_seq_phases_vr[x][y] == target_cluster:
            selected_activations.append(copy.deepcopy(X_activations[count]))
        count = count + 1
X4_activations = np.vstack(selected_activations)



kmeans_X4 = load(kmeans_new_save)



pred_seq_phases_X4 = copy.deepcopy(predicted_seq_phases_vr)

count = 0
for x in range(0, len(predicted_seq_phases_vr)):
    for y in range(0, len(predicted_seq_phases_vr[x])):
        if predicted_seq_phases_vr[x][y] == target_cluster:
            pred_seq_phases_X4[x][y] = n_clusters + kmeans_X4.predict([X4_activations[count]])[0]
            count = count + 1



selected_exec = 0
ai4dl1.PrintTrace(list_of_timeseries_vr,
                  pred_seq_phases_X4,
                  selected_exec, cpu_idx, mem_idx, palette, col_names = ["CPU usage", "MEM usage"], f_name = "trace_vr_post.png")



ai4dl1.PrintVarAnalysis(list_of_timeseries_vr, pred_seq_phases_X4, cpu_idx, mem_idx, palette, col_names = ["CPU variation", "MEM variation"], f_name = "variance_vr_post.png")
