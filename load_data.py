import numpy as np
import pandas as pd

def loaddata(name):
    dir = 'FUFS/upload/'
    file_path = dir + name + '.csv'
    df = pd.read_csv(file_path, header=None, dtype=float)
    df = np.array(df)
    label = df[:, -1]
    data = np.delete(df, -1, 1)

    info_path = dir + name + '_info.txt'
    f = open(info_path, "r")
    lines = f.readlines()
    protect_attribute = int(lines[0])
    n_cluster = int(lines[1])
    protect_values = lines[2].split('\t')
    protect_values = [int(i) for i in protect_values]

    return data, protect_attribute, label, n_cluster, protect_values