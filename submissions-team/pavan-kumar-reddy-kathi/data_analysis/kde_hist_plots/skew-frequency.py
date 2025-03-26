import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import os
from dpputility import  data_set_module as dsm, config_module as cm

# Load data set
dataset = dsm.get_data_frame()
# print(dataset.columns)




# calculate skew
# 0.5 - 1.0 Moderately positive
# > 1 highly positive
# -0.5 - -1 moderately negative
# < -1 highly negative
# print(skew(dataset['carat'])) # highly skewed -1.1166148681277792 not acceptable (-0.5 to 1.5)
# print(skew(dataset['depth'])) # -0.0822917377962474 acceptable (-0.5 to 0.5)
# print(skew(dataset['table'])) #0.7968736878796613 not acceptable value (-0.5 to 0.5)
# acceptable -1 to 1
# print(skew(dataset['width'])) #0.3786658120772073
# print(skew(dataset['height'])) # 1.5223802221853744
# print(skew(dataset['length'])) # 2.434099025011364
# print(skew(dataset['price'])) #1.6183502776053016

# draw histograms to confirm
numeric_columns = ['carat', 'depth', 'table', 'price',
       'width', 'height', 'length']
root_path = os.path.abspath('../../') #'../../docs'
folder_path = cm.read_config_setting('kde_hist_plots_folder')
folder_to_save = os.path.join(root_path, folder_path)
# print(folder_to_save)

for column in numeric_columns:
       plt.figure()
       sns.histplot(dataset[column], bins=10, edgecolor='black', alpha=0.7, kde=True) ,#
       plt.annotate(f"Skew={np.round(skew(dataset[column]),2)}", xy=(0.75,0.9), xycoords='axes fraction')
       plt.title(f"Hist Plot with kde for {column}")
       plt.xlabel('Value')
       plt.ylabel('Frequency')
       path_to_save = os.path.join(folder_to_save, f"Hist-KDE-{column}.png")
       plt.savefig(path_to_save)

