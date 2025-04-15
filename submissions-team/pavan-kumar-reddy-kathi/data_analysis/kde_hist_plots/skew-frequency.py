import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

import os
from dpputility import  data_set_module as dsm, config_module as cm

# Load data set
dataset = dsm.get_data_frame()

# Skew ranges
# 0.5 - 1.0 Moderately Positive
# > 1 Highly Positive
# -0.5 - -1 Moderately Negative
# < -1 Highly Negative
# Acceptable -1 to 1

# Draw histograms to confirm skew presence
numeric_columns = ['carat', 'depth', 'table', 'price',
       'width', 'height', 'length']

# Path to save hist plots
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

