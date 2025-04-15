import os

import matplotlib.pyplot as plt
import seaborn as sns
from dpputility.data_set_module import get_data_frame
from dpputility.config_module import read_config_setting
import pandas as pd

# Reference ranges for Correlation
# 0.9-1 very strong
# 07-.9 strong
# 0.5-0.7 moderate
# 0.3-0.5 weak
# 0-0.3 very weak or no linear Correlation


pd.options.display.max_columns = None

dataset = get_data_frame()

# select only numeric columns
filter_columns = ['cut_encoded', 'color_encoded', 'clarity_encoded', 'carat', 'depth', 'table', 'price',
                       'width', 'height', 'length']
filter_dataset = dataset[filter_columns]

correlation_method = ['pearson', 'spearman', 'kendall']

# Path to Save heatmaps
root_path = os.path.abspath('../../')
folder_path = os.path.join(root_path,read_config_setting('correlation_plots_folder'))

for method in correlation_method:
    correlation_matrix = filter_dataset.corr(method=method)
    plt.figure(figsize=(12,8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(f'Feature Correlations-{method}')
    plt.tight_layout()
    path_to_save = os.path.join(folder_path, f'Feature Correlations-{method}.png')
    plt.savefig(path_to_save)






