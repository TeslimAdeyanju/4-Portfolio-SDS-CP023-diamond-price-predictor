import os

import matplotlib.pyplot as plt
import seaborn as sns
from dpputility.data_set_module import get_data_frame
from dpputility.config_module import read_config_setting
import pandas as pd

pd.options.display.max_columns = None
pd.set_option('display.max_rows',None)

dataset = get_data_frame()
# select only numeric columns
filter_columns = ['cut_encoded', 'color_encoded', 'clarity_encoded', 'carat', 'depth', 'table', 'price',
                       'width', 'height', 'length']
filter_dataset = dataset[filter_columns]

# Capture correlations between columns
# Pearson Correlation (default) --Linear correlations
# pearson_corr = filter_dataset.corr()
# print(pearson_corr)

# Spearman's Rank Correlation - nonlinear relationships, ordinal data non normally distributed continuous data
# spearman_rank_correlation = filter_dataset.corr(method='spearman')
# print(spearman_rank_correlation)

# Kendall's Tau - Small sample sizes , ordinal data,  data with Ties
# kendall_correlation = filter_dataset.corr(method='kendall')
# print(kendall_correlation)

correlation_method = ['pearson', 'spearman', 'kendall']
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






