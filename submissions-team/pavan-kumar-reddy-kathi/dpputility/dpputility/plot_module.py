import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def create_reg_plot(dataset:pd.DataFrame, x_column:str, y_column:str,
                   folder_to_save:str):
    """
    Creates regression plots
    :param dataset: DataFrame with input data
    :param x_column: Column for X axis
    :param y_column: Column for Y axis
    :param folder_to_save: Location to which Regression plot is saved.
    :return: void
    """
    plt.figure(figsize=(8, 6))

    sns.regplot(data=dataset, x=x_column, y=y_column, ci=95,
                scatter_kws={'s': 50, 'alpha': 0.5}, line_kws={'color': 'red', 'linewidth': 2})

    plt.xlim(dataset[x_column].min(), dataset[x_column].max())
    plt.ylim(dataset[y_column].min(), dataset[y_column].max())
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    path_to_save = os.path.join(folder_to_save, f'{x_column}_vs_{y_column}.png')
    plt.savefig(path_to_save)