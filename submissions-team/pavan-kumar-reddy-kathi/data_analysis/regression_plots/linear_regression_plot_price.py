from dpputility import (data_set_module as dsm,
                        config_module as cm,
                        plot_module as pm)
import os

# Path to Save Regression Plots for Price against independent variables
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
folder_path = cm.read_config_setting('regression_plots_folder')
folder_to_save = os.path.join(root_path, folder_path)

dataset = dsm.get_data_frame()
columns_to_exclude = ['cut', 'color', 'clarity', 'price']

for column in dataset.columns:
    if column not in columns_to_exclude:
        pm.create_reg_plot(dataset,column,'price', folder_to_save)



