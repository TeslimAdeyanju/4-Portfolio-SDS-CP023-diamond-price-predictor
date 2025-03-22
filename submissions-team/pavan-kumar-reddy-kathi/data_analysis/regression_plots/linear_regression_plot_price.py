from dpputility import (data_set_module as dsm,
                        config_module as cm,
                        plot_module as pm)
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
dataset = dsm.get_data_frame()

columns_to_exclude = ['cut', 'color', 'clarity', 'price']
folder_path = cm.read_config_setting('regression_plots_folder')
folder_to_save = os.path.join(root_path, folder_path)

for column in dataset.columns:
    if column not in columns_to_exclude:
        pm.create_reg_plot(dataset,column,'price', folder_to_save)



