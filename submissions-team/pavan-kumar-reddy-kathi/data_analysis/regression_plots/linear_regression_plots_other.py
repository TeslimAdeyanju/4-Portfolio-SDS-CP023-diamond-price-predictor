import os.path

from dpputility.data_set_module import get_data_frame
from dpputility.plot_module import create_reg_plot
from dpputility.config_module import read_config_setting

# Load dataset
dataset = get_data_frame()

# Path to Save Regression Plots
root_path = os.path.abspath('../../')
folder_path = read_config_setting('regression_plots_folder')
folder_to_save = os.path.join(root_path,folder_path)

# create_reg_plot(dataset, 'carat', 'depth', folder_to_save)
# create_reg_plot(dataset, 'depth', 'carat', folder_to_save)
# create_reg_plot(dataset, 'carat', 'table', folder_to_save)
# create_reg_plot(dataset, 'table', 'carat', folder_to_save)
# create_reg_plot(dataset, 'width', 'carat', folder_to_save)
# create_reg_plot(dataset, 'length', 'carat', folder_to_save)
# create_reg_plot(dataset, 'height', 'carat', folder_to_save)
create_reg_plot(dataset, 'cut_encoded', 'carat', folder_to_save)