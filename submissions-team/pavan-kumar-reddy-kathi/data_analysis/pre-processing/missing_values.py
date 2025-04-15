from dpputility.data_set_module import get_data_frame


dataset = get_data_frame()
# Look for missing data
print(dataset.isna().sum())