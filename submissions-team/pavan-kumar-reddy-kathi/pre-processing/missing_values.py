from dpputility.data_set_module import get_data_frame


dataset = get_data_frame()

print(dataset.isna().sum())