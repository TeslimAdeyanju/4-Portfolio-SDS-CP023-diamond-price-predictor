from dpputility import data_set_module as dsm, outlier_module as om
import seaborn as sns
import matplotlib.pyplot as plt

dataset = dsm.get_data_frame()

om.analyze_and_remove_outliers_iqr(dataset,'price')
# sns.pairplot(dataset)
# plt.show()


