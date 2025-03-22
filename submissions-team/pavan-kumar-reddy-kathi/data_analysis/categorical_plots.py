import matplotlib.pyplot as plt

from dpputility.data_set_module import get_data_frame
import seaborn as sns

dataset = get_data_frame()
# sns.countplot(data=dataset, x='clarity')
# plt.title('cut barplot')
# plt.show()

sns.boxplot(data=dataset, x='cut', y='price')
# sns.violinplot(data=dataset, x='cut', y='price')
plt.show()
