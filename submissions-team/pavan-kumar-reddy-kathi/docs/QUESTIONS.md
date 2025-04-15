1. x (Premium)', 'z (Very Good)', 'y (Good)' are they just height width and length, or they have any other significance with respect to Premium, very good and good.
2. Steps that need to be undertaken when dealing with numerical features for better model results for regression
3. Tell me more about exploratory data analysis
4. What is skewness, does it create problems and how to identify and overcome it
5. How do  I arrange cut color clarity when I want to do label encoding with ordering
6. What I should do, if some of the independent variables showing non-linear and other showing linear relation with dependent variable
7. What I should do, if some of the independent variables not following some of the assumptions of Linear Regression
8. Do we need to check for correlation only for linear regression algorithm, or we need to do it for any machine learning algorithm
9. Multi collinearity does only refer correlation between features, or it also talks about correlation between feature and target?
10. "Decision trees and ensemble methods like random forests and gradient boosting are less sensitive to feature correlation." does it correlation among features or correlation between feature and target variable?
11. How do I find correlation between target and feature and how to identify whether they are strongly correlated or weekly correlated?
12. Does correlation between features impact any other algorithms other than linear regression?
13. Does skew apply to ordinal categorical variables that are label encoded
14. Which machine learning models affected by skew
15. Min Max scalar Vs Standard Scaler
16. What is Power Transformer and why do we need it? to address skewness or heteroscedasticity 
17. How QuantileTransformer (handles outliers) is different to Power Transformer
18. When should we apply normalization and standardization, I mean use cases
19. When do we label encode categorical variables? is it before test/train split or after test/train split?
20. How to ensure categorical variables are split across train and test data proportionately
21. Are there any ratios we can come up with for diamond price prediction that improves model performance? available features are 'carat', 'depth', 'table','width', 'height', 'length'
22. It seems carat is highly correlated with height, width and length, in that case can I just remove feature height, width and length and just live with carat?
23. How do I know whether I need to go for transformation or remove outliers?
24. Will there be any harm with predictions, if i by default go with data transformations
25. Removing of outliers Vs capping with IQR Vs Winsorization
26. Individual features with invalid values (remove Vs Imputation)
27. When I should I think about applying mathematical transformations like log, sqrt, power etc
28. Can I apply both scaling and normalization to same column? if so in which order do I need to apply?
29. Can I apply both outliers handling by IQR and then apply transformations like log,sqrt,power transformations?
30. What are machine learning models on which we can apply PCA
31. How can PCA reduce dimensionality?
32. Explain Eigen values and Eigen vectors in context of correlation between features
33. How to choose between PCA linear Vs Kernel
34. How do I know whether a given feature is really contributing to Model prediction or not
35. How do I ensure my regression model is not overfitted
36. For regression what metrics helps to find whether my model is overfitted?
37. Outliers applies to only independent variable or does it also applies to dependent variable as well
38. What is threshold % of outlier data we can remove from original dataset safely
39. How to identify outliers? If variable follows normal distribution, any data beyond mean +/- 3*standard deviation is outlier. For normal distribution we can also use 5th and 95th quantile to identify outliers. If it is not normally distributed/skewed we can use IQR proximity rule 75th quantile+1.5/3IQR or 25th quantile-1.5/3IQR
40. Do we need to remove outliers before test/train split or after, if after do we just need to remove outliers from train data alone? ok you need to calculate outlier lower/upper limits based on training data alone and use that data to remove/cap outliers from both train and test data
41. How do I mathematically know if a variable follows normal distribution
42. Difference between GPU and CPU learning
43. Best machine learning project structure along with pickle file deployment using fast api and UI using streamlit
44. Can we draw regression plot of continuous variable against categorical variable
45. How do I know whether a feature really got an impact on dependent/target variable?
