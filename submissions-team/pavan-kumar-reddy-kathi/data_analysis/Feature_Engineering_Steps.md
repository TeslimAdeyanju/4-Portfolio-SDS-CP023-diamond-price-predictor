1. carat-weight, cut-shine, clarity-internal purity, depth - diamond depth percentage, table-flat facet on diamond surface/large flat area visible from above.
2. Identify correlations - Remove redundant columns here length/width/height. cut seems weak/moderate correlation (negative,-0.47) with table. clarity got weak correlation (negative,-0.37) with Width,height,length,carat
3. Correlations affect linear/logistic regression,SVM,PCA,Tree Based models/ensemble methods,KNN,clustering etc. PCA,regularization methods helps to address it
4. Try to see if you can create new features (like ratios etc.) from existing features after removing correlated features. It probably requires domain knowledge, at  this moment I don't see any scope for creating new feature. Once you create new features you need to reapply preprocessing steps as you did for original features.You should introduce new features only after completing preprocessing steps on existing dataset.
5. Skew applies to numerical variables both feature and target. linear/logistic regression/KNN/SVM/Neural networks/PCA are sensitive to skew. Tree based models are less sensitive to skew.Outliers contribute to skew.
6. For Categorical variables it is good idea to look at frequency distribution and Imbalance using bar plot, rather than skew calculations
7. use power transformation ('yeo-johnson') for Gaussian distribution, quantile transformer for uniform and normal distribution
8. apply techniques like IQR when you handle outliers, even transformations doesn't remove outliers
9. The maximum length of diamond that is found as of today is 10.1cm and our data set contains max length as 5.8
10. Can we use imputation techniques for observations that are having zero value as their features
11. for this analysis we are removing records that are having zero as feature value. we got 8 records with width as zero
20 records with height as zero and 7 records with length as zero, as their % is small compared to total number of records, we decided to remove them.
12. For outliers, we should have business knowledge how we should handle them, we can either remove them, cap (1.5, 3) them or leave them as it is (ex: fraudulent transactions) or apply transformations. we do have outliers for numeric columns in our dataset.
     | Attribute | Number of Outliers | Percentage| Skew |
     | ---------- | ----------------- | ----------| ----- |
     | price      | 3532 | 6.55 | 1.62 |
     | carat      | 1883 | 3.49 | 1.12 |
     | depth      | 2543 | 4.72 | -0.08 |
     | table      | 604 | 1.12 | 0.8 |
     | width      | 24 | 0.04 | 0.4 |
     | height     | 29 | 0.05 | 1.58 |
     | length      | 22 | 0.04 | 2.46 |
13. impacts of PCA
14. PCA is affected by outliers and also you should standardize before you apply PCA
15. How do i decide whether I go for PCA or Kernel PCA
16. Skew Limit -0.5 to 0.5 is generally acceptable 
17. based on skew value decide whether it is normal distribution or not, based on that we need to choose the way to identify outliers
18. Look at algorhythms that are impacted by outliers, try to apply trasnformations to see if then can reduce outliers.
19. Below are the metrics after just removing rows with zero values and without specific hyperparameter tuning
     | Model | R2 Train | R2 Test| RMSE Train | RMSE Test
     | ---------- | ----------------- | ----------| ----- |
     | price      | 3532 | 6.55 | 1.62 |
     | carat      | 1883 | 3.49 | 1.12 |
     | depth      | 2543 | 4.72 | -0.08 |
     | table      | 604 | 1.12 | 0.8 |
     | width      | 24 | 0.04 | 0.4 |
     | height     | 29 | 0.05 | 1.58 |
     | length      | 22 | 0.04 | 2.46 |
20. [{'learning_rate' : [0.07, 0.08], 'iterations':[700,750], 'depth':[9,10]}]
params_grid = [{'rsm' : [0.6, 0.8, 1]}]

'depth':[9,10]}    




