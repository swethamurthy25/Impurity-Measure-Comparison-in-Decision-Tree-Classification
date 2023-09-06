# Impurity-Measure-Comparison-in-Decision-Tree-Classification

## Related Course: Analyzing Big Data Concepts & Machine Learning Models

### Project Description

* In this project, we select a classification dataset containing class labels and structured as rows and columns.
* After cleaning the data by removing missing values with clear justifications, we employ Stratified K-Fold cross-validation to split the data into five equal folds.
* We train decision tree classifiers on four folds and evaluate their performance on the remaining fold, calculating classification accuracies.
* We create parameter grids to experiment with 'gini' and 'entropy' impurity measures, ensuring adequate maximum tree depth settings.
* Finally, we determine the best parameter configuration by averaging accuracies obtained across five runs and five folds for each impurity measure.

### Technical Details:

* Dataset used: post-operative.csv
* Programming Language: Python
* Libraries: Pandas, Numpy, sci-kit-learn

### Reading the dataset

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

header_list = ["patients_internal_temperature", "patients_surface_Temperature", "oxygen_saturation","last_measurement_of_blood_pressure","stability_of_patients_surface_temperature","stability_of_patients_core_temperature","stability_of_patients_blood_pressure","patients_perceived_comfort_at_discharge","decision"]
post_operative = pd.read_csv("C:/Users/sweth/Desktop/Spring 2022/IFT 598 - Analysing Big Data/Assignments/Extra Credit Lab/post-operative.csv", names=header_list)
post_operative
```

![image](https://github.com/swethamurthy25/Impurity-Measure-Comparison-in-Decision-Tree-Classification/assets/112581595/be14596a-1670-4636-b3fe-44e8b10ffdbe)

#### What is the number of rows & columns in this dataset? What does each row represent?

* We have 90 rows and 9 columns in the dataset.
* Each row represents a single patient's critical parameters (attributes) using which the output variable discharge decision is to be classified (S, I, or A).

### Data Transformation - Data Cleansing by removing the rows/columns that has missing values 

```python
MV = post_operative.isin(['?']).any(axis=1)
post_operative = post_operative[~(MV)]
print("Missing values are removed and dataset is cleaned now !!!")
post_operative
```
![image](https://github.com/swethamurthy25/Impurity-Measure-Comparison-in-Decision-Tree-Classification/assets/112581595/b8e8b9ed-4c99-43fe-aa34-0bbb15637ecc)

![image](https://github.com/swethamurthy25/Impurity-Measure-Comparison-in-Decision-Tree-Classification/assets/112581595/0b45048b-a072-4567-b72b-fc766fc65c7c)

* The dataset has 87 rows and 9 columns after missing values are removed.
* 3 rows with missing values (“?”) were removed when compared to the original data (90 rows and 9 columns).
* COMFORT (patient's perceived comfort at discharge, measured as an integer between 0 and 20) - The value of the comfort column has to be between 0 – 20 but it is missing,    so we have removed that.

#### New Dataframe created - Include Ratio columns to the new data frame, without transforming

```python
new_dataset = post_operative[['patients_perceived_comfort_at_discharge','decision']]
new_dataset
```

![image](https://github.com/swethamurthy25/Impurity-Measure-Comparison-in-Decision-Tree-Classification/assets/112581595/989ad595-e1b8-4fbe-87a1-8dc5a5beca96)

![image](https://github.com/swethamurthy25/Impurity-Measure-Comparison-in-Decision-Tree-Classification/assets/112581595/4c811ba2-b0ae-494d-b9d5-2ed43d166696)

#### Include ordinal columns to the new dataset, after transforming

```python
Internal_Temp = {'low':0, 'mid':1, 'high':2}
new_dataset['Internal_Temp']= post_operative['patients_internal_temperature'].map(Internal_Temp)

Surface_Temp = {'low':0, 'mid':1, 'high':2}
new_dataset['Surface_Temp']= post_operative['patients_surface_Temperature'].map(Surface_Temp)

oxygen_saturation = {'poor':0, 'fair':1, 'good':2, 'excellent':3}
new_dataset['Oxygen_Saturation']= post_operative['oxygen_saturation'].map(oxygen_saturation)

last_measurement_of_BP = {'low':0, 'mid':1, 'high':2}
new_dataset['last_measurement_of_BP']= post_operative['last_measurement_of_blood_pressure'].map(last_measurement_of_BP)

stability_of_patients_surface_Temp = {'unstable':0, 'mod-stable':1, 'stable':2}
new_dataset['stability_of_patients_surface_Temp']= post_operative['stability_of_patients_surface_temperature'].map(stability_of_patients_surface_Temp)

stability_of_patients_core_Temp = {'unstable':0, 'mod-stable':1, 'stable':2}
new_dataset['stability_of_patients_core_Temp']= post_operative['stability_of_patients_core_temperature'].map(stability_of_patients_core_Temp)

stability_of_patients_BP = {'unstable':0, 'mod-stable':1, 'stable':2}
new_dataset['stability_of_patients_BP']= post_operative['stability_of_patients_blood_pressure'].map(stability_of_patients_BP)
new_dataset
```

![image](https://github.com/swethamurthy25/Impurity-Measure-Comparison-in-Decision-Tree-Classification/assets/112581595/6305599d-92d4-40d2-b91d-d7492f17b5e9)

### Stratified Cross-Validation and Decision Tree Classification with Impurity Measures

```python
X = new_dataset.drop('decision',axis=1)
y = new_dataset.decision

Startified_K_Fold = StratifiedKFold(n_splits=5)
Startified_K_Fold.get_n_splits(X,y)
print(Startified_K_Fold)

gini_lst= []
entropy_lst = []
for train_index, test_index in  Startified_K_Fold.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    for cr in ["gini"]:
        dt = DecisionTreeClassifier(cr, max_depth = 90 )
        dt = dt.fit(X_train,y_train)
        y_pred = dt.predict(X_test)
        gini_acc = accuracy_score(y_test,y_pred)
        gini_lst.append(gini_acc)

    for cr in ["entropy"]:
        dt = DecisionTreeClassifier(cr, max_depth = 90 )
        dt = dt.fit(X_train,y_train)
        y_pred = dt.predict(X_test)
        entropy_acc = accuracy_score(y_test,y_pred)
        entropy_lst.append(entropy_acc)
        
print("Gini Accuracy \n")
print(gini_lst)
print("\n*****************************************************")
print("Entropy Accuracy \n")
print(entropy_lst)
print("\n*****************************************************")
average_gini = np.average(gini_lst)
print("The overall Gini Accuracy",average_gini)
average_entropy = np.average(entropy_lst)
print("The overall Entropy Accuracy",average_entropy)
```

### Output

StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
Gini Accuracy 

[0.5555555555555556, 0.6111111111111112, 0.7058823529411765, 0.5882352941176471, 0.5294117647058824]

*****************************************************
Entropy Accuracy 
[0.6666666666666666, 0.6111111111111112, 0.8823529411764706, 0.5294117647058824, 0.6470588235294118]

*****************************************************
The overall Gini Accuracy 0.5980392156862745
The overall Entropy Accuracy 0.6673202614379085


### Final Decision/Result:

Note that the overall Gini, entropy accuracy values are changing for each execution. Based on this execution, the best parameter is “Entropy”

![image](https://github.com/swethamurthy25/Impurity-Measure-Comparison-in-Decision-Tree-Classification/assets/112581595/8ad17249-d0cc-453c-a354-0f475a622c94)


______________________________________________________________________________________________________________









