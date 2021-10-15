# 1. Imports

import os
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import numpy as np
import logging
from sklearn import *

# 2. Load the df in Python (you can use pandas.read csv)
df = pd.read_csv('data/drug200.csv', dtype={'BP': 'category', 'Cholesterol': 'category', 'Drug': 'category'})
print(df.dtypes)

x = df.loc[:, ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = df['Drug']
classes = np.unique(y)

print(x, y)

'''
# 3. Plot the distribution of the instances in each class and store the graphic in a file called drug-distribution.pdf.
# You can use matplotlib.pyplot. This pre-analysis will allow you to determine if the classes are balanced,
# and which metric is more appropriate to use to evaluate the performance of your classifier.
'''
counts = {drug: np.count_nonzero(y == drug) for drug in classes}
print(counts)

plt.pie(counts.values(), labels=classes, autopct='%1.1f%%')
plt.title('Class Distribution')

plt.savefig('output/drug-distribution.pdf')
plt.show()

'''
4. Convert all ordinal and nominal features in numerical format. Make sure that your converted format
respects the ordering of ordinal features, and does not introduce any ordering for nominal features.
You may want to take a look at pandas.get dummies and pandas.Categorical to do this.
'''

x_numerical = pd.get_dummies(x, columns=['Sex'])
x_numerical['BP'] = x_numerical['BP'].cat.codes
x_numerical['Cholesterol'] = x_numerical['Cholesterol'].cat.codes
y_numerical = y.cat.codes

print(x_numerical)
print(y_numerical)

# 5. Split the df using train test split using the default parameter

x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(x_numerical, y_numerical)

print(x_train)
print(y_train)

'''
6. Run 6 different classifiers:
(a) NB: a Gaussian Naive Bayes Classifier (naive bayes.GaussianNB) with the default parameters.
(b) Base-DT: a Decision Tree (tree.DecisionTreeClassifier) with the default parameters.
(c) Top-DT: a better performing Decision Tree found using (GridSearchCV). The gridsearch will allow
you to find the best combination of hyper-parameters, as determined by the evaluation function that
you have determined in step (3) above. The hyper-parameters that you will experiment with are:
•criterion: gini or entropy
•max depth : 2 different values of your choice
•min samples split: 3 different values of your choice
(d) PER: a Perceptron (linear model.Perceptron), with default parameter values.
(e) Base-MLP: a Multi-Layered Perceptron (neural network.MLPClassifier) with 1 hidden layer of
100 neurons, sigmoid/logistic as activation function, stochastic gradient descent, and default values
for the rest of the parameters.
(f) Top-MLP: a better performing Multi-Layered Perceptron found using grid search. For this, you need
to experiment with the following parameter values:
•activation function: sigmoid, tanh, relu and identity
•2 network architectures of your choice: for eg 2 hidden layers with 30 + 50 nodes, 3 hidden layers
with 10 + 10 + 10
•solver: Adam and stochastic gradient descent

7. For each of the 6 classifier above, append the following information in a file called drugs-performance.txt:
(to make it easier for the TAs, make sure that your output for each sub-question below is clearly marked
in your output file, using the headings (a), (b) . .. )
(a) a clear separator (a sequence of hyphens or stars) and a string clearly describing the model (e.g. the
model name + hyper-parameter values that you changed). In the case of Top-DT and Top-MLP,
display the best hyperparameters found by the gridsearch.
(b) the confusion matrix
(c) the precision, recall, and F1-measure for each class
(d) the accuracy, macro-average F1 and weighted-average F1 of the model

8. Redo steps 6, 10 times for each model and append the average accuracy, average macro-average F1, average 
weighted-average F1 as well as the standard deviation for the accuracy, the standard deviation of the macro-average F1, 
and the standard deviation of the weighted-average F1 at the end of the file drugs-performance.txt. 
Does the same model give you the same performance every time? Explain in a plain text file called drugs-discussion.txt. 
1 or 2 paragraph discussion is expected.
'''
