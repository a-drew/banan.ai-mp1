# 1. Imports

import os
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import numpy as np
import logging
from sklearn import *
from time import time

# controls max iterations of the MLP Classifier. Set higher for better results. Convergence warnings were silenced
max_iter = 4000

# 2. Load the df in Python (you can use pandas.read csv)
t = time()
df = pd.read_csv('data/drug200.csv', dtype={'BP': 'category', 'Cholesterol': 'category', 'Drug': 'category'})
print(df.dtypes)

x = df.loc[:, ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = df['Drug']
classes = np.unique(y)

print(x, y)

'''
3. Plot the distribution of the instances in each class and store the graphic in a file called drug-distribution.pdf.
You can use matplotlib.pyplot. This pre-analysis will allow you to determine if the classes are balanced,
and which metric is more appropriate to use to evaluate the performance of your classifier.
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

print(x_numerical, y_numerical)

# 5. Split the df using train test split using the default parameter

x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(x_numerical, y_numerical)
print(x_train, y_train)

# 6. Run 6 different classifiers:
models = []
# (a) NB: a Gaussian Naive Bayes Classifier (naive bayes.GaussianNB) with the default parameters.
print("\nPerforming Gaussian NB training...")
t0 = time()
gnb = skl.naive_bayes.GaussianNB()
models.append(gnb)
gnb.fit(x_train, y_train)
print("done in %0.3fs" % (time() - t0))
print()

# (b) Base-DT: a Decision Tree (tree.DecisionTreeClassifier) with the default parameters.
print("Performing Base Decision Tree training...")
t0 = time()
bdt = skl.tree.DecisionTreeClassifier()
models.append(bdt)
bdt.fit(x_train, y_train)
print("done in %0.3fs" % (time() - t0))
print()

# (c) Top-DT: a better performing Decision Tree found using (GridSearchCV). The gridsearch will allow
# you to find the best combination of hyper-parameters, as determined by the evaluation function that
# you have determined in step (3) above. The hyper-parameters that you will experiment with are:
#    •criterion: gini or entropy
#    •max depth : 2 different values of your choice
#    •min samples split: 3 different values of your choice
print("Performing Top-DT training w/ grid search...")
parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [4, 5],
    'min_samples_split': [2, 5, 10]
}
print("parameters:", parameters)
t0 = time()
tdt = skl.model_selection.GridSearchCV(skl.tree.DecisionTreeClassifier(), parameters)
models.append(tdt)
tdt.fit(x_train, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best score: %0.3f" % tdt.best_score_)
print("Best parameters: ", tdt.best_params_)
print()

# (d) PER: a Perceptron (linear model.Perceptron), with default parameter values.
print("Performing Perceptron training...")
t0 = time()
per = skl.linear_model.Perceptron()
models.append(per)
per.fit(x_train, y_train)
print("done in %0.3fs" % (time() - t0))
print()

# (e) Base-MLP: a Multi-Layered Perceptron (neural network.MLPClassifier) with 1 hidden layer of
# 100 neurons, sigmoid/logistic as activation function, stochastic gradient descent, and default values
# for the rest of the parameters.
from sklearn.utils._testing import ignore_warnings

print("Performing Base Multi-Layered Perceptron training...")
t0 = time()
bmlp = skl.neural_network.MLPClassifier(hidden_layer_sizes=100, activation='logistic', solver='sgd', max_iter=max_iter)
models.append(bmlp)
with ignore_warnings(category=skl.exceptions.ConvergenceWarning):
    bmlp.fit(x_train, y_train)
print("done in %0.3fs" % (time() - t0))

# (f) Top-MLP: a better performing Multi-Layered Perceptron found using grid search. For this, you need
# to experiment with the following parameter values:
#  •activation function: sigmoid, tanh, relu and identity
#  •2 network architectures of your choice: for eg 2 hidden layers with 30 + 50 nodes, 3 hidden layers with 10 + 10 + 10
#  •solver: Adam and stochastic gradient descent
print("Performing Top-MLP training w/ grid search...")
parameters = {
    'hidden_layer_sizes': [(10, 20), (15, 15, 15)],
    'activation': ['logistic', 'tanh', 'relu', 'identity'],
    'solver': ['sgd', 'adam'],
}
print("parameters:", parameters)
t0 = time()
tmlp = skl.model_selection.GridSearchCV(skl.neural_network.MLPClassifier(max_iter=max_iter), parameters)
models.append(tmlp)
with ignore_warnings(category=skl.exceptions.ConvergenceWarning):
    tmlp.fit(x_train, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best score: %0.3f" % tmlp.best_score_)
print("Best parameters:", tdt.best_params_)
print()

# TODO 7: review warnings and 0 precision/f1 for MLPs
'''
7. For each of the 6 classifier above, append the following information in a file called drugs-performance.txt:
(to make it easier for the TAs, make sure that your output for each sub-question below is clearly marked
in your output file, using the headings (a), (b) . .. )
(a) a clear separator (a sequence of hyphens or stars) and a string clearly describing the model (e.g. the
model name + hyper-parameter values that you changed). In the case of Top-DT and Top-MLP,
display the best hyperparameters found by the gridsearch.
(b) the confusion matrix
(c) the precision, recall, and F1-measure for each class
(d) the accuracy, macro-average F1 and weighted-average F1 of the model
'''

if os.path.exists('output/drug-performance.txt'):
    os.remove('output/drug-performance.txt')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('output/drug-performance.txt', 'w'))
logger.info("Task 7:\n")


def performance_report(model):
    logger.info('================================================================================')
    if isinstance(model, skl.model_selection.GridSearchCV):
        logger.info("Top " + type(model.estimator).__name__)
        #logger.info("Best score: %0.3f" % model.best_score_)
        logger.info("Best parameters: ")
        logger.info(model.best_params_)
    else:
        logger.info(type(model).__name__)

    y_pred = model.predict(x_test)

    logger.info('\n(b) confusion_matrix:')
    logger.info(skl.metrics.confusion_matrix(y_test, y_pred))
    logger.info("\n(c/d) classification_report: ")
    logger.info(skl.metrics.classification_report(y_test, y_pred, target_names=classes))
    logger.info("\n(d) accuracy_score: ")
    logger.info(str(100 * skl.metrics.accuracy_score(y_test, y_pred)) + '%')
    logger.info("\n(d) f1_score (macro avg): ")
    logger.info(str(100 * skl.metrics.f1_score(y_test, y_pred, average='macro')) + '%')
    logger.info("\n(d) f1_score (weighted avg): ")
    logger.info(str(100 * skl.metrics.f1_score(y_test, y_pred, average='weighted')) + '%')
    logger.info('================================================================================')


for model in models:
    performance_report(model)

'''
8. Redo steps 6, 10 times for each model and append the average accuracy, average macro-average F1, average 
weighted-average F1 as well as the standard deviation for the accuracy, the standard deviation of the macro-average F1, 
and the standard deviation of the weighted-average F1 at the end of the file drugs-performance.txt. 
Does the same model give you the same performance every time? Explain in a plain text file called drugs-discussion.txt. 
1 or 2 paragraph discussion is expected.
'''
logger.info("\nTask 8:\n")


def train_predict_collect_metrics(model):
    with ignore_warnings(category=skl.exceptions.ConvergenceWarning):
        model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = skl.metrics.accuracy_score(y_test, y_pred)
    maf1 = skl.metrics.f1_score(y_test, y_pred, average='macro')
    waf1 = skl.metrics.f1_score(y_test, y_pred, average='weighted')
    return acc, maf1, waf1


def general_metrics_report(metrics, desc=None):
    logger.info('================================================================================')
    if desc is not None:
        logger.info(desc)
    average = [sum(x) / len(x) for x in zip(*metrics)]
    logger.info("Average Accuracy: " + str(100 * average[0]) + "%")
    logger.info("Average macro-average F1: " + str(100 * average[1]) + "%")
    logger.info("Average weighted-average F1: " + str(100 * average[2]) + "%")
    std = [np.std(s) for s in zip(*metrics)]
    logger.info("Accuracy Standard Deviation: " + str(std[0]))
    logger.info("Macro-Average F1 Standard Deviation: " + str(std[1]))
    logger.info("Weighted-Average F1 Standard Deviation: " + str(std[2]))
    logger.info('================================================================================')


# (a) NB: a Gaussian Naive Bayes Classifier (naive bayes.GaussianNB) with the default parameters.
print("\nCalculating Metrics for Gaussian NB...")
t0 = time()
gnb_metrics = []
for i in range(10):
    print('run #' + str(i + 1))
    gnb = skl.naive_bayes.GaussianNB()
    gnb_metrics.append(train_predict_collect_metrics(gnb))
    print(gnb_metrics[-1])
print("done in %0.3fs" % (time() - t0))
print()
general_metrics_report(gnb_metrics, desc="Metrics for 10 runs of Gaussian Naive Bayes: ")

# (b) Base-DT: a Decision Tree (tree.DecisionTreeClassifier) with the default parameters.
print("Calculating Metrics for Base Decision Tree...")
t0 = time()
bdt_metrics = []
for i in range(10):
    print('run #' + str(i + 1))
    bdt = skl.tree.DecisionTreeClassifier()
    bdt_metrics.append(train_predict_collect_metrics(bdt))
    print(bdt_metrics[-1])
print("done in %0.3fs" % (time() - t0))
print()
general_metrics_report(bdt_metrics, desc="Metrics for 10 runs of Base Decision Tree: ")

# (c) Top-DT: a better performing Decision Tree found using (GridSearchCV)
print("Calculating Metrics for Top-DT w/ grid search...")
parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [4, 5],
    'min_samples_split': [2, 5, 10]
}
t0 = time()
tdt_metrics = []
for i in range(10):
    print('run #' + str(i + 1))
    tdt = skl.model_selection.GridSearchCV(skl.tree.DecisionTreeClassifier(), parameters)
    tdt_metrics.append(train_predict_collect_metrics(tdt))
    print(tdt_metrics[-1])
print("done in %0.3fs" % (time() - t0))
print()
general_metrics_report(tdt_metrics, desc="Metrics for 10 runs of Top Decision Tree: ")

# (d) PER: a Perceptron (linear model.Perceptron), with default parameter values.
print("Calculating Metrics for Perceptron...")
t0 = time()
per_metrics = []
for i in range(10):
    print('run #' + str(i + 1))
    per = skl.linear_model.Perceptron()
    per_metrics.append(train_predict_collect_metrics(per))
    print(per_metrics[-1])
print("done in %0.3fs" % (time() - t0))
print()
general_metrics_report(per_metrics, desc="Metrics for 10 runs of Perceptron: ")

# (e) Base-MLP: a Multi-Layered Perceptron (neural network.MLPClassifier)
print("Calculating Metrics for Base Multi-Layered Perceptron...")
t0 = time()
bmlp_metrics = []
for i in range(10):
    print('run #' + str(i + 1))
    bmlp = skl.neural_network.MLPClassifier(hidden_layer_sizes=100, activation='logistic', solver='sgd', max_iter=max_iter)
    bmlp_metrics.append(train_predict_collect_metrics(bmlp))
    print(bmlp_metrics[-1])
print("done in %0.3fs" % (time() - t0))
print()
general_metrics_report(bmlp_metrics, desc="Metrics for 10 runs of Base Multi-Layered Perceptron: ")

# (f) Top-MLP: a better performing Multi-Layered Perceptron found using grid search.
print("Calculating Metrics for Top Multi-Layered Perceptron...")
parameters = {
    'hidden_layer_sizes': [(10, 20), (15, 15, 15)],
    'activation': ['logistic', 'tanh', 'relu', 'identity'],
    'solver': ['sgd', 'adam'],
}
t0 = time()
tmlp_metrics = []
for i in range(10):
    print('run #' + str(i + 1))
    tmlp = skl.model_selection.GridSearchCV(skl.neural_network.MLPClassifier(max_iter=max_iter), parameters)
    tmlp_metrics.append(train_predict_collect_metrics(tmlp))
    print(tmlp_metrics[-1])
print("done in %0.3fs" % (time() - t0))
print()
general_metrics_report(tmlp_metrics, desc="Metrics for 10 runs of Top Multi-Layered Perceptron: ")

print("\n\nTotal runtime: %0.3fs" % (time() - t))
