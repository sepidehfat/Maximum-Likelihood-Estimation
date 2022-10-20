import pandas as pd
import numpy as np
# from sklearn.metrics import confusion_matrix

data = pd.read_csv('Boston.csv')
medv_mean = data["medv"].mean()
data['medv'] = np.where(data['medv'] > medv_mean, 1, 0)
test_ratio = 0.6

train_size = int(len(data)*(1-test_ratio))
data_train = data.loc[1:train_size, :]

x_train = data.iloc[0:train_size, 1:14]
y_train = data.iloc[0:train_size, 14]

x_test = data.iloc[train_size:, 1:14]
y_test = data.iloc[train_size:, 14]

feature_size = len(x_train.iloc[1])
#%% prior P(y)
prior = []
prior = (x_train.groupby(y_train).apply(lambda x: len(x))/train_size).to_numpy()


#%%
def gaussian_probability(x_row,class_type, train_mean, train_var):
    a = np.exp((-1 / 2) * ((x_row - train_mean[class_type]) ** 2) / (2 * train_var[class_type]))
    b = np.sqrt(2 * np.pi * train_var[class_type])
    return a / b

#%%
train_mean = x_train.groupby(y_train).apply(np.mean).to_numpy()
train_var = x_train.groupby(y_train).apply(np.var).to_numpy()

np.seterr(divide = 'ignore')
predictions = []
for row in x_test.to_numpy():
    posteriors = {}
    for class_type in range(2):
        posterior = np.sum(np.log(gaussian_probability(row, class_type, train_mean, train_var))) + np.log(prior[class_type])
        posteriors[class_type] = posterior
    if posteriors[0] > posteriors[1]:
        predictions.append(0)
    else:
        predictions.append(1)
#%% accuracy
accuracy = np.sum(y_test == predictions) / len(y_test)

data_confusion = pd.crosstab(y_test.reset_index(drop = True), pd.Series(predictions))

