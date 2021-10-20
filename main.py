import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def read_data(filename): #C:\Users\Anatoly\PycharmProjects\AI_lab1
    return pd.read_csv(os.path.join(filename))

def load_dataset(label_dict):
    train_X = read_data('train.csv').values[:,:-2]
    train_y = read_data('train.csv')['Activity']
    train_y = train_y.map(label_dict).values
    test_X = read_data('test.csv').values[:,:-2]
    test_y = read_data('test.csv')
    test_y = test_y['Activity'].map(label_dict).values
    return(train_X, train_y, test_X, test_y)


label_dict = {'WALKING':0, 'WALKING_UPSTAIRS':1, 'WALKING_DOWNSTAIRS':2, 'SITTING':3, 'STANDING':4, 'LAYING':5}


train_X, train_Y, test_X, test_Y = load_dataset(label_dict)

SVC_model = svm.SVC()
# В KNN-модели нужно указать параметр n_neighbors
# Это число точек, на которое будет смотреть
# классификатор, чтобы определить, к какому классу принадлежит новая точка
KNN_model = KNeighborsClassifier(n_neighbors=6)
lda_model = LatentDirichletAllocation(
   max_iter=10, learning_method='online')

SVC_model.fit(train_X, train_Y)
KNN_model.fit(train_X, train_Y)

pipe = make_pipeline(StandardScaler(),LogisticRegression())
pipe.fit(train_X, train_Y)
SVC_prediction = SVC_model.predict(test_X)
KNN_prediction = KNN_model.predict(test_X)
pipe_prediction = pipe.predict(test_X)

print("SVC model accuracy = " + str(accuracy_score(SVC_prediction, test_Y)))
print("KNN model accuracy = " + str(accuracy_score(KNN_prediction, test_Y)))
print("pipe model accuracy = " + str(accuracy_score(pipe_prediction, test_Y)))

from sklearn.metrics import classification_report
target_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']

print(classification_report(test_Y, SVC_prediction, target_names=target_names))
print(classification_report(test_Y, KNN_prediction, target_names=target_names))
print(classification_report(test_Y, pipe_prediction, target_names=target_names))
