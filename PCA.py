# -*- coding: utf-8 -*-
"""
Created on Sun May  9 09:09:58 2021

@author: sthan
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('diabetes.csv')

X = df.drop('diabetes', 1)
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y )


pca = PCA(n_components = 3)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)


print(pca.explained_variance_ratio_)

lr = LogisticRegression()
lr.fit(X_train_pca, y_train)
lr.score(X_test_pca, y_test)


svm = SVC()
svm.fit(X_train_pca, y_train)
svm.score(X_test_pca, y_test)


