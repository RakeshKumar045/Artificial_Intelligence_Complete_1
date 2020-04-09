'''
Author : Imad Dabbura
Script to train ExtraTreesClassifier to predict TIA.
'''

import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.utils import resample


def load_data(fname):
    df = pd.read_csv(fname)
    return df


def split_data(df):
    X = df.drop(columns=['PatientID', 'TIA'])
    y = df['TIA']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.15,
                                                        stratify=y,
                                                        random_state=123)
    return (X_train, y_train,
            X_test, y_test)


def downsample_data(X_train, y_train):
    X_train_d, y_train_d = resample(X_train[y_train == 0],
                                    y_train[y_train == 0],
                                    replace=True,
                                    n_samples=X_train[y_train == 1].shape[0],
                                    random_state=1)
    X_train_d = np.concatenate((X_train[y_train == 1], X_train_d))
    y_train_d = np.concatenate((y_train[y_train == 1], y_train_d))
    return X_train_d, y_train_d


def preprocess_data():
    cont_feat_idxs = load('cont-feat-idxs')
    cat_feat_idxs = load('cat-feat-idxs')
    categories = load('categories')
    col_transf = ColumnTransformer(
        [('rob_scaler', RobustScaler(), cont_feat_idxs),
         ('ohe', OneHotEncoder(
             handle_unknown='ignore', categories=categories), cat_feat_idxs
          )], remainder='passthrough'
    )
    return col_transf


def train(X_train, y_train, col_transformer):
    xtrees_clf = ExtraTreesClassifier(n_estimators=10,
                                      criterion='entropy',
                                      max_features=0.5,
                                      min_impurity_decrease=0.00425,
                                      class_weight='balanced',
                                      n_jobs=-1)

    pip_extra_trees_clf = Pipeline(steps=[
        ('transformers', col_transformer),
        ('extra_trees_clf', xtrees_clf)
    ])
    pip_extra_trees_clf.fit(X_train, y_train)
    return pip_extra_trees_clf


def main():
    parser = ArgumentParser(
        description='Train ExtraTreesClassifier on TIA dataset_D')
    parser.add_argument('--fname_path', type=str, metavar='',
                        help='Absolute filename path')
    parser.add_argument('--model_path', type=str, default=os.getcwd(), metavar='',
                        help='Directory of where to save the model.')
    args = parser.parse_args()

    if not args.fname_path:
        raise ('Please provide the path to the data file.')

    df = load_data(args.fname)
    X_train, y_train, X_test, y_test = split_data(df)
    X_train_d, y_train_d = downsample_data(X_train, y_train)
    col_transformer = preprocess_data()
    model = train(X_train_d, y_train_d, col_transformer)
    dump(model, f'{args.model_path}/xtree_clf_tia')


if __name__ == "__main__":
    main()
