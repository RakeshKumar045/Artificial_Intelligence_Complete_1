'''
Helper functions used in solving ML coding assignment from pcci.
'''

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score


def plot_class_dist(array, x_labels, figsize=(8, 6)):
    '''Plot class distribution in classification tasks.'''
    plt.figure(figsize=figsize)
    sns.countplot(array)
    plt.xticks(range(len(x_labels)), x_labels, fontsize=18)
    plt.ylabel('Count', fontdict={'fontsize': 18})
    plt.title('Classes Distribution', y=1, fontdict={'fontsize': 20})


def get_ecdf(a):
    '''Get empirical CDF.'''
    x = np.sort(a)
    y = np.arange(1, len(a) + 1) / len(a)
    return x, y


def plot_conf_matrix_and_roc(estimator, X, y, figure_size=(16, 6)):
    '''
    Plot both confusion matrix and ROC curce on the same figure.

    Parameters:
    -----------
    estimator : sklearn.estimator
        model to use for predicting class probabilities.
    X : array_like
        data to predict class probabilities.
    y : array_like
        true label vector.
    figure_size : tuple (optional)
        size of the figure.

    Returns:
    --------
    plot : matplotlib.pyplot
        plot confusion matrix and ROC curve.
    '''
    # Compute tpr, fpr, auc and confusion matrix
    fpr, tpr, thresholds = roc_curve(y, estimator.predict_proba(X)[:, 1])
    auc = roc_auc_score(y, estimator.predict_proba(X)[:, 1])
    conf_mat_rf = confusion_matrix(y, estimator.predict(X))

    # Define figure size and figure ratios
    plt.figure(figsize=figure_size)
    gs = GridSpec(1, 2, width_ratios=(1, 2))

    # Plot confusion matrix
    ax0 = plt.subplot(gs[0])
    ax0.matshow(conf_mat_rf, cmap=plt.cm.Reds, alpha=0.2)

    for i in range(2):
        for j in range(2):
            ax0.text(x=j, y=i, s=conf_mat_rf[i, j], ha='center', va='center')
    plt.title('Confusion matrix', y=1.1, fontdict={'fontsize': 20})
    plt.xlabel('Predicted', fontdict={'fontsize': 14})
    plt.ylabel('Actual', fontdict={'fontsize': 14})

    # Plot ROC curce
    ax1 = plt.subplot(gs[1])
    ax1.plot(fpr, tpr, label='auc = {:.3f}'.format(auc))
    plt.title('ROC curve', y=1, fontdict={'fontsize': 20})
    ax1.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False positive rate', fontdict={'fontsize': 16})
    plt.ylabel('True positive rate', fontdict={'fontsize': 16})
    plt.legend(loc='lower right', fontsize='medium');


def plot_roc(estimators, X, y, figure_size=(16, 6)):
    '''
    Plot both confusion matrix and ROC curce on the same figure.

    Parameters:
    -----------
    estimators : dict
        key, value for model name and sklearn.estimator to use for predicting
        class probabilities.
    X : array_like
        data to predict class probabilities.
    y : array_like
        true label vector.
    figure_size : tuple (optional)
        size of the figure.

    Returns:
    --------
    plot : matplotlib.pyplot
        plot confusion matrix and ROC curve.
    '''
    plt.figure(figsize=figure_size)
    for estimator in estimators.keys():
        # Compute tpr, fpr, auc and confusion matrix
        fpr, tpr, thresholds = roc_curve(
            y, estimators[estimator].predict_proba(X)[:, 1])
        auc = roc_auc_score(y, estimators[estimator].predict_proba(X)[:, 1])

        # Plot ROC curce
        plt.plot(fpr, tpr, label=f'{estimator}: auc = {auc:.3f}')
        plt.title('ROC curve', y=1, fontdict={'fontsize': 20})
        plt.legend(loc='lower right', fontsize='medium')

    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('False positive rate', fontdict={'fontsize': 16})
    plt.ylabel('True positive rate', fontdict={'fontsize': 16});


def plot_feature_imp(
        feat_imp, feat_names, figsize=(12, 8), title='Feature Importance'):
    indices = np.argsort(feat_imp)
    names = np.array([feat_names[i] for i in indices])
    plt.figure(figsize=figsize)
    plt.barh(range(len(feat_imp)), feat_imp[indices])
    plt.yticks(range(len(feat_imp)), names, fontsize=16)
    plt.title(title, {'fontsize': 20})


def plot_corr_matrix(df, method='pearson', figsize=(10, 6)):
    if method == 'pearson':
        corr_matrix = np.round(df.corr(method), 2)
    elif method == 'spearman':
        corr_matrix = np.round(spearmanr(df).correlation, 2)
    else:
        raise Exception('Valid values for method: [pearson, spearman]')

    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask)] = True

    # Plot the heat map
    plt.figure(figsize=figsize)
    plt.style.use('seaborn-white')
    cmap = sns.diverging_palette(0, 120, as_cmap=True)
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, cmap=cmap, center=0, square=True,
        linewidths=.1, cbar_kws={'shrink': .5}, xticklabels=df.columns,
        yticklabels=df.columns)


def plot_feature_based_hier_clustering(
        X, feature_names, corr_method='spearman', linkage_method='single',
        figsize=(16, 12)):
    '''
    Plot features-based hierarchical clustering based on spearman correlation
    matrix.
    '''
    corr = np.round(spearmanr(X).correlation, 4)
    corr_densed = hierarchy.distance.squareform(1 - corr)
    z = hierarchy.linkage(corr_densed, linkage_method)
    plt.figure(figsize=figsize)
    hierarchy.dendrogram(
        z, orientation='left', labels=feature_names, leaf_font_size=16)


def permutation_importances_cv(
        tree_clf, X_train, y_train, scoring='accuracy', k=3):
    '''
        Compute feature importances using permutation based on `k-fold` cross
        validation of the metric.
    '''

    def scorer(tree_clf):
        score = cross_val_score(
            tree_clf, X_train, y_train, cv=k, scoring=scoring, n_jobs=-1)
        return score.mean()

    base_score = scorer(tree_clf)
    feat_imp = []

    for j in range(X_train.shape[1]):
        temp = X_train[:, j].copy()
        X_train[:, j] = np.random.permutation(X_train[:, j])
        score = scorer(tree_clf)
        feat_imp.append(base_score - score)
        X_train[:, j] = temp

    return np.array(feat_imp)


def plot_pca_var_explained(pca_transformer, figsize=(12, 6)):
    var_ratio = pca_transformer.explained_variance_ratio_
    cum_var_exp = np.cumsum(var_ratio)
    plt.figure(figsize=figsize)
    plt.bar(range(1, len(cum_var_exp) + 1), var_ratio, align='center',
            color='red', label='Individual explained variance')
    plt.step(range(1, len(cum_var_exp) + 1), cum_var_exp,
             where='mid', label='Cumulative explained variance')
    plt.xticks(range(1, len(cum_var_exp)))
    plt.legend(loc='best')
    plt.xlabel('Principal component index', {'fontsize': 14})
    plt.ylabel('Explained variance ratio', {'fontsize': 14})
    plt.title('PCA on training data', {'fontsize': 18})
