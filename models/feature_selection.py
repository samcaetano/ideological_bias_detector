"""This is for feature selection"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif


def feature_selector_Kbest(X_train, y_train, X_test, y_test):
    """
    Function for feature selection

    Args:
        X_train, X_test: dataframe containing text
        y_train, y_test: column containing hyperpartisanship labels

    Returns:
        Selected features with high dependencies matching labels
    """
    # Split dataset to select feature and evaluate the classifier

    # Baseline (92 attributes)
    clf = make_pipeline(MinMaxScaler(), LinearSVC())
    clf.fit(X_train, y_train)
    print(f'Dummy classification accuracy without selecting features: {clf.score(X_test, y_test)}')

    K = 2
    scores = []
    while K <= 92:
        print(f'Feature selection for K = {K}')
        
        clf_selected = make_pipeline(
            SelectKBest(f_classif, k=K),
            MinMaxScaler(),
            LinearSVC(),
        )
        
        clf_selected.fit(X_train, y_train)

        accuracy = clf_selected.score(X_test, y_test)
        print('Classification accuracy after univariate feature selection: {:.3f}'.format(accuracy))
        scores.append(accuracy)
        K += 1
        print()

    # Find best K and add 2 as an offset
    best_k = scores.indes(max(scores)) + 2

    selector = SelectKBest(f_classif, k=best_k).fit(X_train, y_train)
    selected_features = selector.get_support(indices=True)

    return X_train.columns[selected_features].tolist()