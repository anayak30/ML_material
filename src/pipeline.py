from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import warnings

iris_df = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_df.data, iris_df.target, test_size=0.3, random_state=0)


def run_pipe():
    # first two - fit and transform
    # classifier - only fit
    pipeline_lr = Pipeline([('scalar1', StandardScaler()),
                            ('pcal', PCA(n_components=2)),
                            ('lr_classifier', LogisticRegression(random_state=0))])
    pipeline_dt = Pipeline([('scalar2', StandardScaler()),
                            ('pca2', PCA(n_components=2)),
                            ('dt_classifier', DecisionTreeClassifier())])
    pipeline_rf = Pipeline([('scalar3', StandardScaler()),
                            ('pca3', PCA(n_components=2)),
                            ('rf_classifier', RandomForestClassifier())])
    pipelines = [pipeline_lr, pipeline_dt, pipeline_rf]

    best_accuracy = 0.0
    best_classifier = 0
    best_pipeline = " "

    pipe_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'RandomForest'}

    for pipe in pipelines:
        pipe.fit(X_train, y_train)

    for i, model in enumerate(pipelines):
        print("{} Test Accuracy: {}".format(pipe_dict[i], model.score(X_test, y_test)))

    for i, model in enumerate(pipelines):
        if model.score(X_test, y_test) > best_accuracy:
            best_accuracy = model.score(X_test, y_test)
            best_pipeline = model
            best_classifier = i
    print('Classifier with best accuracy:{}'.format(pipe_dict[best_classifier]))


def hypertune():
    pipe = Pipeline([("classifier", RandomForestClassifier())])  # just an initializer
    grid_param = [
        {"classifier": [LogisticRegression()],
         "classifier__penalty": ['l2'],
         "classifier__C": np.logspace(0, 4, 10),
         "classifier__max_iter": [1000]

         },
        {"classifier": [LogisticRegression()],
         "classifier__penalty": ['l2'],
         "classifier__C": np.logspace(0, 4, 10),
         "classifier__solver": ['newton-cg', 'saga', 'sag', 'liblinear'] , ##This solvers don't allow L1 penalty
         "classifier__max_iter": [1000]
         },
        {"classifier": [RandomForestClassifier()],
         "classifier__n_estimators": [10, 100, 1000],
         "classifier__max_depth": [5, 8, 15, 25, 30, None],
         "classifier__min_samples_leaf": [1, 2, 5, 10, 15, 100],
         "classifier__max_leaf_nodes": [2, 5, 10]}]
    # create a gridsearch of the pipeline, the fit the best model
    gridsearch = GridSearchCV(pipe, grid_param, cv=5, verbose=0, n_jobs=-1)  # Fit grid search
    best_model = gridsearch.fit(X_train, y_train)
    print(best_model.best_estimator_)
    print("The mean accuracy of the model is:", best_model.score(X_test, y_test))


if __name__ == "__main__":
    run_pipe()
    hypertune()
