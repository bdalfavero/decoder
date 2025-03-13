import marimo

__generated_with = "0.11.19"
app = marimo.App(width="medium")


@app.cell
def _():
    from copy import deepcopy
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_validate
    return RandomForestClassifier, cross_validate, deepcopy, mo, np, pd, plt


@app.cell
def _(pd):
    df = pd.read_csv("training_data.csv", index_col="i")
    array = df.to_numpy()
    X = array[:, :-1] # Features
    Y = array[:, -1] # Labels
    return X, Y, array, df


@app.cell
def _(RandomForestClassifier, X, Y, cross_validate):
    clf = RandomForestClassifier(n_estimators=50, max_depth=None,
        min_samples_split=2, random_state=0)
    result = cross_validate(clf, X, y=Y.astype(int), cv=5)
    return clf, result


@app.cell
def _(result):
    print(result['test_score'])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
