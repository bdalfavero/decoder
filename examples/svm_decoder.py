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
    from sklearn import svm
    from sklearn.model_selection import cross_validate
    return cross_validate, deepcopy, mo, np, pd, plt, svm


@app.cell
def _(pd):
    df = pd.read_csv("training_data.csv", index_col="i")
    array = df.to_numpy()
    return array, df


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        There are four error classes: $I$, $X$, $Y$, and $Z$. However, $Y$ = $XZ$. So, the classification task can be mapped onto a pair of problems: 

        1. Does this representative correspond to a bit flip?
        1. Does this representative correspond to a phase flip?

        Let's try using a pair of support vector machines.
        """
    )
    return


@app.cell
def _(array, np):
    X = array[:, :-1] # Features
    Y = array[:, -1] # Labels

    # We need two sets of labels: bitflip and phaseflip
    Y_bitflip = np.zeros(Y.size, dtype=int)
    Y_phaseflip = np.zeros(Y.size, dtype=int)
    for i, y in enumerate(Y):
        if y == 1 or y == 2:
            Y_bitflip[i] = 1
        else:
            Y_bitflip[i] = -1
        if y == 2 or y == 3:
            Y_phaseflip[i] = 1
        else:
            Y_phaseflip[i] = -1
    return X, Y, Y_bitflip, Y_phaseflip, i, y


@app.cell
def _(X, Y_bitflip, Y_phaseflip):
    # Split the dataset into training and test sets.
    ntest = 1_000
    ntraining = X.shape[0] - ntest
    assert ntraining > 0

    X_training = X[:ntraining, :]
    Y_bf_training = Y_bitflip[:ntraining]
    Y_pf_training = Y_phaseflip[:ntraining]
    X_test = X[:-ntest, :]
    Y_bf_test = Y_bitflip[:-ntest]
    Y_pf_test = Y_phaseflip[:-ntest]
    return (
        X_test,
        X_training,
        Y_bf_test,
        Y_bf_training,
        Y_pf_test,
        Y_pf_training,
        ntest,
        ntraining,
    )


@app.cell
def _(X_training, Y_bf_training, Y_pf_training, svm):
    svm_bitflip = svm.SVC()
    svm_bitflip.fit(X_training, Y_bf_training)

    svm_phaseflip = svm.SVC()
    svm_phaseflip.fit(X_training, Y_pf_training)
    return svm_bitflip, svm_phaseflip


@app.cell
def _(X_test, Y_bf_test, Y_pf_test, svm_bitflip, svm_phaseflip):
    print(svm_bitflip.score(X_test, Y_bf_test))
    print(svm_phaseflip.score(X_test, Y_pf_test))
    return


@app.cell
def _(X, Y, cross_validate, svm):
    classifier = svm.SVC()
    result = cross_validate(classifier, X, y=Y.astype(int), cv=5)
    result['test_score']
    return classifier, result


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
