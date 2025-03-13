import marimo

__generated_with = "0.11.19"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import cross_validate
    return GaussianNB, cross_validate, mo, np, pd


@app.cell
def _(pd):
    df = pd.read_csv("training_data.csv", index_col="i")
    array = df.to_numpy()
    X = array[:, :-1] # Features
    Y = array[:, -1] # Labels
    return X, Y, array, df


@app.cell
def _(GaussianNB, X, Y, cross_validate):
    gnb = GaussianNB()
    result = cross_validate(gnb, X, y=Y.astype(int), cv=5)
    return gnb, result


@app.cell
def _(result):
    print(result['test_score'])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
