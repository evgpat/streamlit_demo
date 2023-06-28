from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pickle import dump, load
import pandas as pd


def split_data(df: pd.DataFrame):
    y = df['Survived']
    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]]

    return X, y


def open_data(path="data/titanic_dataset_train.csv"):
    df = pd.read_csv(path)
    df = df[['Survived', "Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]]

    return df


def preprocess_data(df: pd.DataFrame, test=True):
    df.dropna(inplace=True)

    if test:
        X_df, y_df = split_data(df)
    else:
        X_df = df

    to_encode = ['Sex', 'Embarked']
    for col in to_encode:
        dummy = pd.get_dummies(X_df[col], prefix=col)
        X_df = pd.concat([X_df, dummy], axis=1)
        X_df.drop(col, axis=1, inplace=True)

    if test:
        return X_df, y_df
    else:
        return X_df


def fit_and_save_model(X_df, y_df, path="data/model_weights.mw"):
    model = RandomForestClassifier()
    model.fit(X_df, y_df)

    test_prediction = model.predict(X_df)
    accuracy = accuracy_score(test_prediction, y_df)
    print(f"Model accuracy is {accuracy}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")


def load_model_and_predict(df, path="data/model_weights.mw"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]
    # prediction = np.squeeze(prediction)

    prediction_proba = model.predict_proba(df)[0]
    # prediction_proba = np.squeeze(prediction_proba)

    encode_prediction_proba = {
        0: "Вам не повезло с вероятностью",
        1: "Вы выживете с вероятностью"
    }

    encode_prediction = {
        0: "Сожалеем, вам не повезло",
        1: "Ура! Вы будете жить"
    }

    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = encode_prediction[prediction]

    return prediction, prediction_df


if __name__ == "__main__":
    df = open_data()
    X_df, y_df = preprocess_data(df)
    fit_and_save_model(X_df, y_df)
