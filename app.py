import pandas as pd
import streamlit as st
from PIL import Image
from model import open_data, preprocess_data, split_data, load_model_and_predict


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    image = Image.open('data/titanic.jpg')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Demo Titanic",
        page_icon=image,

    )

    st.write(
        """
        # Классификация пассажиров титаника
        Определяем, кто из пассажиров выживет, а кто – нет.
        """
    )

    st.image(image)


def write_user_data(df):
    st.write("## Ваши данные")
    st.write(df)


def write_prediction(prediction, prediction_probas):
    st.write("## Предсказание")
    st.write(prediction)

    st.write("## Вероятность предсказания")
    st.write(prediction_probas)


def process_side_bar_inputs():
    st.sidebar.header('Заданные пользователем параметры')
    user_input_df = sidebar_input_features()

    train_df = open_data()
    train_X_df, _ = split_data(train_df)
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
    preprocessed_X_df = preprocess_data(full_X_df, test=False)

    user_X_df = preprocessed_X_df[:1]
    write_user_data(user_X_df)

    prediction, prediction_probas = load_model_and_predict(user_X_df)
    write_prediction(prediction, prediction_probas)


def sidebar_input_features():
    sex = st.sidebar.selectbox("Пол", ("Мужской", "Женский"))
    embarked = st.sidebar.selectbox("Порт посадки", (
    "Шербур-Октевиль", "Квинстаун", "Саутгемптон"))
    pclass = st.sidebar.selectbox("Класс", ("Первый", "Второй", "Третий"))

    age = st.sidebar.slider("Возраст", min_value=1, max_value=80, value=20,
                            step=1)

    sib_sp = st.sidebar.slider(
        "Количетсво ваших братьев / сестер / супругов на борту",
        min_value=0, max_value=10, value=0, step=1)

    par_ch = st.sidebar.slider("Количетсво ваших детей / родителей на борту",
                               min_value=0, max_value=10, value=0, step=1)

    translatetion = {
        "Мужской": "male",
        "Женский": "female",
        "Шербур-Октевиль": "C",
        "Квинстаун": "Q",
        "Саутгемптон": "S",
        "Первый": 1,
        "Второй": 2,
        "Третий": 3,
    }

    data = {
        "Pclass": translatetion[pclass],
        "Sex": translatetion[sex],
        "Age": age,
        "SibSp": sib_sp,
        "Parch": par_ch,
        "Embarked": translatetion[embarked]
    }

    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    process_main_page()
