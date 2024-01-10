import numpy as np

from model_service.regression_service import RegressionService
import streamlit as st
import pandas as pd
from os import path, makedirs
import config as conf

UPLOADED_PATH = path.join(conf.BASE_DATASET_PATH, "tmp", "uploaded.csv")


def prediction_to_user_data(prediction: np.ndarray) -> list:
    return list(map(lambda x: int(x[0]) if x[0] > 0 else 0, prediction))


def get_column_order(df: pd.DataFrame, first_col: str):
    columns = df.columns.tolist()
    columns.remove(first_col)
    return (first_col,) + tuple(columns)


def interface():
    st.title("Mosquito count prediction")
    st.divider()

    uploaded_file = st.file_uploader("Upload CSV")

    if uploaded_file is not None:
        makedirs(path.join(conf.BASE_DATASET_PATH, "tmp"), exist_ok=True)
        with open(UPLOADED_PATH, "wb") as file:
            file.write(uploaded_file.read())

        dataframe = pd.read_csv(UPLOADED_PATH, delimiter=",")
        st.write(dataframe)
        st.divider()

        prediction = service.predict_by_csv(UPLOADED_PATH)
        dataframe["mosquito indicator"] = prediction_to_user_data(prediction)
        st.title("Prediction Visualization")
        st.line_chart(dataframe["mosquito indicator"])
        st.divider()
        st.table(dataframe)


if __name__ == "__main__":
    service = RegressionService()
    interface()
