import numpy as np

from model_service.classifier_service import ClassifierService
import streamlit as st
import pandas as pd
from os import makedirs, path
import config as conf

UPLOADED_PATH = path.join(conf.BASE_DATASET_PATH, "tmp", "uploaded.csv")


def prediction_to_safety(prediction: np.ndarray) -> list:
    return list(map(lambda x: "safe" if x[0] == 1 else "not safe", prediction.tolist()))


def get_column_order(df: pd.DataFrame, first_col: str):
    columns = df.columns.tolist()
    columns.remove(first_col)
    return (first_col,) + tuple(columns)


def interface():
    st.title("Water safety prediction")
    st.divider()

    uploaded_file = st.file_uploader("Upload CSV")

    if uploaded_file is not None:
        with open(UPLOADED_PATH, "wb") as file:
            file.write(uploaded_file.read())

        dataframe = pd.read_csv(UPLOADED_PATH, delimiter=",")
        st.write(dataframe)
        st.divider()

        prediction = service.predict_by_csv(UPLOADED_PATH)

        dataframe["is safe"] = prediction_to_safety(prediction)

        st.write("Prediction")
        st.dataframe(dataframe, column_order=get_column_order(dataframe, "is safe"))


if __name__ == "__main__":
    service = ClassifierService()

    interface()
