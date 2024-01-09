from datetime import timedelta, datetime

import streamlit as st
from os import path
import config as conf
import pandas as pd

from kaggle_sets.model_service.forecast_service import ForecastService

UPLOADED_PATH = path.join(conf.BASE_DATASET_PATH, "tmp", "uploaded.csv")


def extend_df(df: pd.DataFrame, prediction: list):
    last_date = df["Date"].iloc[-1]
    last_date = datetime.strptime(last_date, "%Y-%m-%d")
    future_dates = [(last_date + timedelta(days=x)).date() for x in range(len(prediction))]

    future_df = pd.DataFrame({"Date": future_dates, "Temp": prediction})
    return pd.concat([df, future_df], ignore_index=True)


def interface():
    st.title("Time Series Forecasting Interface")
    st.divider()

    uploaded_file = st.file_uploader("Upload CSV")

    if uploaded_file is not None:
        with open(UPLOADED_PATH, "wb") as file:
            file.write(uploaded_file.read())

        dataframe = pd.read_csv(UPLOADED_PATH, delimiter=",")
        st.write(dataframe)
        st.divider()

        prediction = service.predict_by_csv(UPLOADED_PATH)

        final_df = extend_df(dataframe, prediction)

        st.title("Prediction Visualization")
        st.line_chart(final_df, x="Date", y="Temp")

        st.divider()
        st.title("Full dataframe (with prediction)")
        st.dataframe(final_df)


if __name__ == "__main__":
    service = ForecastService()

    interface()
