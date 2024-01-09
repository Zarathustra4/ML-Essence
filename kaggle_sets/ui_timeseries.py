from model_service.forecast_service import ForecastService
import streamlit as st
import pandas as pd
from os import makedirs, path


def interface():
    st.title("Time Series Forecasting Interface")
    st.divider()

    uploaded_file = st.file_uploader("Upload CSV")

    if uploaded_file is not None:
        makedirs("kaggle_sets\datasets", exist_ok=True)
        file_path = path.join("kaggle_sets\datasets", "uploaded.csv")
        with open(file_path, "wb") as file:
            file.write(uploaded_file.read())

        dataframe = pd.read_csv("kaggle_sets/datasets/uploaded.csv", delimiter=",")
        st.write(dataframe)
        st.divider()
        
        prediction = service.predict_by_csv("uploaded")
        df = pd.DataFrame(prediction, columns=['Predicted Values'])
        st.title("Prediction Visualization")
        st.area_chart(df)


if __name__ == "__main__":
    service = ForecastService()
    #prediction = service.predict_by_csv("daily-min-temperatures")

    #print(type(prediction))
    #print(f"Ten first predictions - {prediction[:10]}")
    #print(f"Shape of prediction {prediction.shape}")
    interface()