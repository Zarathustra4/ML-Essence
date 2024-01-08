from model_service.classifier_service import ClassifierService
import streamlit as st
import pandas as pd
from os import makedirs, path


def interface():
    st.title("Binary Classificator Interface")
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
        df = pd.DataFrame(prediction[:100], columns=['Predicted Values'])
        st.title("Prediction Visualization")
        st.bar_chart(prediction[:100])
        st.scatter_chart(df)
        st.divider()
        
        st.table(df)


if __name__ == "__main__":
    service = ClassifierService()
    #prediction = service.predict_by_csv("water-quality")

    #print(type(prediction))
    #print(f"Ten first predictions - {prediction[:10]}")
    #print(f"Shape of prediction {prediction.shape}")
    interface()