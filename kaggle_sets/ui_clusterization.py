from model_service.clusterer_service import ClustererService
import streamlit as st
import pandas as pd
from os import makedirs, path
import matplotlib.pyplot as plt
import config as conf

UPDATED_PATH = path.join(conf.BASE_DATASET_PATH, "tmp", "uploaded.csv")

"""
TODO: Now PCA is applied in predict_by_csv method. 
Use it only for plotting. User has to receive updated dataframe
"""


def prediction_to_userdata(prediction):
    clusters = prediction["Clusters"]
    return list(map(lambda x: "satisfied" if x == 1 else "neutral or dissatisfied", clusters))


def order_cols(df: pd.DataFrame, first_column="Clusters") -> tuple:
    cols = df.columns.tolist()
    cols.remove(first_column)
    return (first_column, ) + tuple(cols)


def interface():
    st.title("Clustering Model Interface")
    st.divider()

    uploaded_file = st.file_uploader("Upload CSV")

    if uploaded_file is not None:
        makedirs(path.join(conf.BASE_DATASET_PATH, "tmp"), exist_ok=True)
        with open(UPDATED_PATH, "wb") as file:
            file.write(uploaded_file.read())

        dataframe = pd.read_csv(UPDATED_PATH, delimiter=",")
        dataframe = dataframe.drop(['Unnamed: 0', 'id'], axis=1)
        dataframe = dataframe.dropna()
        st.write(dataframe)
        st.divider()

        prediction = service.cluster_data_by_csv(UPDATED_PATH)

        st.title("Prediction Visualization")
        # Get X, Y, Z values from the DataFrame
        x_values = prediction['col1']
        y_values = prediction['col2']
        z_values = prediction['col3']

        # Create a 3D scatter plot
        st.title("3D Scatter Plot of Clustering:")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        cmap = 'viridis'

        scatter = ax.scatter(x_values, y_values, z_values, s=40, c=prediction['Clusters'], marker='o', cmap=cmap)

        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Cluster')

        ax.set_title("The Plot Of The Clusters")
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        st.pyplot(fig)

        dataframe["Clusters"] = prediction_to_userdata(prediction)

        st.dataframe(dataframe, column_order=order_cols(dataframe))


if __name__ == "__main__":
    service = ClustererService()
    interface()
