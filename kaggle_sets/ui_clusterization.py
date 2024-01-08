from model_service.clusterer_service import ClustererService
import streamlit as st
import pandas as pd
from os import makedirs, path
import matplotlib.pyplot as plt

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
        
        prediction = service.cluster_data_by_csv("uploaded")
  
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


if __name__ == "__main__":
    service = ClustererService()
    #prediction = service.cluster_data_by_csv("airline-satisfaction")

    #print(type(prediction))
    #print(f"Ten first predictions - {prediction[:10]}")
    #print(f"Shape of prediction {prediction.shape}")
    interface()