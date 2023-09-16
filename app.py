import streamlit as st
import pandas as pd
import numpy as np
import rasterio 
import pickle
import matplotlib.pyplot as plt
import os
from PIL import Image
import tempfile
import io
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix

st.title("Deforestation Detection - WebApp")
st.markdown("Deforestation entails the extensive removal of trees, primarily due to human activities like logging and agriculture, this cause ecological imbalance and climate repercussions. Leverage this advanced model to swiftly determine the existing condition of a specified area – whether it has undergone deforestation or remains unaffected")

st.sidebar.title("Upload ⬆️ ")
st.sidebar.markdown("Upload NDVI satellite image of the area that you want to classify")
st.sidebar.markdown("(Only TIF format images are accepted)")

col1, col2 = st.columns(2)

with open('deforestaion_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

    X_test = np.load('Dataset\X_test.npy')
    y_test = np.load('Dataset\y_test.npy')

y_pred = loaded_model.predict(X_test)

class_names = ['Non-Deforested','Deforested']

 #load the model
@st.cache_data
def prediction(mean_ndvi_value):
    file_name = r'deforestaion_model.pkl'
    with open(file_name, 'rb') as file:
        model = pickle.load(file)
    pred_value = model.predict([[mean_ndvi_value]])
    return pred_value[0] 


#Calculate mean advi
def calculate_mean_ndvi2(tif_path):
    with rasterio.open(tif_path) as src:
        ndvi_data = src.read(1)
        valid_ndvi_values = ndvi_data[~np.isnan(ndvi_data)]
        mean_ndvi = np.mean(valid_ndvi_values)

    return mean_ndvi


def upload_image():
    my_upload = st.sidebar.file_uploader("Upload an image", type=["tif"])
    if my_upload is not None:
        return my_upload
    else:
        st.write("Upload an image to visualize.")

def visualize_tiff_images(upload):
    if upload is not None:
        # Create a temporary file to save the uploaded TIFF image
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_file:
            temp_file.write(upload.read())
            temp_file_name = temp_file.name  # Get the temporary file name
            temp_file.close()  # Explicitly close the temporary file
        
        # Open the temporary file using Rasterio
        with rasterio.open(temp_file_name) as src:
            # Read the NDVI band from the TIFF image
            ndvi_data = src.read(1)

            # Apply a colormap to the NDVI data
            cmap = plt.cm.RdYlGn
            cmap.set_bad(color='brown')
            img_array = cmap(ndvi_data)

            # Convert the array to an RGB PIL Image
            rgb_img_array = (255 * img_array[:, :, :3]).astype(np.uint8)  # Keep only RGB channels
            img = Image.fromarray(rgb_img_array)

            # Convert the RGB PIL Image to JPEG format
            jpeg_buffer = io.BytesIO()
            img.save(jpeg_buffer, format="JPEG")
            jpeg_img = jpeg_buffer.getvalue()

            # Display the JPEG image in Streamlit with a smaller width
            col1.image(jpeg_img, caption="Uploaded NDVI Image", use_column_width=False, width=300)

            #masked
            agree = st.sidebar.checkbox('Mark Deforested Area')
            if agree:
                threshold = 0.33
                deforested_mask = ndvi_data < threshold
                ndvi_data[deforested_mask] = np.nan

                # Apply a colormap to the NDVI data
                cmap = plt.cm.RdYlGn
                cmap.set_bad(color='red')
                img_array = cmap(ndvi_data)

                # Convert the array to an RGB PIL Image
                rgb_img_array = (255 * img_array[:, :, :3]).astype(np.uint8)  # Keep only RGB channels
                img = Image.fromarray(rgb_img_array)

                # Convert the RGB PIL Image to JPEG format
                jpeg_buffer = io.BytesIO()
                img.save(jpeg_buffer, format="JPEG")
                jpeg_img = jpeg_buffer.getvalue()

                # Display the JPEG image in Streamlit with a smaller width
                col2.image(jpeg_img, caption="Deforested Areas", use_column_width=False, width=300)
                return temp_file_name
            return temp_file_name
    else:
        st.sidebar.text("Upload a TIFF image to visualize.")

#performance matrices related functions
def show_accuracy():
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("Accuracy")
    st.write(accuracy)

def show_precision():
    precision = precision_score(y_test, y_pred)
    st.subheader("Precision")
    st.write(precision)

def show_confusion_matrix():
    st.subheader("Confusion Matrix")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plot_confusion_matrix(loaded_model,X_test,y_test,display_labels = class_names)
    st.pyplot()
    #st.write(conf_matrix)


def main():
    
    uploaded_image =  upload_image()
    
    if uploaded_image:
        input_img = visualize_tiff_images(uploaded_image)

    if upload_image is not None:
        st.sidebar.title("Results")

        if st.sidebar.button('Deforestation Status of Area'):
            mean_ndvi = calculate_mean_ndvi2(input_img)
            st.write("Mean NDVI Value:", mean_ndvi)
            prediction_result = prediction(mean_ndvi)
            
            if prediction_result == 1:
                st.write("The area is classified as Deforested Area.")
            else:
                st.write("The area is classified as non-deforested Area.")

    st.sidebar.title("Model Performance Matrices")
    option = st.sidebar.selectbox(
    'Select a performance measure',
    ('-','Accuracy', 'Precision', 'Confusion Matrix'))

    options_dict = {
    'Accuracy': show_accuracy,
    'Precision': show_precision,
    'Confusion Matrix': show_confusion_matrix,
    }
    
    if option in options_dict:
        options_dict[option]()


if __name__ == '__main__':
    main()
