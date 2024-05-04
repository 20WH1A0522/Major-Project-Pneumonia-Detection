from flask import Flask, render_template, request
import numpy as np
import cv2
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
import base64
from PIL import Image
import io
import os
import tempfile
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
import base64


app = Flask(__name__)

# Set Flask app to debug mode
app.debug = True

# Define the UNet model architecture
def unet(input_size=(256,256,1)):
    inputs = Input(input_size)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])

# Load the pre-trained model
model = unet()
model.load_weights('models/model.h5')


# Preprocess the uploaded image
def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to match the input shape of the model (256x256)
    resized_image = cv2.resize(image, (256, 256))

    # Normalize the pixel values to be in the range [0, 1]
    normalized_image = resized_image / 255.0

    # Expand the dimensions to add a batch dimension (model expects input shape of (None, 256, 256, 1))
    preprocessed_image = np.expand_dims(normalized_image, axis=-1)

    return preprocessed_image



# Prediction function
def predict_segmentation(image_path, model):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    # Perform prediction
    prediction = model.predict(preprocessed_image)

    # Threshold the prediction to get binary mask
    thresholded_prediction = (prediction > 0.5).astype(np.uint8)

    return thresholded_prediction



# Function to display image with prediction
def display_image_with_prediction(image_path, prediction):
    # Read the original image
    original_image = cv2.imread(image_path)

    # Display the original image and the segmented image
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(prediction.squeeze(), cmap='gray')
    axes[1].set_title('Segmented Image')
    plt.show()

    # Print the prediction result
    pneumonia_detected = np.max(prediction) == 1
    if pneumonia_detected:
        print("Pneumonia detected.")
    else:
        print("No pneumonia detected.")



# Route for home page
@app.route('/')
def home():
    return render_template('home.html')

def encode_image(image):
    _, img_encoded = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return img_base64

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        if file:
            try:
                # Save the uploaded file to a temporary location
                temp_dir = tempfile.mkdtemp()
                temp_file_path = os.path.join(temp_dir, file.filename)
                file.save(temp_file_path)

                # Perform segmentation
                segmented_image = predict_segmentation(temp_file_path, model)

                # Encode the segmented image to base64
                segmented_image_base64 = encode_image(segmented_image)
                
                if segmented_image_base64 is not None:
                    return render_template('index.html', segmented_image=segmented_image_base64)

            except Exception as e:
                print(f"Error processing image: {e}")

    return render_template('index.html', error="Please upload a valid image.")




@app.route('/index')
def index_page():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()