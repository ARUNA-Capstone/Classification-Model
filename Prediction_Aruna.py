#


import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import keras
from PIL import Image
#from keras.applications.inception_v3 import preprocess_input

# Gadang
# Honai
# Joglo
# Panjang
# Tongkonan


def getLabel():
    label = ['Gadang','Honai','Joglo','Panjang','Tongkonan']
    return label

# Load pre-trained ResNet50 model
def loadmodel():
    model = keras.models.load_model('cnn_model1.h5')
    return model

def cropimage(image_path):
    image_path = image_path

    img = Image.open(image_path).convert("RGB")
    # Get the dimensions of the original image
    original_width, original_height = img.size

    # Calculate the size of the square crop
    crop_size = min(original_width, original_height)

    # Calculate the coordinates for the crop
    left = (original_width - crop_size) // 2
    top = (original_height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    # Crop the image to a square centered region
    cropped_image = img.crop((left, top, right, bottom))
    cropped_dir = 'croppedimage/cropped.png'
    cropped_image.save(cropped_dir)
    return cropped_dir

def predict_class(image_path):
    model = loadmodel()
    #crop the picture
    cropimage(image_path)

    # Load and preprocess the image
    img = Image.open('croppedimage/cropped.png').convert("RGB")
    img = img.resize((224, 224))
    img_array = np.asarray(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Get model predictions
    predictions = model.predict(img_array)

    return predictions

# Replace 'path_to_your_image.jpg' with the path to your image file
image_path = 'WhatsApp Image 2023-12-11 at 9.44.31 AM.jpeg'

class_names = getLabel()
predicted_class = predict_class(image_path)
print(predicted_class)

for label in class_names:
    score = tf.nn.softmax(predicted_class[0])

print(score)
print('Predicted class: {}; with a {:.2f} percent confidence.'.format(class_names[np.argmax(score)], 100 * np.max(score)),'\n')
