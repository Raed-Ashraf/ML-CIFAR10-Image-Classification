import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL
from PIL import Image

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model = tf.keras.models.load_model('image_classifier.model')

# # using openCV
# for i in range(6):
#     image = cv.imread(f'test_0{i+1}.jpg')
#     image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#     plt.imshow(image)
#     resized_image = cv.resize(image, (32,32))
#     input_image = np.asarray([np.array(resized_image)])
#     prediction = model.predict(input_image/255)
#     output_index = np.argmax(prediction)
#     plt.xticks()
#     plt.yticks()
#     plt.xlabel(f'I think this is a {class_names[output_index]}')
#     plt.show()

# Image PIL
for i in range(4):
    image = Image.open(f'test_0{i+1}.jpg')
    plt.imshow(image)
    resized_image = image.resize((32,32))
    input_image = np.asarray([np.array(resized_image)])
    prediction = model.predict(input_image/255)
    output_index = np.argmax(prediction)
    plt.xticks()
    plt.yticks()
    plt.title(f'{class_names[output_index]} with {round(np.amax(prediction)*100, 2)}%', fontsize=20)
    plt.show()

