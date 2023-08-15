import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np

new_model = load_model(os.path.join('models','cats-or-dogs-ann.h5'))

img = cv2.imread('dog_test.jpg')
resize = tf.image.resize(img, (256,256))

yhat = new_model.predict(np.expand_dims(resize/255, 0))

if yhat > 0.5: 
    print(f'Predicted class is DOG')
else:
    print(f'Predicted class is CAT')
