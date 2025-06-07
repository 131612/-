import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import sys
import os


MODEL_PATH = 'animal_classifier_finetuned.keras'
IMG_PATH = '90.jpg'
IMG_SIZE = (224, 224)


model = load_model(MODEL_PATH)
print(" 模型加载成功")


img = image.load_img(IMG_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0


predictions = model.predict(img_array)
predicted_index = np.argmax(predictions[0])


class_names = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']

predicted_label = class_names[predicted_index]
confidence = predictions[0][predicted_index]

plt.imshow(img)
plt.title(f"prediction: {predicted_label}(accuracy: {confidence:.2f})")
plt.axis('off')
plt.show()
print("预测原始概率输出：", predictions[0])
print("预测结果索引：", predicted_index)