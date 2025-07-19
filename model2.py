# -*- coding: utf-8 -*-

!pip install tensorflow==2.15.1
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import math
import os

SAMPLES = 11000 #increased number of samples test on to 10000
x_values = np.random.uniform(low=0, high=2 * math.pi, size=SAMPLES).astype(np.float32)
np.random.shuffle(x_values)
y_values = np.sin(x_values).astype(np.float32)

y_values += 0.1 * np.random.randn(*y_values.shape)

TRAIN_SPLIT = int(0.6 * SAMPLES)
TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)
x_train, x_test, x_validate = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_test, y_validate = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

model_2 = tf.keras.Sequential()
model_2.add(keras.layers.Dense(64, activation='relu', input_shape=(1,)))
model_2.add(keras.layers.Dense(64, activation='relu'))
model_2.add(keras.layers.Dense(32, activation='relu'))



model_2.add(keras.layers.Dense(1))
model_2.compile(optimizer='rmsprop', loss="mse", metrics=["mae"])

history_2 = model_2.fit(x_train, y_train, epochs=400, batch_size=16, validation_data=(x_validate, y_validate))

train_loss = history_2.history['loss']
val_loss = history_2.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'g.', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

predictions = model_2.predict(x_train)
plt.clf()
plt.title('Training data predicted vs actual values')
plt.plot(x_test, y_test, 'b.', label='Actual')
plt.plot(x_train, predictions, 'r.', label='Predicted')
plt.legend()
plt.show()

MODELS_DIR = 'models/'
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)


model_2.save(MODELS_DIR + 'model_2.h5')

model_path = MODELS_DIR + 'model_2.h5'
model_size = os.path.getsize(model_path)

print("OG model size: %d bytes" % model_size)

import tensorflow as tf
import numpy as np
import os

converter = tf.lite.TFLiteConverter.from_keras_model(model_2)
tflite_model = converter.convert()

open("habeel_model2.tflite", "wb").write(tflite_model)

converter = tf.lite.TFLiteConverter.from_keras_model(model_2)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset_generator():
    for value in x_test:
        yield [np.array(value, dtype=np.float32, ndmin=2)]

converter.representative_dataset = representative_dataset_generator

tflite_model = converter.convert()

open("habeel_model2_quantized.tflite", "wb").write(tflite_model)

basic_model_size = os.path.getsize("habeel_model2.tflite")
print("Basic model is %d bytes" % basic_model_size)

quantized_model_size = os.path.getsize("habeel_model2_quantized.tflite")
print("Quantized model is %d bytes" % quantized_model_size)

difference = basic_model_size - quantized_model_size
print("Difference is %d bytes" % difference)



!apt-get -qq install xxd
!xxd -i habeel_model2_quantized.tflite > habeel_model2_quantized.cc
!cat habeel_model2_quantized.cc
#