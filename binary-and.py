import tensorflow as tf
from tensorflow import keras
import numpy as np

train_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int16)
train_labels = np.array([0, 0, 0, 1], dtype=np.int16)

repeatition = 2000

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

train_data = np.repeat(train_data, repeatition, 0)
train_labels = np.repeat(train_labels, repeatition, 0)
train_data, train_labels = unison_shuffled_copies(train_data, train_labels)
print(train_data.shape, train_labels.shape)

test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int16)
test_labels = np.array([0, 0, 0, 1], dtype=np.int16)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(2, )),
    keras.layers.Dense(2, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])

model.summary()

model.fit(train_data, train_labels, epochs=50, validation_data=(test_data, test_labels), verbose=2)

history = model.evaluate(test_data, test_labels)

print(history)