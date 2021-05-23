from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
valid_x, train_x = train_x[:5000] / 255, train_x[5000:] / 255
valid_y, train_y = train_y[:5000] / 255, train_y[5000:] / 255
class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]
r = 4
cols = 10

plt.figure(figsize=(cols * 1.2, r * 1.2))
for row in range(r):
    for col in range(cols):
        index = cols * row + col
        plt.subplot(r, cols, index + 1)
        plt.imshow(train_x[index], cmap='binary', interpolation='nearest')
        plt.axis('off')
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.show()

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()
print(model.layers)
hidden_1 = model.layers[1]
weights, biases = hidden_1.get_weights()
print(weights)
print(biases)
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=50, validation_data=(valid_x, valid_y))
print(model.evaluate(test_x, test_y))
x_new = test_x[:3]
y_probable = model.predict(x_new)
y_probable.round(2)
y_predicted = np.argmax(model.predict(x_new), axis=-1)
print(y_predicted)
print(np.array(class_names)[y_predicted])
y_new = test_y[:3]
print(y_new)

for index, image in enumerate(x_new):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis('off')
    plt.title(class_names[test_y[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
