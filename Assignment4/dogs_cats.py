from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


def generate_model():
    classifier = Sequential()
    classifier.add(
        Conv2D(32, (3, 3), input_shape=(64, 64, 3))
    )
    classifier.add(Activation('relu'))
    classifier.add(
        MaxPooling2D(pool_size=(2, 2))
    )
    classifier.add(
        Conv2D(32, (3, 3))
    )
    classifier.add(Activation('relu'))
    classifier.add(
        MaxPooling2D(pool_size=(2, 2))
    )
    classifier.add(
        Conv2D(32, (3, 3))
    )
    classifier.add(
        Activation('relu')
    )
    classifier.add(MaxPooling2D(
        pool_size=(2, 2))
    )
    classifier.add(
        Flatten()
    )
    classifier.add(

        Dense(64)
    )
    classifier.add(
        Activation('relu')
    )
    classifier.add(
        Dropout(0.5)
    )
    classifier.add(
        Dense(1)
    )
    classifier.add(
        Activation('sigmoid')
    )
    classifier.summary()
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    training_set = train_datagen.flow_from_directory(
        './train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary'
    )
    test_set = test_datagen.flow_from_directory(
        './test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary'
    )
    classifier.fit(
        training_set,
        steps_per_epoch=625,
        epochs=40,
        validation_data=test_set,
        validation_steps=5000
    )
    classifier.save('catdog_cmm.h5')


def test_model():
    model = load_model('./catdog_cmm.h5')
    cat = np.expand_dims(
        image.img_to_array(
            image.load_img(
                './test/cats/cat.1.jpg',
                target_size=(64, 64)
            )
        ),
        axis=0
    )
    dog = np.expand_dims(
        image.img_to_array(
            image.load_img(
                './test/dogs/dog.1.jpg',
                target_size=(64, 64)
            )
        ),
        axis=0
    )
    result_cat = model.predict(cat)
    result_dog = model.predict(dog)
    print(result_cat, result_dog)


if __name__ == '__main__':
    generate_model()
    test_model()
