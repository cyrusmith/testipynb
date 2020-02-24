import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pylab as plt
import os
import numpy as np

from tensorflow import keras

tf.compat.v1.enable_eager_execution()


def save_image(image, to_file):
    plt.imshow(image, cmap=plt.cm.binary)
    plt.savefig(to_file)


CIFAR10_CATEGS = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck']


# https://www.tensorflow.org/datasets

def main():
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    print("tri={}, trl={}, tsi={}, tsl={}".format(
        train_images.shape,
        train_labels.shape,
        test_images.shape,
        test_labels.shape))

    # save_image(train_images[0], "image-{}.png".format(train_labels[0]))

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(30, activation='sigmoid'),
        keras.layers.Dense(10, activation='sigmoid')
    ])

    if os.path.exists('one.model.index'):
        model.load_weights('one.model')
    else:
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0),
                      loss='sparse_categorical_crossentropy',  # 'sparse_categorical_crossentropy'
                      metrics=['accuracy'])
        model.fit(train_images, train_labels, epochs=10, validation_split=0.1)
        model.save_weights('one.model')

    predictions = model.predict(test_images[:3])
    print("predictions: {}".format(np.argmax(predictions, axis=1)))
    print("labels: {}".format(test_labels[:3]))


if __name__ == '__main__':
    main()
