import tensorflow as tf
import tensorflow.keras.layers as layers


def main():
    mnist = tf.keras.datasets.mnist

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # One-hot encoding
    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)

    # normalise inputs
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train.reshape(
        x_train.shape[0], 28, 28, 1
    )

    x_test = x_test.reshape(
        x_test.shape[0], 28, 28, 1
    )

    model = tf.keras.models.Sequential()

    input_layer = layers.InputLayer(input_shape=(28, 28, 1))
    model.add(input_layer)
    model.summary()

    weights = tf.keras.initializers.RandomNormal()
    biases = tf.keras.initializers.Constant(0.1)

    conv_layer_1 = layers.Conv2D(
        25, (12, 12), strides=(2, 2), padding="valid",
        kernel_initializer=weights, bias_initializer=biases,
        activation='relu')
    model.add(conv_layer_1)

    conv_layer_2 = layers.Conv2D(
        64, (5, 5), padding="same", activation="relu", bias_initializer=biases,
        kernel_initializer=weights
    )
    model.add(conv_layer_2)

    max_pool_layer = layers.MaxPooling2D((2, 2))
    model.add(max_pool_layer)

    flatten_layer = layers.Flatten()
    model.add(flatten_layer)

    conn_layer_1 = layers.Dense(
        1024, activation="relu", kernel_initializer=weights,
        bias_initializer=biases
    )
    model.add(conn_layer_1)

    dropout_layer = layers.Dropout(0.2)
    model.add(dropout_layer)

    conn_layer_2 = layers.Dense(
            10,
            activation="softmax",
            kernel_initializer=weights,
            bias_initializer=biases
        )
    model.add(conn_layer_2)

    model.summary()

    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=50, epochs=4)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('Test accuracy:', test_acc)


""" Outputs
    Epoch 1/4
    1200/1200 [==============================] - 19s 16ms/step - loss: 0.3359 - accuracy: 0.9014
    Epoch 2/4
    1200/1200 [==============================] - 20s 16ms/step - loss: 0.1052 - accuracy: 0.9688
    Epoch 3/4
    1200/1200 [==============================] - 20s 17ms/step - loss: 0.0703 - accuracy: 0.9789
    Epoch 4/4
    1200/1200 [==============================] - 20s 17ms/step - loss: 0.0536 - accuracy: 0.9834
    313/313 - 1s - loss: 0.0422 - accuracy: 0.9860
    Test accuracy: 0.9860000014305115
"""


main()