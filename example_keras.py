import tensorflow

(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tensorflow.keras.models.Sequential([
    tensorflow.keras.layers.Flatten(input_shape=(28, 28)),
    tensorflow.keras.layers.Dense(512, activation=tensorflow.nn.relu),
    tensorflow.keras.layers.Dropout(0.2),
    tensorflow.keras.layers.Dense(10, activation=tensorflow.nn.softmax),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
