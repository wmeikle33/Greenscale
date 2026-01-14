import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def read_and_decode(filename, reshape_dims):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=IMG_CHANNELS)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return tf.image.resize(image, reshape_dims)

def show_image(filename):
    image = read_and_decode(filename, [IMG_HEIGHT, IMG_WIDTH])
    plt.imshow(image.numpy());
    plt.axis('off');
    
def decode_csv(csv_row):
    record_defaults = ['Var1', 'Var2']
    filename, score = tf.io.decode_csv(csv_row, record_defaults)
    score = tf.convert_to_tensor(float(score), dtype=tf.float32)
    image = read_and_decode(filename, [IMG_HEIGHT, IMG_WIDTH])
    return image, score

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") / 255.0)[..., None]  # add channel dim: (N,28,28,1)
x_test  = (x_test.astype("float32")  / 255.0)[..., None]

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=2)
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.3f}")

probs = model.predict(x_test[:5], verbose=0)
print("Predicted:", probs.argmax(axis=1))
print("True     :", y_test[:5])
