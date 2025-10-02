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
    record_defaults = ['Id', 'Weight']
    filename, pawpularity = tf.io.decode_csv(csv_row, record_defaults)
    pawpularity = tf.convert_to_tensor(float(pawpularity), dtype=tf.float32)
    image = read_and_decode(filename, [IMG_HEIGHT, IMG_WIDTH])
    return image, pawpularity
