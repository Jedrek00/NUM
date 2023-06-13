import os
import tensorflow as tf
import mlflow

# code paritally adapted from https://www.tensorflow.org/datasets/keras_example

IMG_SIZE = 224
PATH_1 = "data/Pistachio_Image_dataset/Kirmizi_Pistachio/"
PATH_2 = "data/Pistachio_Image_dataset/Siirt_Pistachio/"

batch_size = 32
learning_rate = 0.001
epochs = 10

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return tf.cast(image, tf.float32) / 255., label


def create_dataset_from_dirs(folder1, folder2, test_split=0.2):
    file_list1 = tf.data.Dataset.list_files(os.path.join(folder1, '*.jpg'))
    file_list2 = tf.data.Dataset.list_files(os.path.join(folder2, '*.jpg'))

    ds1 = file_list1.map(lambda x: (tf.image.decode_jpeg(tf.io.read_file(x)), 0))
    ds2 = file_list2.map(lambda x: (tf.image.decode_jpeg(tf.io.read_file(x)), 1))

    ds = ds1.concatenate(ds2)
    ds = ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.cache()
    ds = ds.shuffle(len(file_list1) + len(file_list2))
    num_samples = len(file_list1) + len(file_list2)
    test_size = int(num_samples * test_split)

    ds_train = ds.skip(test_size)
    ds_test = ds.take(test_size)

    ds_train = ds_train.cache()
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.cache()
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    print(len(ds_train), len(ds_test))

    return ds_train, ds_test


def train_model(model, ds_train, ds_test):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    history = model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_test,
    )

    train_loss=history.history['loss'][-1]
    train_acc=history.history['sparse_categorical_accuracy'][-1]
    val_loss=history.history['val_loss'][-1]
    val_acc=history.history['val_sparse_categorical_accuracy'][-1]

    print("train_loss: ", train_loss)
    print("train_accuracy: ", train_acc)
    print("val_loss: ", val_loss)
    print("val_accuracy: ", val_acc)

    tf.keras.models.save_model(model, "./model")

    with mlflow.start_run(run_name="run_name"):
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("val_loss", val_loss)
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_artifacts("./model")


def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    return model


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model = create_model()

    ds_train, ds_test = create_dataset_from_dirs(PATH_1, PATH_2)
    train_model(model, ds_train, ds_test)