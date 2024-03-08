import tensorflow as tf
from tensorflow import keras
from jax import numpy as jnp
from sklearn.model_selection import train_test_split
import numpy as np


def downsample(x_array, new_size):
    """
    Downsample the dataset according to the size given by the user.
    :arg x_array (numpy array): Array containing the image to downsample.
    :arg new_size (int): New size.
    :returns new_array (numpy array): resized image.
    """
    size = (new_size, new_size)
    ## tensorflow expects a tensor to reshape, so we add 1 extra dimension
    x_array = np.reshape(x_array, (x_array.shape[0], x_array.shape[1], 1))
    new_array = tf.image.resize(x_array, size)
    return new_array.numpy()

class Dataset:
    """
    Class that creates a binary dataset out of Mnist after specifying the classes of images to deal with,
    based on Keras's API.
    The dataset created here is composed by train and validation set.
    arg: classes_of_items (list): classes of images;
    arg: num_train_samples (int): number of training images;
    arg: shuffle (bool): if True the images are shuffled before creating the dataset;
    arg: resize (int): new image dimension (it must be lower than 28);
    arg: myseed (int): seed for reproducibility.

    :returns:
          X_train (array): set of training images of shape (num_images, new_shape*new_shape);
          y_train (array): set of training labels of shape (num_images,);
          X_val (array): set of validation images of shape (num_images, new_shape*new_shape);
          y_val (array): set of validation labels of shape (num_images,).
    Note that the number of validation data is 0.2*num_train_samples.
    """
    def __init__(self, classes_of_items, num_train_samples, shuffle, resize, my_seed, interface="JAX") -> None:
        self.classes_of_items = classes_of_items
        self.num_train_samples = num_train_samples
        self.shuffle = shuffle
        self.resize = resize
        self.my_seed = my_seed
        self.interface = interface
    def data_generator(self):
        np.random.seed(self.my_seed)
        (train_X, train_y), (_, _) = keras.datasets.mnist.load_data()

        X_train_filtered = train_X[np.isin(train_y, [self.classes_of_items[0], self.classes_of_items[1],
                                                     self.classes_of_items[2], self.classes_of_items[3]])]
        y_train_filtered = train_y[np.isin(train_y, [self.classes_of_items[0], self.classes_of_items[1],
                                                     self.classes_of_items[2], self.classes_of_items[3]])]

        X_train_filtered = X_train_filtered.astype('float16') / 255
        X_train_new = []
        if self.resize is not None and self.resize <= 28:
            for train in X_train_filtered:
                X_train_new.append(downsample(train, self.resize))
        else:
            raise Exception("The new size must be smaller than the actual Mnist size that is 28!")
        ### shuffle
        X_train_new = np.array(X_train_new)
        if self.shuffle:
            shuffled_indices = np.arange(len(X_train_new))
            np.random.shuffle(shuffled_indices)
            X_train_new = X_train_new[shuffled_indices]
            y_train_filtered = y_train_filtered[shuffled_indices]

        if self.num_train_samples is not None:
            num_samples_per_class = self.num_train_samples // len(self.classes_of_items)
            selected_indices = []
            for class_idx in self.classes_of_items:
                class_indices = np.where(y_train_filtered == class_idx)[0][:num_samples_per_class]
                selected_indices.extend(class_indices)
            X_train_ = X_train_new[selected_indices]
            y_train_filtered = y_train_filtered[selected_indices]

        X_train_ = X_train_.reshape(X_train_.shape[0], X_train_.shape[1] * X_train_.shape[2])

        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_, y_train_filtered, test_size=0.2, random_state=42)

        return (
            np.asarray(X_train_final),
            np.asarray(y_train_final),
            np.asarray(X_val),
            np.asarray(y_val),
        )






