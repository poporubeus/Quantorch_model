import matplotlib.pyplot as plt
import numpy as np
from dataset import Dataset


new_shape = 8
dataset = Dataset(classes_of_items=[0,1], num_train_samples=100, shuffle=True, resize=new_shape, my_seed=999)
X_train, y_train, X_val, y_val = dataset.data_generator()

def plot_data(name_set, nrows, ncols):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    n_item = int(nrows*ncols)
    if name_set == "training":
        x_array = X_train
        y_array = y_train
        for ax, (image, label) in zip(axs.flatten(), zip(x_array[:n_item], y_array[:n_item])):
            ax.imshow(image.reshape((8, 8)), cmap='viridis')
            ax.set_title(f"Train: {label}")
            ax.axis('off')
    elif name_set == "validation":
        x_array = X_val
        y_array = y_val
        for ax, (image, label) in zip(axs.flatten(), zip(x_array[:n_item], y_array[:n_item])):
            ax.imshow(image.reshape((8, 8)), cmap='viridis')
            ax.set_title(f"Val: {label}")
            ax.axis('off')
    else:
        raise ValueError("Only training and testing are valid name, use one of the two provided names!")
    plt.tight_layout()
    return plt.show()

rows_to_show = 4
cols_to_show = 5
if __name__ == "__main__":
    plot_data(name_set="validation", nrows=rows_to_show, ncols=cols_to_show)


