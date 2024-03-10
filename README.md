The code contains a purely quantum neural network model trained on MNIST dataset to predict multilabel images.
The method that has been used to create the circuit emulates the data re-uploading technique, where the feature map 
is repeated three times embedding the same features, while a parametrized quantum layer is applied after each feature map but learning updating parameters.
The classical Deep Learning interface used in this work is Torch, combined with Pennylane package to create the quantum layer connected to the Torch model.
The main difference between the Torch hybridized models is that this model is purely quantum, and it does not mix any classical convolutional, flatten even pooling layer.
In the config.py file you will find some constants used within the trianing process. 
The logic behind the training process is to evaluate the training loss and the validation loss at the same time in order to detect possible overfitting.
The evalutation on the test set will be uploaded later on.
