import torch
from quantum_model import *
from dataset import Dataset
import matplotlib.pyplot as plt


# Generate the train and validation dataset
new_shape = 8
dataset = Dataset(classes_of_items=[0, 1, 2, 3], num_train_samples=500, shuffle=True,
                  resize=new_shape, my_seed=999, interface="torch")
X_train, y_train, X_val, y_val = dataset.data_generator()

# Assign the weight_shapes to the model as the number of parameters expected by the circuit
weight_shapes = {"learning_params": (param_per_gate*n_qubits + param_per_gate*3 + param_per_gate*n_qubits, layers)}

dev = qml.device('default.qubit', wires=n_qubits)
params_per_each_layer = param_per_gate*layers + param_per_gate*3 + param_per_gate*n_qubits
@qml.qnode(dev)
def quantum_neural_network(inputs: np.ndarray, learning_params: np.ndarray):
    # Call the instance of the quantum model
    quantum_model = QuantumModel(qubits=n_qubits)
    quantum_model.QuantumFeatureMap(X=inputs)
    quantum_model.Ring_like_layer(params=learning_params[:, 0])
    qml.Barrier(only_visual=True)  # first layer
    quantum_model.QuantumFeatureMap(X=inputs)
    quantum_model.Ring_like_layer(params=learning_params[:, 1])
    qml.Barrier(only_visual=True)  # second layer
    quantum_model.QuantumFeatureMap(X=inputs)
    quantum_model.Ring_like_layer(params=learning_params[:, 2])
    qml.Barrier(only_visual=True)  # third layer
    return qml.probs(wires=[0, 1])

# Configure the torch model purely quantum to make multi-label classification
learning_rate = 5e-4
QUANTUM_LAYER = qml.qnn.TorchLayer(quantum_neural_network, weight_shapes)
MODEL = torch.nn.Sequential(*[QUANTUM_LAYER])
optim = torch.optim.Adam(MODEL.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()
batch_size = 16


X = torch.tensor(X_train, requires_grad=True).float()
Xval = torch.tensor(X_val, requires_grad=True).float()
y = y_train

# Create train and validation data loader
train_dataloader = torch.utils.data.DataLoader(
    dataset=list(zip(X, y)), batch_size=batch_size, shuffle=True, drop_last=True
)
val_data_loader = torch.utils.data.DataLoader(
    dataset=list(zip(Xval, y_val)), batch_size=batch_size, shuffle=True, drop_last=True
)


# Set up the training process
loss_list, loss_val_list, acc_list, acc_val_list = [], [], [], []
n_epochs = 200

def train(n_epochs: int):
    for epoch in range(n_epochs):
        train_loss = 0.
        correct = 0
        val_correct = 0
        total_samples = 0.
        total_samples_val = 0.
        MODEL.train()
        for xs, ys in train_dataloader:
            optim.zero_grad()

            y_pred = MODEL(xs)
            loss_value = loss_fn(y_pred, ys)

            loss_value.backward()
            optim.step()
            train_loss += loss_value.item()

            # Calculate accuracy
            predicted = torch.argmax(y_pred, 1)
            correct += (predicted == ys).sum().item()
            total_samples += ys.size(0)


        val_loss = 0.
        with torch.no_grad():
            MODEL.eval()
            for xv, yv in val_data_loader:
                yval_pred = MODEL(xv)
                loss_value = loss_fn(yval_pred, yv)
                val_loss += loss_value.item()

                # Accuracy on validation
                v_predicted = torch.argmax(yval_pred, 1)
                val_correct += (v_predicted == yv).sum().item()
                total_samples_val += yv.size(0)


        avg_loss = train_loss / len(train_dataloader)
        avg_acc = correct / total_samples
        avg_loss_val = val_loss / len(val_data_loader)
        avg_acc_val = val_correct / total_samples_val

        loss_list.append(avg_loss)
        acc_list.append(avg_acc)

        loss_val_list.append(avg_loss_val)
        acc_val_list.append(avg_acc_val)

        print(
            f"Epoch {epoch + 1}: ---train_loss: {avg_loss}, ---train_acc.: {avg_acc},"
            f"---val_loss: {avg_loss_val},"f"---val_acc: {avg_acc_val}")
        return (
                loss_list,
                acc_list,
                loss_val_list,
                acc_val_list
        )

if __name__ == "__main__":
    train_loss, train_acc, val_loss, val_acc = train(n_epochs)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6,4))
    fig.suptitle("Train and validation loss / accuracy")
    axs[0].plot(loss_list, label="Train Loss", color="royalblue")
    axs[0].plot(val_loss, label="Validation Loss", color="darkblue")
    axs[1].plot(acc_list, label="Train Accuracy", color="orangered")
    axs[1].plot(val_acc, label="Validation Accuracy", color="tomato")
    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()
    plt.show()