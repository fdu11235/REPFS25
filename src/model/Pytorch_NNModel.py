import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.dummy import DummyClassifier
from tqdm import tqdm
import os


class NNModel(nn.Module):
    def __init__(self, in_dim, n_classes, dropout_rate=0.3):
        super(NNModel, self).__init__()

        self.fc1 = nn.Linear(in_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.fc4 = nn.Linear(32, n_classes)

        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.dropout(self.activation(self.bn1(self.fc1(x))))
        x = self.dropout(self.activation(self.bn2(self.fc2(x))))
        x = self.dropout(self.activation(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x

    """
class NNModel(nn.Module):
    def __init__(self, in_dim, n_classes):
        super(NNModel, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, n_classes)
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

super(NNModel, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, n_classes)

        self.act1 = nn.LeakyReLU(0.01)
        self.act2 = nn.LeakyReLU(0.01)
        self.act3 = nn.LeakyReLU(0.01)

        self.dropout = nn.Dropout(0.1)  # Just a touch of regularization

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.dropout(x)
        x = self.act2(self.fc2(x))
        x = self.dropout(x)
        x = self.act3(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)  # No softmax here!
    """

    def train_model(
        self,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        epochs,
        checkpoint_dir=None,
        patience=10,
    ):
        if checkpoint_dir is not None and not os.path.exists(checkpoint_dir):
            print("creating directory")
            os.makedirs(checkpoint_dir)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
            for i, (inputs, labels) in progress_bar:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(
                    outputs, labels.clone().detach().long()
                )  # torch.tensor(labels, dtype=torch.long))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                train_acc = torch.sum(predicted == labels)
                # correct_predictions += (predicted == labels).sum().item()
                accuracy = 100 * train_acc / outputs.size(0)
                progress_bar.set_description(
                    f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / (i+1):.4f}, Acc: {accuracy:.2f}%"
                )

            # Validation Phase
            self.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = self(inputs)
                    loss = criterion(outputs, labels.long())
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            val_acc = 100 * correct / total
            avg_val_loss = val_loss / len(val_loader)

            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss / len(train_loader):.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            scheduler.step(avg_val_loss)
            for param_group in optimizer.param_groups:
                print(f"Current learning rate: {param_group['lr']}")
            """
            # Save checkpoint
            if checkpoint_dir is not None:
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
                torch.save(
                    self.state_dict(),
                    checkpoint_path,
                )
            """
            # Save checkpoint if validation improves
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                if checkpoint_dir is not None:
                    best_model_path = os.path.join(checkpoint_dir, f"best_model.pt")
                    torch.save(self.state_dict(), best_model_path)
            else:
                epochs_no_improve += 1
                print(f"No improvement in val loss for {epochs_no_improve} epoch(s).")

            # Early stopping check
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    def predict(self, data_loader):
        predictions = []
        with torch.no_grad():
            progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
            for i, (inputs, labels) in progress_bar:
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.tolist())
        return predictions

    def print_num_parameters(self):
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of parameters in the model: {num_parameters}")

    def save(self, checkpoint_dir):
        # Save checkpoint
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, f"final_model.pt")
            torch.save(
                self.state_dict(),
                checkpoint_path,
            )


def prepare_data(train_data, labels):
    tensor_x = torch.tensor(train_data, dtype=torch.float32)
    tensor_y = torch.tensor(labels)
    dataset = TensorDataset(tensor_x, tensor_y)
    return dataset


# Dummy model for comparison
class DummyModel:
    def __init__(self):
        self.dummy_clf = DummyClassifier(strategy="stratified", random_state=2987)

    def train(self, train_data, y):
        self.dummy_clf.fit(train_data, y)

    def predict(self, pred_data):
        return self.dummy_clf.predict(pred_data)
