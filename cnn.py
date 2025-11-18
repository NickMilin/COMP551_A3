import torch
import torch.nn as nn
import time

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # History variables

        self.train_losses_cnn = []
        self.train_accuracies_cnn = []
        self.val_losses_cnn = []
        self.val_accuracies_cnn = []
        self.test_accuracies_cnn = []

        # CNN Variables
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 28x28 -> conv -> pool -> 14x14
        # 14x14 -> conv -> pool -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # -> 32x14x14
        x = self.pool(self.relu(self.conv2(x)))   # -> 64x7x7

        x = x.view(-1, 64 * 7 * 7)             # flatten

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def train_cnn(self, num_epochs, optimizer, loss_fn, train_loader_cnn, val_loader_cnn, test_loader_cnn, device): 
        self.num_epochs = num_epochs
        start_time = time.time()

        total_epochs = len(self.train_losses_cnn) + num_epochs

        for epoch in range(self.num_epochs):
            self.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader_cnn:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()  # Clear gradients from previous batch
                
                outputs = self(images)
                preds = torch.argmax(outputs, dim=1)

                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (preds == labels).sum().item()
                total += len(preds)

            epoch_loss = total_loss / len(train_loader_cnn)
            epoch_acc = correct / total
            self.train_losses_cnn.append(epoch_loss)
            self.train_accuracies_cnn.append(epoch_acc)

            # Validation
            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader_cnn:
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = self(images)
                    preds = torch.argmax(outputs, dim=1)

                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()
                    val_correct += (preds == labels).sum().item()
                    val_total += len(preds)

            val_epoch_loss = val_loss / len(val_loader_cnn)
            val_epoch_acc = val_correct / val_total
            self.val_losses_cnn.append(val_epoch_loss)
            self.val_accuracies_cnn.append(val_epoch_acc)

            # Test accuracy (on the full test set)
            self.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for images, labels in test_loader_cnn:
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = self(images)
                    preds = torch.argmax(outputs, dim=1)

                    test_correct += (preds == labels).sum().item()
                    test_total += len(preds)
            test_acc = test_correct / test_total
            self.test_accuracies_cnn.append(test_acc)

            print(f'Epoch {len(self.train_losses_cnn)}/{total_epochs}: Train loss={epoch_loss:.4f}, Train acc={epoch_acc*100:.2f}%, Val loss={val_epoch_loss:.4f}, Val acc={val_epoch_acc*100:.2f}%, Test acc={test_acc*100:.2f}%')

        self.train_time = time.time() - start_time