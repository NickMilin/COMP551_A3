import torch
import torch.nn as nn
import time

class CustomResNet18(nn.Module):
    def __init__(self, resnet, num_classes=10):
        super(CustomResNet18, self).__init__()
        # Load the pre-trained ResNet18
        self.resnet = resnet

        # Add custom layers
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),  # ResNet18's last feature map has 512 channels
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc_layers(x)
        return x
    
    def train_cnn(self, num_epochs, optimizer, loss_fn, train_loader_cnn, val_loader_cnn, test_loader_cnn, device): 
        self.num_epochs = num_epochs
        self.train_losses_cnn = []
        self.train_accuracies_cnn = []
        self.val_losses_cnn = []
        self.val_accuracies_cnn = []
        self.test_accuracies_cnn = []
        start_time = time.time()

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

            print(f'Epoch {epoch+1}/{self.num_epochs}: Train loss={epoch_loss:.4f}, Train acc={epoch_acc*100:.2f}%, Val loss={val_epoch_loss:.4f}, Val acc={val_epoch_acc*100:.2f}%, Test acc={test_acc*100:.2f}%')

        self.train_time = time.time() - start_time