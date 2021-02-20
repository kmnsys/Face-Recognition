import torch
from network import ConvNetwork
from dataloader import LoadData
from train_test import TT

# Training parameters
num_epochs = 50
batch_size = 32
learning_rate = 0.01

# Load Data
data_dir = 'D:/data2'
data = LoadData(data_dir, batch_size)
train_loader, test_loader, lendata = data.load()

# Plotting
data.showBatch()

# Network
network = ConvNetwork()
print(network)

# Train and get model
model = TT().train(network, train_loader, num_epochs, learning_rate, lendata)

# Save Model
PATH = 'model/face_model_original.pth'
torch.save(model.state_dict(), PATH)

# Test
classes = data.get_classes()
TT().test(model, test_loader, classes)
