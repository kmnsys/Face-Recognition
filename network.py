import torch.nn as nn
import torch.nn.functional as F

class ConvNetwork(nn.Module):
    def __init__(self):
        super(ConvNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*59*59, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 38) # 38 represents number of people. Change this according to dataset you are using.


    def forward(self, image):
        image = self.pool(F.relu(self.conv1(image)))
        image = self.pool(F.relu(self.conv2(image)))
        image = image.view(-1, 16 * 59 * 59)
        image = F.relu(self.fc1(image))
        image = F.relu(self.fc2(image))
        image = self.fc3(image)
        return image
