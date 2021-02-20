import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import matplotlib.pyplot as plt
import numpy as np
import torchvision


class LoadData():

    def __init__(self, data_dir, batch_size):
        super(LoadData, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.classes =  os.listdir(self.data_dir + "/train")
        print(self.classes)

    def load(self):
        self.train_dataset = ImageFolder(self.data_dir + '/train', transform=tt.Compose([tt.ToTensor(),
                                                                                         tt.Resize(256),
                                                                                         tt.CenterCrop(250)]))

        self.test_dataset = ImageFolder(self.data_dir + '/test', transform=tt.Compose([tt.ToTensor(),
                                                                                         tt.Resize(256),
                                                                                         tt.CenterCrop(250)]))

        lentrain = len(self.train_dataset)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size,shuffle=True)

        return self.train_loader, self.test_loader, lentrain

    def showBatch(self):
        imgs, labels = next(iter(self.train_loader))
        imgs = torchvision.utils.make_grid(imgs)

        npimg = imgs.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

        title = [self.classes[i.item()] for i in labels]

        plt.title(title)
        plt.show()

    def get_classes(self):
        return self.classes

