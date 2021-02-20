import torch
import torch.nn as nn

class TT():
    def __init__(self):
        super(TT, self).__init__()

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def train(self, model, train_loader, num_epochs, learning_rate, lentrain):

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        model.to(self.device)

        batch_number = len(train_loader)
        # print(batch_number)

        running_loss = 0.0
        running_correct = 0

        print("\nTraining Started\n")
        for epoch in range(num_epochs):

            for i, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                preds = model(images)

                loss = criterion(preds, labels)

                # Backward- compute grads
                optimizer.zero_grad()
                loss.backward()

                # update weights
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(preds.data, 1)
                running_correct += (predicted == labels).sum().item()

                if (i + 1) % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{batch_number}], Loss: {loss.item():.4f}')

            train_loss = running_loss / len(train_loader)
            train_acc = 100 * running_correct / lentrain

            #writer.add_scalar('training loss', train_loss, epoch + 1)
            #writer.add_scalar('training accuracy', train_acc, epoch + 1)

            print("\nLoss and Acc", train_loss, train_acc, running_correct, "\n")

            running_loss = 0.0
            running_correct = 0

            if train_loss < 0.2:
                print("Total Epoch = ", epoch + 1)
                break
        print("\nTraining Finished\n")
        return model


    def test(self, model, test_loader, classes):
        print("\nTest Started\n")
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(len(classes))]
            n_class_samples = [0 for i in range(len(classes))]
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                # max returns (value ,index)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()
                for i in range(len(images)):
                    label = labels[i]
                    pred = predicted[i]
                    if (label == pred):
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network: {acc} %\n')

            for i in range(len(classes)):
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f'Accuracy of {classes[i]}: {acc} %')
        print("\nTest Finished\n")