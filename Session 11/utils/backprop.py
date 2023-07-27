import torch
from tqdm import tqdm
from utils.utils import denormalize
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import get_device 

def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def GetInCorrectPredCount(prediction, labels):
    prediction = prediction.argmax(dim=1)
    indices = prediction.ne(labels).nonzero().reshape(-1).tolist()
    return indices, prediction[indices].tolist(), labels[indices].tolist()

class Train(object):
    def __init__(self, model, train_loader, criterion, optimizer, scheduler=None, perform_step=False, l1=0):
        self.model = model
        self.device = get_device()
        self.criterion = criterion
        self.train_loader =train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.perform_step = perform_step
        self.l1 = l1

        self.train_losses = list()
        self.train_accuracies = list()

    def __call__(self):
        self.model.train()
        pbar = tqdm(self.train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Predict
            pred = self.model(data)

            # Calculate loss
            loss = self.criterion(pred, target)
            if self.l1 > 0:
                loss += self.l1 * sum(p.abs().sum() for p in self.model.parameters())

            train_loss += loss.item() * len(data)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            correct += GetCorrectPredCount(pred, target)
            processed += len(data)

            pbar.set_description(
                desc=f"Train: Average Loss: {train_loss / processed:0.2f}, Accuracy: {100 * correct / processed:0.2f}"
                     + (f" LR: {self.scheduler.get_last_lr()[0]}" if self.perform_step else "")
            )
            if self.perform_step:
                self.scheduler.step()

        train_acc = 100 * correct / processed
        train_loss /= processed
        self.train_accuracies.append(train_acc)
        self.train_losses.append(train_loss)

        return train_loss, train_acc

    def plot_train_stats(self):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(self.train_losses)
        axs[0].set_title("Training Loss")
        axs[1].plot(self.train_accuracies)
        axs[1].set_title("Training Accuracy")


class Test(object):

    def __init__(self,model,test_loader,criterion):

        self.model = model
        self.device = get_device()
        self.test_loader = test_loader
        self.criterion = criterion

        self.test_losses = list()
        self.test_accuracy = list()

    def __call__(self,incorrect_preds=None):

        self.model.eval()

        test_loss = 0
        correct = 0
        processed = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                pred = self.model(data)

                test_loss += self.criterion(pred, target).item() * len(data)

                correct += GetCorrectPredCount(pred, target)
                processed += len(data)

                if incorrect_preds is not None:
                    idx, pred, truth = GetInCorrectPredCount(pred, target)
                    incorrect_preds["images"] += data[idx]
                    incorrect_preds["ground_truths"] += truth
                    incorrect_preds["predicted_vals"] += pred

        test_acc = 100 * correct / processed
        test_loss /= processed
        self.test_accuracy.append(test_acc)
        self.test_losses.append(test_loss)

        print('Test set: Average loss: {:.2f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(self.test_loader.dataset),100. * correct / len(self.test_loader.dataset)))

        return test_loss, test_acc
    
    def plot_test_stats(self):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(self.test_losses)
        axs[0].set_title("Testing Loss")
        axs[1].plot(self.test_accuracy)
        axs[1].set_title("Testing Accuracy")

def get_misclassified_images(model , test_loader , device):
        model.eval()

        images = []
        predictions = []
        labels = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                output = model(data)

                _, preds = torch.max(output, 1)

                for i in range(len(preds)):
                    if preds[i] != target[i]:
                        images.append(data[i])
                        predictions.append(preds[i])
                        labels.append(target[i])

        return images, predictions, labels

def show_misclassified_images(
    images,
    predictions,
    labels,
    classes):
    
    fig = plt.figure(figsize=(20, 10))
    for i in range(len(images)):
        sub = fig.add_subplot(len(images) // 5, 5, i + 1)
        image = images[i]
        npimg = denormalize(image.cpu().numpy().squeeze())
        plt.imshow(npimg, cmap="gray")
        predicted = classes[predictions[i]]
        correct = classes[labels[i]]
        sub.set_title(
            "Correct class: {}\nPredicted class: {}".format(correct, predicted)
        )
    plt.tight_layout()
    plt.show()

def accuracy_classes(model, classes, test_loader):

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


