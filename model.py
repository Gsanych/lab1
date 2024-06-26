import json
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collection import load_data

from config import settings
from loguru import logger


def get_model():
    # model = models.resnet50(pretrained=False, num_classes=100)
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)


    return model

def build_model():     
    #1. load dataset
    train_loader, val_loader, test_loader = load_data(settings.batch_size)
    model = get_model()   
    # model = net

    #4. train the model
    best_accuracy = train_model(model, train_loader, val_loader)
    
    save_model(model, f"{settings.local_model_path}/{settings.local_model_name}")
    
    #5. evaluate the model
    test_accuracy = evaluate_model(model, test_loader)

    metrics = {
        'best_validation_accuracy': best_accuracy,
        'test_accuracy': test_accuracy,
    }

    
    metrics_file = os.path.join(settings.local_model_path, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f)

    return model


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader):

    logger.info(f'batch_size {settings.batch_size}')
    logger.info(f'SGD + Learning rate {settings.learning_rate}')

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=settings.learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=settings.learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_accuracy = 0.0
    total_train = 0
    correct_train = 0
    for epoch in range(settings.epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            if i % 200 == 199:  # Print every 200 mini-batches
                logger.info(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

        train_accuracy = 100 * correct_train / total_train
        logger.info(f'[Epoch {epoch + 1}] Training Accuracy: {train_accuracy:.2f}%')

        scheduler.step() 

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        logger.info(f'Validation loss after epoch {epoch + 1}: {val_loss / len(val_loader):.3f}, Accuracy: {val_accuracy:.2f}%')

        # Check if this is the best accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_wts = model.state_dict().copy()
            best_epoch = epoch
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Early stopping
        if no_improvement_count >= settings.patience:
            logger.info(f'Early stopping at epoch {epoch + 1}')
            break

    # Load the best model weights
    model.load_state_dict(best_model_wts)

    logger.info('Finished Training')
    logger.info(f'Best validation accuracy: {best_accuracy:.2f}% at epoch {best_epoch + 1}')

    return best_accuracy

def evaluate_model(model: nn.Module, testloader: DataLoader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    logger.info(f'Accuracy of the model on the test images: {test_accuracy:.2f}%')

    return test_accuracy

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path):
    torch.nn.Module._load_from_state_dict = torch.nn.Module._load_from_state_dict_inner

    model = torch.load(path, map_location=torch.device('cpu'), pickle_module=torch)

    return model
    