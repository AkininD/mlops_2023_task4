import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from clearml import Task


class Net(nn.Module):
    def __init__(self, lin1_size, lin2_size, p_dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p_dropout)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 53 * 53, out_features=lin1_size)
        self.fc2 = nn.Linear(in_features=lin1_size, out_features=lin2_size)
        self.fc3 = nn.Linear(in_features=lin2_size, out_features=3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train(task):
    logger = task.get_logger()

    config = {
        'batch_size': 8,
        'lr': 0.01,
        'lin1_size': 120,
        'lin2_size': 64,
        'p_dropout': 0.1,
    }

    config = task.connect(config)

    net = Net(
        config['lin1_size'],
        config['lin2_size'],
        config['p_dropout']
    )

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, min_lr=0.001)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder('data/train', transform=transform)
    val_dataset = datasets.ImageFolder('data/val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    for epoch in range(10):
        running_loss = 0.0
        epoch_steps = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_steps += 1
        train_loss_average = running_loss / epoch_steps

        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for inputs, labels in val_loader:
            with torch.no_grad():
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        val_loss_average = val_loss / val_steps
        scheduler.step(val_loss_average)
        accuracy = correct / total

        print(
            f'{epoch + 1},'
            f'train loss: {round(val_loss_average, 3)},',
            f'val loss: {round(val_loss_average, 3)},',
            f'acc: {round(accuracy, 3)}',
        )

        logger.report_scalar(title='Training Loss', series='train_loss', value=train_loss_average, iteration=epoch)
        logger.report_scalar(title='Validation Loss', series='train_loss', value=val_loss_average, iteration=epoch)
        logger.report_scalar(title='Validation Accuracy', series='accuracy', value=correct / total, iteration=epoch)
    print('Finished Training')


task = Task.init(project_name='HPO', task_name='HP optimization base')
train(task)
