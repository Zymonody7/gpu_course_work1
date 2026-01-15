# program1_scnn_train.py

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.backends.cudnn as cudnn


from spikingjelly.activation_based import neuron, functional, layer


LEARNING_RATE = 2e-3
BATCH_SIZE = 512
EPOCHS = 150
T_TIMESTEPS = 2
TARGET_ACC = 0.8978

# Network width configuration (kept compatible with inference)
CONV1_OUT = 12
CONV2_OUT = 24
FC1_OUT = 240
FC2_OUT = 120


class SCNN(nn.Module):
    def __init__(self, T: int):
        super(SCNN, self).__init__()
        self.T = T

        self.conv1 = layer.Conv2d(1, CONV1_OUT, 5)
        self.if1 = neuron.IFNode()
        self.pool1 = layer.MaxPool2d(2, 2)

        self.conv2 = layer.Conv2d(CONV1_OUT, CONV2_OUT, 5)
        self.if2 = neuron.IFNode()
        self.pool2 = layer.MaxPool2d(2, 2)

        self.flatten = layer.Flatten()

        self.fc1 = layer.Linear(CONV2_OUT * 4 * 4, FC1_OUT)
        self.if3 = neuron.IFNode()

        self.fc2 = layer.Linear(FC1_OUT, FC2_OUT)
        self.if4 = neuron.IFNode()

        self.fc3 = layer.Linear(FC2_OUT, 10)

    def forward(self, x: torch.Tensor):
        outputs = []
        for t in range(self.T):
            y = self.conv1(x)
            y = self.if1(y)
            y = self.pool1(y)
            y = self.conv2(y)
            y = self.if2(y)
            y = self.pool2(y)
            y = self.flatten(y)
            y = self.fc1(y)
            y = self.if3(y)
            y = self.fc2(y)
            y = self.if4(y)
            y = self.fc3(y)
            outputs.append(y)
        
        outputs = torch.stack(outputs, dim=0)
        return outputs.mean(0)

script_dir = os.path.dirname(os.path.abspath(__file__))


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
trainset = torchvision.datasets.FashionMNIST(data_dir, download=True, train=True, transform=train_transform)
testset = torchvision.datasets.FashionMNIST(data_dir, download=True, train=False, transform=test_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True
model = SCNN(T=T_TIMESTEPS).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))


print("--- Starting SCNN Training (Tuned for Convergence) ---")
max_accuracy = 0.0
for epoch in range(EPOCHS):

    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        functional.reset_net(model)

        with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(trainloader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
    scheduler.step()


    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            functional.reset_net(model)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f} %')
    

    if accuracy > max_accuracy:
        max_accuracy = accuracy
        print(f'New best accuracy: {max_accuracy:.2f} %. Saving model parameters...')
        output_dir = os.path.join(script_dir, 'weights_tuned')
        os.makedirs(output_dir, exist_ok=True)
        for name, param in model.named_parameters():
            np.savetxt(os.path.join(output_dir, f'{name}.txt'), param.detach().cpu().numpy().flatten())
        if max_accuracy >= TARGET_ACC:
            print(f"Target accuracy {TARGET_ACC:.2f}% reached. Stopping early.")
            break


print('--- Finished Training ---')
print(f'Best accuracy achieved: {max_accuracy:.2f} %')
print("--- Final model parameters have been exported. ---")
