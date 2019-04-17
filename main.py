import torch.nn as nn
import torch
from torch import optim
from models import *
from data_loader import load_dataset
import torch.utils.data as utils_data
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(net, dataloader, idx2map2d, optimizer, criterion, epoch):
    
    total_loss = 0.0
    running_loss = 0.0

    for i, data in enumerate(dataloader, 0):
        indices, labels = data
        inputs = []
        for idx in range(len(indices)):
            map2d = idx2map2d[indices.data[idx].item()]
            map4d = np.zeros((24, 24, 24, 167))
            map4d[map2d[0,:],map2d[1,:],map2d[2,:],map2d[3,:]] = map2d[4,:]
            inputs.append(map4d)
        inputs = torch.FloatTensor(inputs)
        inputs = inputs.to(device)
        labels = labels.type(torch.FloatTensor)
        labels = labels.to(device)

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        running_loss += loss.item()
        if (i + 1) % 2 == 0:
            print(running_loss/2)
            running_loss = 0.0
    print("The total loss is " + str(total_loss/i))

def test(net, dataloader, idx2map2d):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            indices, labels = data
            inputs = []
            for idx in range(len(indices)):
                map2d = idx2map2d[indices.data[idx].item()]
                map4d = np.zeros((24, 24, 24, 167))
                map4d[map2d[0,:],map2d[1,:],map2d[2,:],map2d[3,:]] = map2d[4,:]
                inputs.append(map4d)
            inputs = torch.FloatTensor(inputs)
            inputs = inputs.to(device)
            labels = labels.type(torch.FloatTensor)
            labels = labels.to(device)
            outputs = net(inputs)
            values, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            for i in range(labels.size(0)):
                x = predicted[i].item()
                y = labels[i].item()
                if (x == y):
                    correct += 1
    print('Correct: ' + str(correct))
    print('Total: ' + str(total))


def main():
    num_epochs = 10
    fraction_test = 0.2
    fraction_validation = 0.2

    # Load true protein structures and fake modelled structures as TensorDataset
    trueX, trueY, true_idx2map2d = load_dataset('/net/scratch/aivan/decoys/ornate/pkl.natives')
    modelX, modelY, model_idx2map2d = load_dataset('/net/scratch/aivan/decoys/ornate/pkl.rand70', starting_index=len(trueX))    

    # For fake modeled structures, make the label 0
    modelY = np.zeros_like(modelX)
    for y in trueY:
        assert y == 1
    for y in modelY:
        assert y == 0

    # Concatenate true structures and fake ones, and shuffle them
    # to train the discriminator
    numpyX = np.asarray(np.concatenate((trueX, modelX)))
    numpyY = np.asarray(np.concatenate((trueY, modelY))).reshape(-1, 1)
    np.random.shuffle(numpyX)
    np.random.shuffle(numpyY)

    # Also concatenate the two dictionaries of (true, modelled) structures
    # (each structure is stored in the 2D representation here)
    idx2map2d = {**true_idx2map2d, **model_idx2map2d}

    # Split data into training/validation/test
    validation_start = int((1 - fraction_test - fraction_validation) * len(numpyX))
    test_start = int((1 - fraction_test) * len(numpyX))
    tensor_train_x = torch.from_numpy(numpyX[:validation_start])
    tensor_train_y = torch.from_numpy(numpyY[:validation_start])
    tensor_validation_x = torch.from_numpy(numpyX[validation_start:test_start])
    tensor_validation_y = torch.from_numpy(numpyY[validation_start:test_start])
    tensor_test_x = torch.from_numpy(numpyX[test_start:])
    tensor_test_y = torch.from_numpy(numpyY[test_start:])

    # Create TensorDatasets
    train_dataset = utils_data.TensorDataset(tensor_train_x, tensor_train_y)
    validation_dataset = utils_data.TensorDataset(tensor_validation_x, tensor_validation_y)
    test_dataset = utils_data.TensorDataset(tensor_test_x, tensor_test_y)
    
    # Create DataLoaders to handle minibatching
    train_dataloader = utils_data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    validation_dataloader = utils_data.DataLoader(validation_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_dataloader = utils_data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

    # Actually train the model
    net = OrnateReplicaModel(15)

    # Binary cross-entropy loss for binary classification (is the structure real or not?)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        train(net, train_dataloader, idx2map2d, optimizer, criterion, epoch)
        test(net, validation_dataloader, idx2map2d)

if __name__=='__main__':
    main()
