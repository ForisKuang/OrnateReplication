import torch.nn as nn
import torch
from torch import optim
from models import *
from file_list_utils import produce_shuffled_file_list, read_file_list
from residue_dataset import ResidueDataset
import torch.utils.data as utils_data
import glob
import numpy as np
import os
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE IS {0}".format(str(device)))

def train(net, dataloader, optimizer, criterion, epoch, train_loss_file, model_file):

    total_loss = 0.0
    running_loss = 0.0
    num_total = 0.
    num_correct = 0.

    for i, batch in enumerate(dataloader, 0):
        inputs = batch['inputs']
        labels = batch['labels']
        optimizer.zero_grad()
        inputs = torch.FloatTensor(inputs).to(device=device)
        labels = torch.FloatTensor(labels).to(device=device)

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        for j in range(labels.size(0)):
            x = outputs[j].item()
            y = labels[j].item()
            print('TRAINING Label: {0:.3f}, predicted: {1:.3f}'.format(y, x))
            num_total += 1
            if round(x) == round(y):
                num_correct += 1
        total_loss += loss.item()
        running_loss += loss.item()
        if (i + 1) % 2 == 0:
            print("Minibatch number: {0}".format(i))
            print("Current loss: {0}".format(running_loss/2))
            running_loss = 0.0

    # Save network to file so that it can be reused
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss / (i+1),
    }, model_file)

    with open(train_loss_file, 'a+') as f:
        f.write(str(epoch) + ', ' + str(total_loss/(i+1)) + ', ' + str(num_correct / num_total) + '\n')
        print("EPOCH " + str(epoch) + ": total TRAINING loss is " + str(total_loss/(i+1)))
        print("Accuracy", str(num_correct / num_total))


def test(net, dataloader, criterion, epoch, test_loss_file):
    error = 0.
    total_loss = 0.
    num_total = 0.
    num_correct = 0.
    with torch.no_grad():
        for i, batch in enumerate(dataloader, 0):
            inputs = batch['inputs']
            labels = batch['labels']
            inputs = torch.FloatTensor(inputs).to(device=device) 
            labels = torch.FloatTensor(labels).to(device=device)
            outputs = net(inputs)
            test_loss = criterion(outputs, labels)
            total_loss += test_loss.item()

            for j in range(labels.size(0)):
                x = outputs[j].item()
                y = labels[j].item()
                print('Test Label: {0:.3f}, predicted: {1:.3f}'.format(y, x))
                error += abs(x - y)
                num_total += 1
                if round(x) == round(y):
                    num_correct += 1

    print('Average absolute-value error: ' + str(error / num_total))
    print('Test loss: ' + str(total_loss / (i+1)))
    with open(test_loss_file, 'a+') as f:
        f.write(str(epoch) + ', ' + str(total_loss/(i+1)) + ',' + str(num_correct / num_total) + '\n')
        print("EPOCH " + str(epoch) + ": total TEST loss is " + str(total_loss/(i+1)))
        print("Accuracy", str(num_correct / num_total))


def main():
    # Create necessary output directories if they don't exist already 
    output_dirs = ['output/models', 'output/discriminator_loss', 'output/file_lists']
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)



    training_runs = 10  # Number of experiments
    num_epochs = 30
    fraction_test = 0.2
    fraction_validation = 0.2

    model_prefix = 'discriminator_discrete_v2'
    num_true_files = 500
    num_fake_files = 2000

    # True if we want to do binary classification. False if we want
    # to directly predict the continuous score.
    discrete = True

    # For fake structures, only include structures whose quality score is LESS than
    # this number
    fake_upper_bound = 0.5

    # Get lists of files
    true_file_list_file = 'output/file_lists/true_files.txt'
    fake_file_list_file = 'output/file_lists/fake_files.txt'
    if os.path.exists(true_file_list_file):
        true_files = read_file_list(true_file_list_file)
    else:
        true_files = produce_shuffled_file_list('/net/scratch/aivan/decoys/ornate/pkl.natives', true_file_list_file)
    if os.path.exists(fake_file_list_file):
        fake_files = read_file_list(fake_file_list_file)
    else:
        fake_files = produce_shuffled_file_list('/net/scratch/aivan/decoys/ornate/pkl.rand70', fake_file_list_file)  
    true_files = true_files[:num_true_files]
    fake_files = fake_files[:num_fake_files]

    # Create a dataset for each file (true and fake)
    all_datasets = []
    true_examples = 0
    for true_file in true_files:
        true_dataset = ResidueDataset(true_file, label=1)
        all_datasets.append(true_dataset)
        true_examples += len(true_dataset)
    fake_examples = 0
    for fake_file in fake_files:
        fake_dataset = ResidueDataset(fake_file, label=0, upper_bound=fake_upper_bound)
        all_datasets.append(fake_dataset)
        fake_examples += len(fake_dataset)
    print('True examples', true_examples)
    print('Fake examples', fake_examples)

    # Create a combined dataset by concatenating the file datasets
    full_dataset = utils_data.ConcatDataset(all_datasets)

    # Split into train/validation/test
    validation_size = int(fraction_validation * len(full_dataset))
    test_size = int(fraction_test * len(full_dataset))
    train_size = len(full_dataset) - validation_size - test_size
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, validation_size, test_size])

    # Create DataLoaders from the datasets
    train_dataloader = utils_data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    validation_dataloader = utils_data.DataLoader(validation_dataset, batch_size=32, shuffle=True, num_workers=4)


    for run in range(training_runs):
        # Actually train the model
        net = Discriminator(15, device=device).to(device)
        epoch = 0

        train_loss_file = 'output/discriminator_loss/' + model_prefix + '_' + str(run) + '_train.csv'
        test_loss_file = 'output/discriminator_loss/' + model_prefix + '_' + str(run) + '_test.csv'
        model_file = 'output/models/' + model_prefix + '_' + str(run) + '_model'

        # If we're doing (discrete) binary classification, use Binary cross-entropy loss.
        # otherwise, use mean squared error or Huber loss (similar to mean squared error, but penalize outliers less)
        if discrete:
            criterion = nn.BCELoss()
        else:
            criterion = nn.MSELoss()
            #criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        # If there is a partially-trained model already, just load it
        if os.path.exists(model_file):
            checkpoint = torch.load(model_file)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch'] + 1
            loss = checkpoint['loss']

        while epoch < num_epochs:
            net.train()
            train(net, train_dataloader, optimizer, criterion, epoch, train_loss_file, model_file)
            # NOTE: Is this needed?
            #net.eval()
            test(net, validation_dataloader, criterion, epoch, test_loss_file)
            epoch += 1

if __name__=='__main__':
    main()




"""
OLD CODE. IGNORE EVERYTHING BELOW THIS.

        #indices, labels = data
        #inputs = []

        #for idx in range(len(indices)):
            #map2d = idx2map2d[indices.data[idx].item()]
            #print('Map2D:', map2d)

            #assert not np.isnan(map2d).any()
            #map4d = np.zeros((24, 24, 24, 167))
            #map4d[map2d[0,:],map2d[1,:],map2d[2,:],map2d[3,:]] = map2d[4,:]
            #inputs.append(map4d)

            #indices, labels = data
            #inputs = []
            #for idx in range(len(indices)):
                #map2d = idx2map2d[indices.data[idx].item()]
                #map4d = np.zeros((24, 24, 24, 167))
                #map4d[map2d[0,:],map2d[1,:],map2d[2,:],map2d[3,:]] = map2d[4,:]
                #inputs.append(map4d)

    # Load true protein structures and fake modelled structures as TensorDataset
    trueX, trueY, true_idx2map2d = load_dataset('/net/scratch/aivan/decoys/ornate/pkl.natives', num_files=true_files)
    fakeX, fakeY, fake_idx2map2d = load_dataset('/net/scratch/aivan/decoys/ornate/pkl.rand70', num_files=fake_files, starting_index=len(trueX))    

    # If discrete, for the fake/modeled structures, make the label 0
    if discrete:
        fakeY = np.zeros_like(fakeX)

    # Concatenate true structures and fake ones, and shuffle them
    # to train the discriminator
    numpyX = np.asarray(np.concatenate((trueX, fakeX)))
    numpyY = np.asarray(np.concatenate((trueY, fakeY))).reshape(-1, 1)

    # Shuffle
    indices = np.arange(numpyX.shape[0])
    np.random.shuffle(indices)
    numpyX = numpyX[indices]
    numpyY = numpyY[indices]

    # Also concatenate the two dictionaries of (true, modelled) structures
    # (each structure is stored in the 2D representation here)
    idx2map2d = {**true_idx2map2d, **fake_idx2map2d}

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
    train_dataloader = utils_data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    validation_dataloader = utils_data.DataLoader(validation_dataset, batch_size=64, shuffle=True, num_workers=8)
    test_dataloader = utils_data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=8)
    """


