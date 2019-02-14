import torch.nn as nn
import torch
from torch import optim
from models import *
from data_loader import load_dataset
import torch.utils.data as utils_data
import numpy as np

cuda = torch.device('cuda')

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
        inputs = torch.IntTensor(inputs)
        inputs = inputs.to(cuda)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(cuda)

        # forward + backward + optimize
        outputs = net(inputs, labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        running_loss += loss.item()
        if (i + 1) % 2000 == 0:
            print(running_loss/2000)
            running_loss = 0.0
    print("The total loss is " + str(total_loss/i))

def test(net, dataloader):
	correct = 0
	total = 0
	dataTestLoader = dataloader
	with torch.no_grad():
		for data in dataTestLoader:
			inputs, labels = data
			inputs = inputs.type(torch.FloatTensor)
			inputs = inputs.to('cuda')
			labels = labels.type(torch.LongTensor)
			labels = labels.to('cuda')
			outputs = net(inputs)
			values, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			for i in range(labels.size(0)):
				x = predicted[i][0].item()
				y = labels[i][0].item()
				if (x == y):
					correct += 1

def main():
	num_epochs = 100

# Load train data and input as TensorDatset
	trainX, trainY, idx2map2d = load_dataset('../prot3d_100')
	trainNumpyX = np.asarray(trainX)
	trainNumpyY = np.asarray(trainY)

	tensor_train_x = torch.from_numpy(trainNumpyX)
	tensor_train_y = torch.from_numpy(trainNumpyY)

	train_dataset = utils_data.TensorDataset(tensor_train_x, tensor_train_y)

# Load test data and input as TensorDatset
	#testX, testY, test_idx2map2d = load_dataset('test_filename')
	#testNumpyX = np.asarray(testX)
	#testNumpyY = np.asarray(testY)

	#tensor_test_x = torch.from_numpy(testNumpyX)
	#tensor_test_y = torch.from_numpy(testNumpyY)

	#test_dataset = utils_data.TensorDataset(tensor_test_x, tensor_test_y)

	train_dataloader = utils_data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

	#test_dataloader = utils_data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4)

# Actually train the model
	net = OrnateReplicaModel(15)
	criterion = nn.MSELoss()
	optimizer = optim.Adam(net.parameters(), lr=0.01)
	for epoch in range(num_epochs):
		train(net, train_dataloader, idx2map2d, optimizer, criterion, epoch)
		test(net, train_dataloader)
	#test(net, test_dataloader)

if __name__=='__main__':
	main()
