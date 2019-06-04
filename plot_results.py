import matplotlib.pyplot as plt


prefix = 'discriminator_continuous_v2_'
runs = 1
epochs = range(30)
all_train_losses = []
all_test_losses = []
for i in range(runs):
    train_file = prefix + str(i) + '_train.csv'
    test_file = prefix + str(i) + '_test.csv'
    train_losses = []
    test_losses = []

    for line in open(train_file, 'r'):
        tokens = line.split(',')
        #if len(tokens) == 2:
        train_losses.append(float(tokens[1].strip()))
    for line in open(test_file, 'r'):
        tokens = line.split(',')
        #if len(tokens) == 2:
        test_losses.append(float(tokens[1].strip()))
    
    all_train_losses.append(train_losses)
    all_test_losses.append(test_losses)

for curve in all_train_losses:
    train_curve, = plt.plot(epochs, curve, color='blue', label='Train loss')
for curve in all_test_losses:
    test_curve, = plt.plot(epochs, curve, color='orange', label='Test loss')
plt.legend(handles=[train_curve, test_curve])
plt.xlabel('Epoch #')
plt.ylabel('Loss (binary cross-entropy)')
plt.savefig('discriminator_discrete.png')

