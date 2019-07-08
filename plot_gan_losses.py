import matplotlib.pyplot as plt
import numpy as np

prefix = 'output/gan_loss/gan_1_'
runs = 1
loss_types = ['KL loss', 'Reconstruction loss', 'Discriminator loss', 'Generator loss']

# 3-D list. First, there is a 2-D list for each loss type (Discriminator loss, Generator loss, etc).
# Then, each 2-D list contains lists for each run of the model; each individual list represents
# the loss at each epoch.
all_losses = {}
for loss_type in loss_types:
    all_losses[loss_type] = []

for i in range(runs):
    train_file = prefix + str(i) + '_loss.csv'
    run_train_losses = {}
    for loss_type in loss_types:
        run_train_losses[loss_type] = []

    i = 0
    for line in open(train_file, 'r'):
        if i == 0:
            i += 1
            continue
        i += 1
        print(i)

        tokens = line.split(',')
        assert len(tokens) == len(loss_types)+1
        for j, loss_type in enumerate(loss_types):
            run_train_losses[loss_type].append(float(tokens[j+1]))
        if i > 1000000:
            break
    for loss_type in run_train_losses:
        all_losses[loss_type].append(run_train_losses[loss_type])

for loss_type in all_losses:
    for curve in all_losses[loss_type]:
        running_mean = np.convolve(curve, np.ones((100,))/100, mode='same')
        print(len(running_mean))
        epochs = range(len(curve))
        train_curve, = plt.plot(epochs, curve, color='blue', label=loss_type)
        running_mean_curve, = plt.plot(epochs, running_mean, color='black', label=loss_type + ' (running average)')
    plt.legend(handles=[train_curve, running_mean_curve])
    plt.xlabel('Batch #')
    plt.ylabel(loss_type)
    plt.savefig(prefix + '_' + loss_type + '.png')
    plt.close()
