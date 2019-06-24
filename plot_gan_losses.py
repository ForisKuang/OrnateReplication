import matplotlib.pyplot as plt


prefix = 'discriminator_continuous_v2_'
runs = 1
epochs = range(30)
loss_types = ['KL loss', 'Reconstruction loss', 'Generator loss', 'Discriminator loss']

# 3-D list. First, there is a 2-D list for each loss type (Discriminator loss, Generator loss, etc).
# Then, each 2-D list contains lists for each run of the model; each individual list represents
# the loss at each epoch.
all_losses = {}
for loss_type in loss_types:
    all_losses[loss_type] = []

for i in range(runs):
    train_file = prefix + str(i) + '_train.csv'
    run_train_losses = {}
    for loss_type in loss_types:
        run_train_losses[loss_type] = []

    for line in open(train_file, 'r'):
        tokens = line.split(',')
        assert len(tokens) == len(loss_types)+1
        for j, loss_type in enumerate(loss_types):
            run_train_losses[loss_type].append(tokens[j+1])
    
    for loss_type in run_train_losses:
        all_losses[loss_type].append(run_train_losses[loss_type])

for loss_type in all_train_losses:
    for curve in all_train_losses[loss_types]:
        train_curve, = plt.plot(epochs, curve, color='blue', label='Train loss')
    plt.legend(handles=[train_curve, test_curve])
    plt.xlabel('Epoch #')
    plt.ylabel(loss_type)
    plt.savefig(prefix + '_' + loss_type + '.png')

