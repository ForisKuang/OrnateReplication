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


class VAEGAN(nn.Module):

    # TODO: This has too many parameters :O
    def train(self, netVAE, netG, netD, true_dataloader, fake_dataloader, optimizerVAE, optimizerG, optimizerD, epoch, loss_file, model_file):
         
        true_iter = iter(true_dataloader)
        fake_iter = iter(fake_dataloader)
        while True:
            true_data = true_iter.next()
            if not true_data:
                break
            fake_data = fake_iter.next()
            if not fake_data:
                break
            true_data = true_data['inputs'].to(device)
            fake_data = fake_data['inputs'].to(device)
            batch_size = fake_data.shape[0]

            ###################################################################
            # Forward pass through the network
            ###################################################################
            # Feed fake data through VAE encoder to get mean/variance vectors
            means, sigmas = netVAE(fake_data)

            # Sample latent vector z_x from a Gaussian with the mean/variance returned by surface VAE
            eps = torch.randn((batch_size, 200)).to(device)
            z_x = means + sigmas * eps

            # Structure produced by generator (decoder) from the initial fake structure
            G_dec = netG(z_x)
            D_dec_fake = netD(G_dec)

            # Now, use the generator to generate a structure from a random
            # latent vector sampled from Normal(0, 1).
            G_train = netG(z)

            # Score the generated structure with the discriminator
            D_fake = netD(G_train)

            # Score the true (native) structure with the discriminator
            D_legit = netD(real_data)

            ###################################################################
            # Compute gradient penalty. Goal is to push the L2-norm of the gradient of the
            # discriminator (at interpolated structures between the generated and real 
            # ones) to be close to 1.
            ###################################################################
            alpha = torch.rand((batch_size, 1)) # Sample from Uniform(0, 1)
            difference = G_train - real_data
            inter = []
            for i in range(batch_size):
                inter.append(difference[i] * alpha[i])
            inter = torch.unbind(inter)

            # TODO: Check if this is the correct conversion from Tensorflow
            slopes = torch.sqrt(torch.square(gradients).sum(axis=1))
            gradient_penalty = ((slopes - 1.)**2).mean()

            #################################################################################
            # Loss calculations
            #################################################################################

            # The KL divergence between N(mu, sigma) and N(0, 1) is:
            # -log(sigma) + 0.5*(sigma^2 + mu^2 - 1)
            # See https://stats.stackexchange.com/questions/406221/understanding-kl-divergence-between-two-univariate-gaussian-distributions
            # for the derivation.

            # NOTE: In this code, "sigma" is actually the LOG standard deviation.
            # The reason we do "exp(2*sigmas)" is because
            # "exp(2 * log(sigma)) = exp(log(sigma^2)) = sigma^2"
            kl_loss = (-sigmas + 0.5*(torch.exp(2.*sigmas) + torch.square(means) - 1.)).mean()

            # Reconstruction error between the real protein structure and our generated structure.
            # This tests the quality of our refinement.
            # QUESTION: is there a mapping between the fake (modeled) and real (native) structures?
            recon_loss = (torch.square(real_data - G_dec)).mean()

            # Discriminator loss. Discriminator should give a high score to real structures
            # (D_legit) and a low score to generated structures (D_fake). Note that here,
            # the generator structures are just generated from any latent vector (sampled
            # from Normal(0, 1), not just starting from our modeled structures.
            d_loss = -D_legit.mean() + D_fake.mean() + 10. * gradient_penalty

            # Generator loss. The generator tries to produce structures that fool the
            # discriminator into giving a high score. It should also try to accurately
            # map modeled (fake) structures to the native (real) ones.
            g_loss = -D_fake.mean() + 5.*recon_loss

            # Autoencoder loss. The autoencoder tries to encode modeled structures to latent
            # vectors and then tries to "reconstruct" a better structure from the latent vector.
            # We also want the latent vectors to have a distribution that's close to Normal(0, 1).
            v_loss = kl_loss + recon_loss


            ################################################################################
            # Optimization
            ################################################################################
            # Compute gradients w.r.t. discriminator, VAE, and reconstruction loss, and update the discriminator & VAE
            optimizerVAE.zero_grad()
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            d_loss.backward()
            v_loss.backward()
            recon_loss.backward()
            optimizerD.step()
            optimizerVAE.step()

            # Every 5th batch, compute gradient w.r.t generator loss and update the generator
            if i % 5 == 0:
                g_loss.backward()
                recon_loss.backward()
                optimizerG.step()

            # Save network to file so that it can be reused
            torch.save({
                'epoch': epoch,
                'VAE_state_dict': netVAE.state_dict(),
                'G_state_dict': netG.state_dict(),
                'D_state_dict': netD.state_dict(),
                'optimizerVAE_state_dict': optimizerVAE.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
            }, model_file)

            # Write losses to file
            with open(loss_file, 'a+') as f:
                f.write(str(epoch) + ', ' + str(kl_loss.item()) + ', ' + str(recon_loss.item()) + ', ' + str(d_loss.item()) + ', ' + str(g_loss.item()) + '\n') 




    def main(self):
        training_runs = 1  # Number of experiments
        num_epochs = 50  # Number of epochs to train the discriminator. Note that the generator is only updated every 5th epoch.

        fraction_test = 0.2
        fraction_validation = 0.2

        model_prefix = 'gan_1'

        # TODO: When model works, drastically increase this
        num_true_files = 5
        num_fake_files = 20

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


        # TODO: All this needs to be redone if it turns out that there
        # is a mapping between the true and fake data

        # Create a dataset for each file (true and fake)
        true_datasets = []
        true_examples = 0
        for true_file in true_files:
            true_dataset = ResidueDataset(true_file, label=1)
            true_datasets.append(true_dataset)
            true_examples += len(true_dataset)

        fake_datasets = []
        fake_examples = 0
        for fake_file in fake_files:
            fake_dataset = ResidueDataset(fake_file, label=0, upper_bound=fake_upper_bound)
            fake_datasets.append(fake_dataset)
            fake_examples += len(fake_dataset)
        print('True examples', true_examples)
        print('Fake examples', fake_examples)

        # Create combined datasets by concatenating the file datasets
        fake_full_dataset = utils_data.ConcatDataset(fake_datasets)
        true_full_dataset = utils_data.ConcatDataset(true_datasets)

        # Split into train/validation/test for real data
        true_validation_size = int(fraction_validation * len(true_full_dataset))
        true_test_size = int(fraction_test * len(true_full_dataset))
        true_train_size = len(true_full_dataset) - true_validation_size - true_test_size
        true_train_dataset, true_validation_dataset, true_test_dataset = torch.utils.data.random_split(true_full_dataset, [true_train_size, true_validation_size, true_test_size])

        fake_validation_size = int(fraction_validation * len(fake_full_dataset))
        fake_test_size = int(fraction_test * len(fake_full_dataset))
        fake_train_size = len(fake_full_dataset) - fake_validation_size - fake_test_size
        fake_train_dataset, fake_validation_dataset, fake_test_dataset = torch.utils.data.random_split(fake_full_dataset, [fake_train_size, fake_validation_size, fake_test_size])


        # Create DataLoaders from the datasets
        true_train_dataloader = utils_data.DataLoader(true_train_dataset, batch_size=32, shuffle=True, num_workers=4)
        fake_train_dataloader = utils_data.DataLoader(fake_train_dataset, batch_size=32, shuffle=True, num_workers=4)

        for run in range(training_runs):
            # Create networks
            netVAE = SurfaceVAE(device=device).to(device)
            netG = Generator().to(device)
            netD = Discriminator(device=device).to(device)
            
            # Create optimizers
            optimizerVAE = optim.Adam(netVAE.parameters(), lr=0.001)
            optimizerG = optim.Adam(netG.parameters(), lr=0.001)
            optimizerD = optim.Adam(netD.parameters(), lr=0.001)
 
            epoch = 0

            loss_file = 'output/gan_loss/' + model_prefix + '_' + str(run) + '_loss.csv'
            with open(loss_file, 'w') as f:
                f.write('epoch, kl_loss, recon_loss, d_loss, g_loss')
            model_file = 'output/model/' + model_prefix + '_' + str(run) + '_model'

            # If there is a partially-trained model already, just load it
            #if os.path.exists(model_file):
                #checkpoint = torch.load(model_file)
                #netVAE.load_state_dict(checkpoint['VAE_state_dict'])
                #netG.load_state_dict(checkpoint['G_state_dict'])
                #netD.load_state_dict(checkpoint['D_state_dict'])

                #optimizerVAE.load_state_dict(checkpoint['optimizerVAE_state_dict'])
                #optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
                #optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
                #epoch = checkpoint['epoch'] + 1
                #loss = checkpoint['loss']

            while epoch < num_epochs:
                self.train(netVAE, netG, netD, true_train_dataloader, fake_train_dataloader, optimizerVAE, optimizerG, optimizerD, epoch, loss_file, model_file)
                #test(net, validation_dataloader, criterion, epoch, test_loss_file)
                epoch += 1     


if __name__ == '__main__':
    VAEGAN().main()
