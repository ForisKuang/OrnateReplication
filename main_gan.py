import torch.nn as nn
import torch
from torch import optim
from models import *
from file_list_utils import produce_shuffled_file_list, read_file_list
from residue_dataset import ResidueDataset
from ShapeNets_dataset import ShapeNetsDataset
import torch.utils.data as utils_data
import glob
import numpy as np
import os
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE IS {0}".format(str(device)))


class VAEGAN(nn.Module):

    # TODO: This has too many parameters :O
    def train(self, netVAE, netG, netD, dataloader, optimizerVAE, optimizerG, optimizerD, epoch, loss_file, model_file, generated_output_dir):
        iterator = iter(dataloader)
        while True:
            try:
                data = iterator.next()
                if not data:
                    break
            except StopIteration:
                break
            real_data = data['real_data'].to(device)
            fake_data = data['fake_data'].to(device)
            assert(fake_data.shape[0] == real_data.shape[0])
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
            z = torch.randn((batch_size, 200)).to(device)
            G_train = netG(z)

            # Score the generated structure with the discriminator
            D_fake = netD(G_train)

            # Score the real (native) structure with the discriminator
            D_legit = netD(real_data)

            ###################################################################
            # Compute gradient penalty. Goal is to push the L2-norm of the gradient of the
            # discriminator (at interpolated structures between the generated and real
            # ones) to be close to 1.
            ###################################################################
            alpha = torch.rand((batch_size, 1)).to(device) # Sample from Uniform(0, 1)
            difference = G_train - real_data
            inter = []
            for i in range(batch_size):
                inter.append(difference[i] * alpha[i])
            inter = torch.stack(inter)
            interpolates = real_data + inter

            # TODO: Check if this is the correct conversion from Tensorflow
            gradients = torch.autograd.grad(netD(interpolates).mean(), interpolates)[0]
            print('gradient shape', gradients.shape)
            slopes = torch.sqrt((gradients ** 2).sum())
            gradient_penalty = ((slopes - 1.)**2)

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
            kl_loss = (-sigmas + 0.5*(torch.exp(2.*sigmas) + means**2 - 1.)).mean()

            # Reconstruction error between the real protein structure and our generated structure.
            # This tests the quality of our refinement.
            # QUESTION: is there a mapping between the fake (modeled) and real (native) structures?
            recon_loss = ((real_data - G_dec) ** 2).mean()

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
            d_loss.backward(retain_graph=True)
            v_loss.backward(retain_graph=True)

            optimizerD.step()
            optimizerVAE.step()

            # Every 5th batch, compute gradient w.r.t generator loss and update the generator
            if i % 5 == 0:
                recon_loss.backward(retain_graph=True)
                g_loss.backward()
                optimizerD.step()
                optimizerG.step()
                optimizerVAE.step()
            else:
                recon_loss.backward()
                optimizerD.step()
                optimizerVAE.step()

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
                f.write(str(epoch) + ', ' + str(kl_loss.item()) + ', ' + str(recon_loss.item()) + ', ' + str(d_loss.item()) + ', ' + str(g_loss.item()) + ', ' + str(gradient_penalty.item()) + ', ' + str(D_fake.mean().item()) + ', ' + str(D_legit.mean().item()) + '\n')

        # Write example generated outputs to file
        for i in range(G_dec.shape[0]):
            file_path = os.path.join(generated_output_dir, 'epoch_' + str(epoch) + '_gen_decoded_' + str(i) + '.npy')
            np.save(file_path, G_dec[i].cpu().detach())
        for i in range(G_train.shape[0]):
            file_path = os.path.join(generated_output_dir, 'epoch_' + str(epoch) + '_gen_random_' + str(i) + '.npy')
            np.save(file_path, G_train[i].cpu().detach())


    def main(self):
        # Create necessary output directories if they don't exist already
        output_dirs = ['output/models', 'output/gan_loss', 'output/file_lists', 'output/generated']
        for output_dir in output_dirs:
            os.makedirs(output_dir, exist_ok=True)

        training_runs = 1  # Number of experiments
        num_epochs = 1000  # Number of epochs to train the discriminator. Note that the generator is only updated every 5th epoch.

        fraction_test = 0.
        fraction_validation = 0.2

        model_prefix = 'gan_1'

        # TODO: When model works, drastically increase this
        num_real_files = 20000

        # For fake structures, only include structures whose quality score is LESS than
        # this number
        fake_upper_bound = 0.5

        # Get lists of files
        real_file_list_file = 'output/file_lists/real_chairs.txt'
        fake_file_list_file = 'output/file_lists/fake_chairs.txt'

        """
        Load data of protein residues

        if os.path.exists(real_file_list_file):
            real_files = read_file_list(real_file_list_file)
        else:
            real_files = produce_shuffled_file_list('/net/scratch/aivan/decoys/ornate/pkl.natives', real_file_list_file)
        if os.path.exists(fake_file_list_file):
            fake_files = read_file_list(fake_file_list_file)
        else:
            fake_files = produce_shuffled_file_list('/net/scratch/aivan/decoys/ornate/pkl.rand70', fake_file_list_file)
        real_files = real_files[:num_real_files]
        fake_files = fake_files[:num_fake_files]
        """


        if os.path.exists(fake_file_list_file):
            fake_files = read_file_list(fake_file_list_file)
        else:
            fake_files = produce_shuffled_file_list('/home/forisk/3D-IWGAN/3D-Reconstruction-Kinect/data/surfaces/train/chair/', fake_file_list_file)

        real_files = []
        for fake_file in fake_files:
            real_filename = '/' + fake_file.split('/')[-1].split('_')[-2] + '.npy'
            real_files.append('/home/forisk/3D-IWGAN/3D-Reconstruction-Kinect/data/train/chair' + real_filename)

        #real_files = real_files[:num_real_files]
        #fake_files = fake_files[:num_real_files]
        print('Real files', real_files[0:10])
        print('Fake files', fake_files[0:10])

        # Create a dataset for each pair of real/fake files
        datasets = []
        for i in range(len(real_files)):
            dataset = ShapeNetsDataset(real_files[i], fake_files[i])
            datasets.append(dataset)
        print('Num real examples = num fake examples = ', len(datasets))

        # Create combined datasets by concatenating the file datasets
        full_dataset = utils_data.ConcatDataset(datasets)

        # Split into train/validation/test for real data
        validation_size = int(fraction_validation * len(full_dataset))
        test_size = int(fraction_test * len(full_dataset))
        train_size = len(full_dataset) - validation_size - test_size
        train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, validation_size, test_size])

        # Create DataLoaders from the datasets
        train_dataloader = utils_data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

        for run in range(training_runs):
            # Create networks
            netVAE = SurfaceVAE(device=device, num_retype=1).to(device)
            netG = Generator().to(device)
            netD = Discriminator(device=device, num_retype=1).to(device)

            # Create optimizers
            optimizerVAE = optim.Adam(netVAE.parameters(), lr=1e-4)
            optimizerG = optim.Adam(netG.parameters(), lr=1e-4)
            optimizerD = optim.Adam(netD.parameters(), lr=1e-5)

            epoch = 0

            loss_file = 'output/gan_loss/' + model_prefix + '_' + str(run) + '_loss.csv'
            with open(loss_file, 'w') as f:
                f.write('epoch, kl_loss, recon_loss, d_loss, g_loss, gradient penalty, D_fake, D_legit\n')
            model_file = 'output/models/' + model_prefix + '_' + str(run) + '_model'
            generated_output_dir = 'output/generated/'

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
                print('Epoch', epoch)
                self.train(netVAE, netG, netD, train_dataloader, optimizerVAE, optimizerG, optimizerD, epoch, loss_file, model_file, generated_output_dir)
                #test(net, validation_dataloader, criterion, epoch, test_loss_file)
                epoch += 1



if __name__ == '__main__':
    VAEGAN().main()
