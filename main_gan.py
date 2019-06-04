




class VAEGAN(nn.Module):
    def __init__(self, device, size):
        self.device = device
        self.size = size



    def main():
        # TODO: Load data
        dataloader = ????
        netVAE = SurfaceVAE().to(self.device)
        netG = Generator().to(self.device)
        netD = Discriminator().to(self.device)




    # TODO: Make this work with DataLoaders. Right now this method
    # only takes in one batch.
    def train(self, netVAE, netG, netD, dataloader):
       
        for i, data in enumerate(dataloader, 0):
            fake_data = batch['fake']
            real_data = batch['real']
            batch_size = fake_data.shape[0]

            # Batch sizes for real/fake data must be equal
            assert fake_data.shape[0] == real_data.shape[0]

            #################################################################################
            # Forward pass through the network
            #################################################################################
            # Feed fake data through VAE encoder to get mean/variance vectors
            means, sigmas = netVAE(fake_data)

            # Sample latent vector z_x from a Gaussian with the mean/variance returned by surface VAE
            eps = torch.randn((batch_size, 200))
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

            #################################################################################
            # Compute gradient penalty. Goal is to push the L2-norm of the gradient of the
            # discriminator (at interpolated structures between the generated and real 
            # ones) to be close to 1.
            #################################################################################
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
            d_loss = = -D_legit.mean() + D_fake.mean() + 10. * gradient_penalty

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
            # TODO Create optimizers on the various losses and actually train them. 5 generator epochs for each discriminator epoch (I think, check the paper)

 
