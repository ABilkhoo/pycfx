"""
pycfx/models/variational_autoencoder.py
Variational Autoencoder for use with REVISE
"""

import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from pycfx.models.mlp_pytorch import PyTorchModel
from pycfx.models.latent_encodings import LatentEncoding

#https://mbernste.github.io/posts/vae/
#https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/

class VariationalAutoencoder(PyTorchModel, LatentEncoding):

    class VAE(nn.Module):
        def __init__(
            self, 
            x_dim,
            hidden_dim,
            z_dim=10
            ):
            super(VariationalAutoencoder.VAE, self).__init__()

            # Define autoencoding layers
            self.enc_layer1 = nn.Linear(x_dim, hidden_dim)
            self.enc_layer2_mu = nn.Linear(hidden_dim, z_dim)
            self.enc_layer2_logvar = nn.Linear(hidden_dim, z_dim)

            # Define autoencoding layers
            self.dec_layer1 = nn.Linear(z_dim, hidden_dim)
            self.dec_layer2 = nn.Linear(hidden_dim, x_dim) 

        def encoder(self, x):
            x = F.relu(self.enc_layer1(x))
            mu = F.relu(self.enc_layer2_mu(x))
            logvar = F.relu(self.enc_layer2_logvar(x))
            return mu, logvar

        def reparameterize(self, mu, logvar):
            std = torch.exp(logvar/2)
            eps = torch.randn_like(std)
            z = mu + std * eps
            return z

        def decoder(self, z):
            # Define decoder network
            output = F.relu(self.dec_layer1(z))
            output = F.relu(self.dec_layer2(output))
            return output

        def forward(self, x):
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
            output = self.decoder(z)
            return output, z, mu, logvar
            
    # Define the loss function
    def loss_function(output, x, mu, logvar):
        batch_size = x.size(0)
        recon_loss = F.mse_loss(output, x, reduction='sum') / batch_size
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + 0.002  * kl_loss

    def _build_model(self):
        self.hidden_dim = self.config.get("hidden_dim", 32)
        self.latent_dim = self.config.get("latent_dim", 6)     

        model = VariationalAutoencoder.VAE(x_dim=self.input_properties.n_features, hidden_dim=self.hidden_dim, z_dim=self.latent_dim)
        optimiser = optim.Adam(model.parameters(), lr=self.lr)
        return model, VariationalAutoencoder.loss_function, optimiser

    def train(self, X_train, y_train=None):
        X_train = torch.tensor(X_train).float()

        # Create DataLoader object to generate minibatches
        dataset = torch.utils.data.TensorDataset(X_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Train the model
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch in dataloader:
                # Zero the gradients
                self.optimiser.zero_grad()

                # Get batch
                x = batch[0]
                x = x.to(self.device)

                # Forward pass
                output, z, mu, logvar = self.pytorch_model(x)

                # Calculate loss
                loss = self.loss_fn(output, x, mu, logvar)

                # Backward pass
                loss.backward()

                # Update parameters
                self.optimiser.step()

                # Add batch loss to epoch loss
                epoch_loss += loss.item()

            # Print epoch loss
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(X_train)}")
            
        return self.pytorch_model

    def evaluate(self, X_test, y_test=None):
        model = self.pytorch_model
        X_test = torch.tensor(X_test).float()

        dataset = torch.utils.data.TensorDataset(X_test)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)
    
        model.eval()
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for x in dataloader:  
                x = x[0]
                x = x.to(self.device)

                recon_x, z, mu, logvar = model(x)  

                recon_loss = F.mse_loss(
                    recon_x, x, reduction='sum'
                )

                # KL divergence
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                # Accumulate
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_samples += x.size(0)

        avg_recon_loss = total_recon_loss / total_samples
        avg_kl_loss = total_kl_loss / total_samples
        avg_elbo = -(avg_recon_loss + avg_kl_loss)  # ELBO = - (Recon + KL)

        return avg_recon_loss, avg_kl_loss, avg_elbo
    
    def predict(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype='float')
        
        x = x.to(self.device)
        mu, logvar = self.pytorch_model.encoder(x)
        z = self.pytorch_model.reparameterize(mu, logvar)
        return z
    
    def decode(self, z):
        if not torch.is_tensor(z):
            x = torch.tensor(z, dtype='float')
        
        z = z.to(self.device)

        return self.pytorch_model.decoder(z)

