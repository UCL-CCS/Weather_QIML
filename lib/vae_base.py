import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import operator


def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega


# A VAE neural network
class EncoderNet(nn.Module):
    def __init__(self, layers, nonlinearity , variational = False):
        super(EncoderNet, self).__init__()
        
        self.hidden = nn.ModuleList()
        self.n_layers = len(layers) - 1
        self.variational = variational
        assert self.n_layers >= 1
        for k in range(self.n_layers):
            if k < self.n_layers - 1:
                self.hidden.append(nn.Linear(layers[k], layers[k+1]))
                self.hidden.append(nonlinearity())
            elif variational: # if variational, make nets for mu and logvar
                  self.hidden.append(nn.Linear(layers[k], layers[k+1][0])) 
                  self.hidden.append(nn.Linear(layers[k], layers[k+1][1]))
            else: # if not variational, make nets for latent state only
                self.hidden.append(nn.Linear(layers[k], layers[k+1])) 
            

    def forward(self, x):
        if self.variational:
            for k, layer in enumerate(self.hidden):
                  if k < len(self.hidden) - 2:
                        x = layer(x)
                  elif k == len(self.hidden) - 2:
                        mu = layer(x)
                  else:
                        logvar = layer(x)
                        return mu, logvar
        else:
            for layer in self.hidden:
                  x = layer(x)
            return x
        
    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c
    

class DecoderNet(nn.Module):
    def __init__(self, layers, nonlinearity):
        super(DecoderNet, self).__init__()
        
        self.hidden = nn.ModuleList()
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        for k in range(self.n_layers):    
            self.hidden.append(nn.Linear(layers[k], layers[k+1]))
            if k < self.n_layers - 1:
                self.hidden.append(nonlinearity())
    
    def forward(self, x):
        for layer in self.hidden:
            x = layer(x)
        return x
    
    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c


class dynamics(nn.Module):
    def __init__(self, b, init_scale):
        super(dynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = gaussian_init_(b, std=1)           
        U, _, V = torch.svd(self.dynamics.weight.data)
        self.dynamics.weight.data = torch.mm(U, V.t()) * init_scale
        
    def forward(self, x):
        x = self.dynamics(x)
        return x
    
    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c


class dynamics_back(nn.Module):
    def __init__(self, b, omega):
        super(dynamics_back, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = torch.pinverse(omega.dynamics.weight.data.t())     

    def forward(self, x):
        x = self.dynamics(x)
        return x
    
    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c


class KoopmanAE(nn.Module):
    def __init__(self, encoder_layers, decoder_layers, steps, steps_back, init_scale=1, nonlinearity = nn.Tanh()):
        super(KoopmanAE, self).__init__()
        self.steps = steps
        self.steps_back = steps_back
        
        self.encoder = EncoderNet(layers = encoder_layers, nonlinearity = nonlinearity)
        self.dynamics = dynamics(decoder_layers[0], init_scale)
        self.backdynamics = dynamics_back(decoder_layers[0], self.dynamics)
        self.decoder = DecoderNet(layers = decoder_layers, nonlinearity = nonlinearity)


    def forward(self, x, mode='forward'):
        out = []
        out_id = []
        out_back = []
        out_back_id = []
        z = self.encoder(x.contiguous())
        q = z.contiguous()

        
        if mode == 'forward':
            for _ in range(self.steps):
                q = self.dynamics(q)
                out.append(self.decoder(q))

            out_id.append(self.decoder(z.contiguous())) 
            return out, out_id  

        if mode == 'backward':
            for _ in range(self.steps_back):
                q = self.backdynamics(q)
                out_back.append(self.decoder(q))

            out_back_id.append(self.decoder(z.contiguous())) 
            return out_back, out_back_id    
        
    def count_params(self):
        return self.encoder.count_params() + self.dynamics.count_params() + self.backdynamics.count_params() + self.decoder.count_params()


# VAE architecture with forward operator
class VAE_fwd(nn.Module):
    def __init__(self, encoder_layers, forward_layers, decoder_layers, nonlinearity, variational = True):
        super(VAE_fwd, self).__init__()
        self.encoder = EncoderNet(layers = encoder_layers, nonlinearity = nonlinearity, variational = variational)
        self.forward_operator = EncoderNet(layers = forward_layers, nonlinearity = nonlinearity, variational = variational)
        self.decoder = DecoderNet(layers = decoder_layers, nonlinearity = nonlinearity)

    def reparameterize(self, mu, logvar):
        # equals to draw a sample from Gaussian distribution
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std   
     
    def forward(self, x):
        mu1, logvar1 = self.encoder(x)
        z_t = self.reparameterize(mu1, logvar1)
        mu2, logvar2 = self.forward_operator(z_t)
        z_t1 = self.reparameterize(mu2, logvar2)
        x_pred = self.decoder(z_t1)
        x_identity = self.decoder(z_t)
        return x_pred, x_identity, mu1, logvar1, mu2, logvar2

    def count_params(self):     
        return self.encoder.count_params() + self.decoder.count_params() + self.forward_operator.count_params()

# AE architecture with forward operator
class AE_fwd(nn.Module):
    def __init__(self, encoder_layers, forward_layers, decoder_layers, nonlinearity):
        super(AE_fwd, self).__init__()
        self.encoder = EncoderNet(layers = encoder_layers, nonlinearity = nonlinearity)
        # self.forward_operator = EncoderNet(layers = forward_layers, nonlinearity = nonlinearity)
        assert forward_layers[0] == forward_layers[1], 'forward operator must be a square matrix'
        self.forward_operator = nn.Linear(forward_layers[0], forward_layers[1], bias = False)
        self.decoder = DecoderNet(layers = decoder_layers, nonlinearity = nonlinearity)
     
    def forward(self, x):
        z_t = self.encoder(x)
        x_identity = self.decoder(z_t)

        z_t1 = self.forward_operator(z_t)
        x_pred = self.decoder(z_t1)
        return x_pred, x_identity

    def count_params(self):     
        return self.encoder.count_params() + self.decoder.count_params() # + self.forward_operator.count_params()


# vanilla VAE architecture
class VAE(nn.Module):
    def __init__(self, encoder_layers, decoder_layers, nonlinearity):
        super(VAE, self).__init__()
        self.encoder = EncoderNet(layers = encoder_layers, nonlinearity = nonlinearity)
        self.decoder = DecoderNet(layers = decoder_layers, nonlinearity = nonlinearity)

    def reparameterize(self, mu, logvar):
        # equals to draw a sample from Gaussian distribution
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std   
     
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z)
      #   x_identity = self.decoder(mu)
        return x_pred, mu, logvar

    def count_params(self):     
        return self.encoder.count_params() + self.decoder.count_params()