import torch
import torch.nn as nn

class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [B, len, x_dim/y_dim]
    '''

    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        # x_samples shape: [B, len, x_dim]
        B, len, x_dim = x_samples.shape
        x_samples = x_samples.view(B * len, x_dim)  # Flatten to [B*len, x_dim]
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        mu = mu.view(B, len, -1)  # Reshape back to [B, len, y_dim]
        logvar = logvar.view(B, len, -1)  # Reshape back to [B, len, y_dim]
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()

        # log of conditional probability of negative sample pairs
        prediction_1 = mu.unsqueeze(2)  # shape [B, len, 1, y_dim]
        y_samples_1 = y_samples.unsqueeze(1)  # shape [B, 1, len, y_dim]
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=2) / 2. / logvar.exp()

        return (positive.sum(dim=-1).mean(dim=1) - negative.sum(dim=-1).mean(dim=1)).mean()

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=-1).mean(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)
