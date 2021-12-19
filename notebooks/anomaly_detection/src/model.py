import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(196, 64),
            nn.ELU(),
            nn.Linear(64, 16),
            nn.ELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ELU(),
            nn.Linear(64, 196),
            nn.ELU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x