import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
            #version 1
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(196, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10),
#         )
        #self.linear_relu_stack = nn.Sequential(
            # version 3
            #nn.Linear(196, 4096), #59 #13 #196
            #nn.ReLU(),
            #nn.BatchNorm1d(4096),
            #nn.Dropout(0.25),
            #nn.Linear(4096, 8192),
            #nn.ReLU(),
            #nn.BatchNorm1d(8192),
            #nn.Dropout(0.25),
            #nn.Linear(8192, 16384),
            #nn.ReLU(),
            #nn.BatchNorm1d(16384),
            #nn.Dropout(0.25),
            ###
            # version 3 + version 4
#             nn.Linear(16384, 8192),
#             nn.ReLU(),
#             nn.BatchNorm1d(8192),
#             nn.Dropout(0.25),
#             nn.Linear(8192, 4096),
#             nn.ReLU(),
#             nn.BatchNorm1d(4096),
#             nn.Dropout(0.25),
            #nn.Linear(16384, 10)
        #)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(196, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits