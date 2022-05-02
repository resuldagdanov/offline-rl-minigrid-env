import torch.nn as nn


class ConvDQN(nn.Module):
    
    def __init__(self, state_size, action_size, hidden_size):
        super(ConvDQN, self).__init__()

        self.obs_space = state_size
        self.action_space = action_size

        n = self.obs_space[0]
        m = self.obs_space[1]
        
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*hidden_size

        # define image embedding
        self.image_convolution = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.image_embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_space)
        )

    def forward(self, obs):
        out = obs.transpose(1, 3).transpose(2, 3)
        
        out = self.image_convolution(out)

        embeding = out.reshape(out.shape[0], -1)

        logit = self.mlp(embeding)

        return logit
