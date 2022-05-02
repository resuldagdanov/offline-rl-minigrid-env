import torch
import torch.nn as nn


class DDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(DDQN, self).__init__()

        self.input_shape = state_size
        self.action_shape = action_size
        
        self.layer_1 = nn.Linear(self.input_shape[0], hidden_size)
        self.leyer_2 = nn.Linear(hidden_size, hidden_size)
        self.layer_3 = nn.Linear(hidden_size, self.action_shape)

    def forward(self, input):
        out = torch.relu(self.layer_1(input))
        out = torch.relu(self.leyer_2(out))
        out = self.leyer_3(out)
        
        return out
        