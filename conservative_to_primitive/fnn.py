import torch
import torch.nn as nn
import torch.nn.functional as F

class FNN_Small(nn.Module):
    def __init__(self, input_size, output_size):
        super(FNN_Small, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 600),
            nn.ReLU(),
            nn.Linear(600, 450),
            nn.ReLU(),
            nn.Linear(450, output_size)
        )

    def forward(self, x):
        return self.model(x)

class FNN_Large(nn.Module):
    def __init__(self, input_size, output_size):
        super(FNN_Large, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.model(x)

class FNN_Dynamic(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super(FNN_Dynamic, self).__init__()
        layers = []

        in_features = input_size
        for hidden_units in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.ReLU())
            in_features = hidden_units

        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



class FNN_Block(nn.Module):
    def __init__(self, input_dim, hidden_dim1=600, hidden_dim2=200):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, input_dim)
        )

    def forward(self, x):
        return self.model(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_blocks):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_blocks)

    def forward(self, x):
        logits = self.fc(x)
        return F.softmax(logits, dim=-1)

class DAIN_GRMHD(nn.Module):
    def __init__(self, input_dim, output_dim, num_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            FNN_Block(input_dim) for _ in range(num_blocks)
        ])
        self.gater = GatingNetwork(input_dim, num_blocks)
        self.final = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        gates = self.gater(x)
        outputs = []

        for i, block in enumerate(self.blocks):
            out = block(x)
            weighted_out = out * gates[:, i].unsqueeze(-1)
            outputs.append(weighted_out)

        # combined = sum(outputs)# changing this
        combined = torch.sum(torch.stack(outputs), dim=0)
        return self.final(combined)