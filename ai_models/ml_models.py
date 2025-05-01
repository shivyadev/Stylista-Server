import torch.nn as nn


class EnhancedFocalMLPClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EnhancedFocalMLPClassifier, self).__init__()
        self.network = nn.Sequential(
            # Input layer with larger initial dimension
            nn.Linear(input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.6),
            
            # Additional residual-like connections
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            # Deeper network with skip connections
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, num_classes)
        )
            
    def forward(self, x):
        return self.network(x)
