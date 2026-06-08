# model.py
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential( #couches entièrements connectées
            nn.Linear(32 * 14 * 14, 128),
            nn.ReLU(),
            # nn.Dropout(p=0.5),   
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)           # [1, 28, 28] → [32, 14, 14] : 1 canal d'entrée, 32 canaux de sortie, kernel de 3x3, padding de 1 pour conserver la taille, max pooling de 2x2 pour réduire la taille de moitié
        # x = self.conv_block2(x)           # [32, 14, 14] → [64, 7, 7] : 32 canaux d'entrée, 64 canaux de sortie, kernel de 3x3, padding de 1 pour conserver la taille, max pooling de 2x2 pour réduire la taille de moitié
        x = x.view(x.size(0), -1)         # [64, 7, 7] → [3136] : x.size(0) = batch size / -1 : infère la taille automatiquement
        x = self.fc_layers(x)             # [3136] → [10] : 10 classes de sortie 
        return x