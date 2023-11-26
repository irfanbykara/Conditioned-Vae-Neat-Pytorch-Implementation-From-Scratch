import torch
import torch.nn as nn
import torch.nn.functional as F

# Define Encoder Module


class EncoderModule(nn.Module):

    def __init__(self, latent_dim,num_classes):

        super(EncoderModule, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7 + num_classes, 256)  # Add num_classes to input
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x,y):

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        y_one_hot = F.one_hot(y, num_classes=self.num_classes)

        # Reshape to have the same number of dimensions as x
        y_one_hot = y_one_hot.view(y.size(0), self.num_classes)  # Adjust the size based on your specific use case

        x = torch.cat((x, y_one_hot), dim=1)
        x = torch.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class DecoderModule(nn.Module):

    def __init__(self, latent_dim,num_classes):
        super(DecoderModule, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(latent_dim + num_classes, 256)  # Add num_classes to input
        self.fc2 = nn.Linear(256,64*7*7)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(1)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # Adjusted parameters
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)   # Adjusted parameters

    def forward(self, x,y):

        y_one_hot = F.one_hot(y, num_classes=self.num_classes)

        # Reshape to have the same number of dimensions as x
        y_one_hot = y_one_hot.view(y.size(0), self.num_classes)  # Adjust the size based on your specific use case
        x = torch.cat((x, y_one_hot), dim=1)
        x = torch.relu(self.fc(x))
        x = torch.relu(self.fc2(x))
        x = x.view(x.size(0), 64, 7, 7)  # Reshape for convolutional layers
        x = torch.relu(self.bn1(self.deconv1(x)))
        x = torch.sigmoid(self.bn2(self.deconv2(x)))

        return x
