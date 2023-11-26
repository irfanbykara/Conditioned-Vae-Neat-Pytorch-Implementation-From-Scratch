import torch
from modules import *
from models import VariationalAutoEncoder
from losses import VAELoss
from dataset import MNISTAttributes
from torch.utils.data import Dataset,random_split
from torchvision import transforms
from trainer import BaselineTrainer
from torch.utils.data import DataLoader
from metrics import InceptionScore,FrechetInceptionDistance
import torch.nn.functional as F
import random

def main():

    if torch.cuda.is_available():
        device = "cuda:0"
        cuda_available = True

    else:
        device = "cpu"
        cuda_available = False

    num_samples = 10 #This is for calculating the metrics and showing some results. Should be increased in ideal cases
    latent_dim = 64
    num_classes = 10 #10 for mnist. Change it for your custom dataset.
    beta = 10

    encoder_module = EncoderModule(latent_dim=latent_dim,num_classes=num_classes)
    decoder_module = DecoderModule(latent_dim=latent_dim, num_classes=num_classes)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    vae = VariationalAutoEncoder(latent_dim= latent_dim,encoder_module=encoder_module, decoder_module=decoder_module)
    vae = vae.to(device)
    # Apply the initialization to your model
    vae.apply(init_weights)

    learning_rate = 0.01
    weight_decay = 1e-5  # Choose an appropriate value for weight decay
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=weight_decay)

    vae_loss = VAELoss()

    # Example of usage:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = MNISTAttributes(transform=transform, mode='train')

    # Define the sizes for the train and test sets
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    # Create train and test datasets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    dataloader_train = DataLoader(train_dataset, batch_size=64, shuffle=True)
    dataloader_test = DataLoader(test_dataset, batch_size=64, shuffle=True)

    #This is for conditioned VAE. Creating a random tensor of 10 label.
    random_conditions = torch.randint(0, 10, (10,),device=device)
    print(f"Random conditiones are: {random_conditions}")
    trainer = BaselineTrainer(model=vae,loss=vae_loss,optimizer=optimizer,use_cuda=cuda_available)
    trainer.fit(train_data_loader=dataloader_train,epoch=2)


    # Set the model to evaluation mode
    vae.eval()

    inception_calculator = InceptionScore(num_samples=num_samples,latent_dim=latent_dim)
    frechet_calculator = FrechetInceptionDistance(num_samples=num_samples,latent_dim=latent_dim)

    random_indices = random.sample(range(len(test_dataset)), num_samples)
    subset_test_dataset = [test_dataset[i] for i in random_indices]

    # Create a new DataLoader with the subset of 10 items
    subset_test_dataloader = DataLoader(subset_test_dataset, batch_size=64, shuffle=True)
    metrics = [inception_calculator,frechet_calculator]
    scores = trainer.eval(subset_test_dataloader,metrics,vae,random_conditions)
    print(f"Inception score for the model model is: {scores[0]}\nFrechet Distance is: {scores[1]}.")

if __name__ == "__main__":
    main()