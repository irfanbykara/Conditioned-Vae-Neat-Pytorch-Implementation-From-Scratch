from typing import Callable, List
import torch
import torch.utils.data as data
from metrics import Metric


class BaselineTrainer:
    def __init__(self, model: torch.nn.Module,
                 loss: Callable,
                 optimizer: torch.optim.Optimizer,
                 use_cuda=True):
        self.loss = loss
        self.use_cuda = use_cuda
        self.optimizer = optimizer
        self.model = model

        if use_cuda:
            self.model = model.to(device="cuda:0")

    def fit(self, train_data_loader: data.DataLoader,
            epoch: int):
        avg_loss = 0
        self.model.training = True
        for e in range(epoch):
            print(f"Start epoch {e+1}/{epoch}")
            n_batch = 0
            for i, (x, attr) in enumerate(train_data_loader):
                # Reset previous gradients
                self.optimizer.zero_grad()
                # Move data to cuda is necessary:
                if self.use_cuda:
                    x = x.cuda()
                    attr = attr.cuda()
                batch_size = x.shape[0]
                # x = x.view(batch_size, -1)

                # Make forward
                # TODO change this part to fit with the VAE framework
                x,x_hat,mean,log_var = self.model(x,attr)
                loss = self.loss(x,x_hat,mean,log_var)
                loss.backward()

                # Adjust learning weights
                self.optimizer.step()
                avg_loss += loss.item()
                n_batch += 1

                print(f"\r{i+1}{len(train_data_loader)}: loss = {loss / n_batch}", end='')
            print()

        return avg_loss

    def eval(self, val_data_loader: data.DataLoader,
             metrics: List[Metric],
             vae:torch.nn.Module,
             conditions: torch.Tensor = None
             ) -> torch.Tensor:
        scores = torch.zeros(len(metrics), dtype=torch.float32)
        with torch.no_grad():
            for i, m in enumerate(metrics):
                scores[i] = m.compute_metric(val_data_loader, vae,conditions)

        return scores
