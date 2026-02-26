import torch
from torch import nn
from torch import amp
from tqdm import tqdm

from unet import Unet

from dataset import get_train_data


class CustomStaticGradScaler():
    def __init__(self, scale_factor: float = 2.0**16):
        self.scale_factor = scale_factor

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale_factor

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.div_(self.scale_factor)
        optimizer.step()
        optimizer.zero_grad()
    
    def update(self) -> None:
        pass

class CustomDynamicGradScaler():
    def __init__(
        self,
        scale_factor: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 100,
    ):
        self.scale_factor = scale_factor
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.consecutive_iterations_without_inf_or_nan = 0
        self._found_inf = False

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale_factor

    def _has_inf_or_nan(self, optimizer: torch.optim.Optimizer) -> bool:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        return True
        return False
 
    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if not self._has_inf_or_nan(optimizer):
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.data.div_(self.scale_factor)
            optimizer.step()
            self.consecutive_iterations_without_inf_or_nan += 1
            self._found_inf = False
        else:
            self.consecutive_iterations_without_inf_or_nan = 0
            self._found_inf = True
        optimizer.zero_grad()

    def update(self) -> None:
        if self._found_inf:
            self.scale_factor *= self.backoff_factor
        elif self.consecutive_iterations_without_inf_or_nan >= self.growth_interval:
            self.scale_factor *= self.growth_factor
            self.consecutive_iterations_without_inf_or_nan = 0


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
) -> None:
    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        with torch.amp.autocast(device.type, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        
        scaler.step(optimizer)

        scaler.update()


        accuracy = ((outputs > 0.5) == labels).float().mean()

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")


def train(scaler: amp.GradScaler | CustomStaticGradScaler | CustomDynamicGradScaler):
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data()

    num_epochs = 5
    for epoch in range(0, num_epochs):
        train_epoch(train_loader, model, criterion, optimizer, scaler, device=device)
