import torch
import numpy as np
import torch.nn.functional as F
from config import cfg, update_cfg
from get_data import create_dataset

from model_utils.hyperbolic_dist import hyperbolic_dist
from trainer import run
from get_model import create_model

import hydra
from torch.utils.data import DataLoader,random_split
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor





def train(train_loader, model, optimizer, evaluator, device, momentum_weight, sharp=None, criterion_type=0):

    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    step_losses, num_targets = [], []
    for data in train_loader:
        if model.use_lap: # Sign flips for eigenvalue PEs
            batch_pos_enc = data.lap_pos_enc
            sign_flip = torch.rand(batch_pos_enc.size(1))
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
        data = data.to(device)
        optimizer.zero_grad()
        target_x, target_y = model(data)
        if criterion_type == 0:
            loss = criterion(target_x, target_y)
        elif criterion_type == 1:
            loss = F.mse_loss(target_x, target_y)
        elif criterion_type == 2:
            loss = hyperbolic_dist(target_x, target_y)
        else:
            print('Loss function not supported! Exiting!')
            exit()
        # Will need these for the weighted average at the end of the epoch
        step_losses.append(loss.item())
        num_targets.append(len(target_y))
        
        # Update weights of the network 
        loss.backward()
        optimizer.step()

        # Update the target encoder using an exponential smoothing of the context encoder
        with torch.no_grad():
            for param_q, param_k in zip(model.context_encoder.parameters(), model.target_encoder.parameters()):
                param_k.data.mul_(momentum_weight).add_((1.-momentum_weight) * param_q.detach().data)
        
    epoch_loss = np.average(step_losses, weights=num_targets)
    return None, epoch_loss # Leave none for now since maybe we'd like to return the embeddings for visualization


@ torch.no_grad()
def test(loader, model, evaluator, device, criterion_type=0):
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    step_losses, num_targets = [], []
   # print(model)
    for data in loader:
        data = data.to(device)
        target_x, target_y = model(data)
        if criterion_type == 0:
            loss = criterion(target_x, target_y)
        elif criterion_type == 1:
            loss = F.mse_loss(target_x, target_y)
        elif criterion_type == 2:
            loss = hyperbolic_dist(target_x, target_y)
        else:
            print('Loss function not supported! Exiting!')
            exit()
        # Will need these for the weighted average at the end of the epoch
        step_losses.append(loss.item())
        num_targets.append(len(target_y))

    epoch_loss = np.average(step_losses, weights=num_targets)
    return None, epoch_loss


@hydra.main(version_base=None,config_path="train/configs", config_name="zinc")
def main(cfg: DictConfig):
    run(cfg, create_dataset=create_dataset, create_model=create_model, train=train,test= test)
    

if __name__ == "__main__":
    main()
