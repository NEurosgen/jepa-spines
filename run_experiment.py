
import torch
import numpy as np
import torch.nn.functional as F
from config import cfg, update_cfg
from get_data import create_dataset

from model_utils.hyperbolic_dist import hyperbolic_dist
from trainer import run
from get_model import create_model

import hydra
from torch_geometric.loader import DataLoader  # <-- только DataLoader берём из PyG
from torch.utils.data import random_split
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor
from log import config_logger
from asam import ASAM
import os
import torch
import random
import time
import numpy as np



def train(train_loader, model, optimizer, evaluator, device, momentum_weight, sharp=None, criterion_type=0):

    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    step_losses, num_targets = [], []
    for data in train_loader:
        if model.use_lap: 
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
        step_losses.append(loss.item())
        num_targets.append(len(target_y))
        
  
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for param_q, param_k in zip(model.context_encoder.parameters(), model.target_encoder.parameters()):
                param_k.data.mul_(momentum_weight).add_((1.-momentum_weight) * param_q.detach().data)
        
    epoch_loss = np.average(step_losses, weights=num_targets)
    return None, epoch_loss


@ torch.no_grad()
def test(loader, model, evaluator, device, criterion_type=0):
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    step_losses, num_targets = [], []
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
        step_losses.append(loss.item())
        num_targets.append(len(target_y))

    epoch_loss = np.average(step_losses, weights=num_targets)
    return None, epoch_loss



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def exp_train(cfg, create_dataset, create_model, train, evaluator=None,seed = 42):
    set_seed(seed)
    train_dataset, val_dataset, test_dataset = create_dataset(cfg,seed=seed)

    train_loader = DataLoader(train_dataset, cfg.train.batch_size, shuffle=True,  num_workers=cfg.num_workers)
    val_loader   = DataLoader(val_dataset,  cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)

  
    from pathlib import Path
    run_dir = Path(f"runs/{cfg.dataset}_jepa/seed_{seed:04d}")
    run_dir.mkdir(parents=True, exist_ok=True)

    report = []  
    
    model = create_model(cfg).to(cfg.device)
    print(f"\nNumber of parameters: {count_parameters(model)}")

    if cfg.train.optimizer == 'ASAM':
        sharp = True
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train.lr, momentum=0.9, weight_decay=cfg.train.wd)
        minimizer = ASAM(optimizer, model, rho=0.5)
        optim_for_scheduler = optimizer  # ReduceLROnPlateau будет смотреть сюда
    else:
        sharp = False
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
        optim_for_scheduler = optimizer

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim_for_scheduler, mode='min', factor=cfg.train.lr_decay, patience=cfg.train.lr_patience, verbose=True
    )

    start_outer = time.time()
    per_epoch_time = []

    # Create EMA scheduler for target encoder param update
    ipe = max(1, len(train_loader))
    ema_params = [0.996, 1.0]
    momentum_scheduler = (ema_params[0] + i*(ema_params[1]-ema_params[0])/(ipe*cfg.train.epochs)
                          for i in range(int(ipe*cfg.train.epochs)+1))

    # трекинг лучшей модели
    best_val = float("inf")
    best_state = None
    save_ckpt = bool(getattr(cfg, "checkpoint", False))

    for epoch in range(cfg.train.epochs):
        start = time.time()
        model.train()
        _, train_loss = train(
            train_loader, model, optimizer if not sharp else minimizer,
            evaluator=evaluator, device=cfg.device, momentum_weight=next(momentum_scheduler),
            sharp=sharp, criterion_type=cfg.jepa.dist
        )
        model.eval()
        _, val_loss = test(val_loader, model, evaluator=evaluator, device=cfg.device, criterion_type=cfg.jepa.dist)

        time_cur_epoch = time.time() - start
        per_epoch_time.append(time_cur_epoch)

        # лог лернинга
        cur_lr = optim_for_scheduler.param_groups[0]['lr']
        report.append({'Epoch': epoch, 'train_loss': float(train_loss), 'val_loss': float(val_loss),
                       'lr': float(cur_lr), 'time_sec': float(time_cur_epoch)})

        # чекпоинты last/best (если включены)
        if save_ckpt:
            torch.save({'model': model.state_dict()}, run_dir / 'last.pt')
            if val_loss < best_val:
                best_val = float(val_loss)
                best_state = {'model': model.state_dict()}
                torch.save(best_state, run_dir / 'best.pt')

     
        if scheduler is not None:
            scheduler.step(val_loss)

        if not sharp:
            if optimizer.param_groups[0]['lr'] < cfg.train.min_lr:
                print("!! LR EQUAL TO MIN LR SET.")
                break

    

    #  если был лучший стейт — загрузим его в модель
    if best_state is not None:
        model.load_state_dict(best_state['model'])

    return model, report

from representation_metrtic import encode_repr,fit_and_eval_linear



def run_experiment(
    cfg,
    create_dataset=create_dataset,
    create_model=create_model,
    train=train,
    test=test,
    seeds=(42,),
):
    """
    Запускает цикл по сидами: тренируем, кодируем train/val,
    учим линейный классификатор, собираем и возвращаем метрики.
    Ничего не сохраняет на диск.
    """
    import numpy as np, torch

    all_linear = {}  
    all_hist = {}   

    for seed in seeds:

        model, history = exp_train(
            cfg,
            create_dataset=create_dataset,
            create_model=create_model,
            train=train,
            evaluator=None,
            seed=seed,
        )
        model.eval()
        all_hist[seed] = history


        train_ds, val_ds, test_ds = create_dataset(cfg, seed=seed)
        assert len(train_ds) > 0, "Empty train_ds for this trial"
        assert len(val_ds)   > 0, "Empty val_ds for this trial"

        train_loader = DataLoader(train_ds, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers)
        val_loader   = DataLoader(val_ds,   cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)


        X_train, y_train = encode_repr(loader=train_loader, model=model, device=str(cfg.device))
        X_val,   y_val   = encode_repr(loader=val_loader,   model=model, device=str(cfg.device))

 
        X_train = torch.as_tensor(X_train, dtype=torch.float32)
        y_train = torch.as_tensor(y_train, dtype=torch.long)
        X_val   = torch.as_tensor(X_val, dtype=torch.float32)
        y_val   = torch.as_tensor(y_val, dtype=torch.long)

        lin_report = fit_and_eval_linear(X_tr=X_train, y_tr=y_train, X_te=X_val, y_te=y_val)
        all_linear[seed] = {k: float(v) for k, v in lin_report.items()}


    agg = {}
    if all_linear:
        keys = sorted(next(iter(all_linear.values())).keys())
        agg = {f"{k}_mean": float(np.mean([all_linear[s][k] for s in seeds])) for k in keys}
        agg.update({f"{k}_std": float(np.std([all_linear[s][k] for s in seeds], ddof=0)) for k in keys})

    return {
        "histories": all_hist,    
        "linear": all_linear,      
        "linear_agg": agg,        
    }










