from estimate_representation.representation_metrtic import load_model,fit_and_eval_linear,encode_repr,plot_umap_2d


from torch_geometric.loader import DataLoader 
import torch
from omegaconf import OmegaConf
from data_utils.get_data import create_dataset
cfg = OmegaConf.load("train/configs/zinc.yaml")
model =load_model(cfg)
model.eval()

train_ds,val_ds,test_ds = create_dataset(cfg)

train_loader = DataLoader(train_ds, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers)
val_loader   = DataLoader(val_ds,   cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)
test_loader   = DataLoader(test_ds,   cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)

X_train, y_train = encode_repr(loader=train_loader, model=model, device=str(cfg.device))
X_val,   y_val   = encode_repr(loader=val_loader,   model=model, device=str(cfg.device))
X_test,   y_test   = encode_repr(loader=test_loader,   model=model, device=str(cfg.device))
 
X_train = torch.as_tensor(X_train, dtype=torch.float32)
y_train = torch.as_tensor(y_train, dtype=torch.long)
X_val   = torch.as_tensor(X_val, dtype=torch.float32)
y_val   = torch.as_tensor(y_val, dtype=torch.long)

X_test = torch.as_tensor(X_test, dtype=torch.float32)
y_test = torch.as_tensor(y_test, dtype=torch.long)


lin_report = fit_and_eval_linear(X_tr=X_train, y_tr=y_train, X_te=X_test, y_te=y_test)
# plot_umap_2d(X=X_val,y=y_val)
print(lin_report)

def get_report_model(model , train_loader ,val_loader ,device):
    X_train, y_train = encode_repr(loader=train_loader, model=model, device=device)
    X_train = torch.as_tensor(X_train, dtype=torch.float32)
    y_train = torch.as_tensor(y_train, dtype=torch.long)
    X_val, y_val = encode_repr(loader=val_loader, model=model, device=device)
    X_val = torch.as_tensor(X_val, dtype=torch.float32)
    y_val = torch.as_tensor(y_val, dtype=torch.long)

    lin_report =fit_and_eval_linear(X_tr=X_train, y_tr=y_train, X_te=X_val, y_te=y_val)
    return lin_report


