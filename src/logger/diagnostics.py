# diagnostics.py
import torch

def tstats(x, name, logger, max_items=5):
    if x is None:
        logger.debug(f"{name}: None")
        return
    if isinstance(x, torch.Tensor):
        device = x.device
        shape = tuple(x.shape)
        nan = torch.isnan(x).any().item()
        inf = torch.isinf(x).any().item()
        logger.debug(f"{name}: shape={shape}, dtype={x.dtype}, device={device}, nan={nan}, inf={inf}")
        # немного значений (осторожно с большими тензорами)
        with torch.no_grad():
            flat = x.detach().view(-1)
            if flat.numel() > 0:
                head = flat[:max_items].cpu().tolist()
                logger.debug(f"{name}: head={head}")
    else:
        logger.debug(f"{name}: type={type(x)} value={str(x)[:200]}")

def assert_finite(x, what, logger):
    if x is None: 
        return
    if torch.isnan(x).any() or torch.isinf(x).any():
        logger.error(f"Found NaN/Inf in {what}")
        raise FloatingPointError(f"NaN/Inf in {what}")
