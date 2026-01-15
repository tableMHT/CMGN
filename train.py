# train.py
import time
import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import compute_metrics

def train_one_epoch(model, dataloader, optimizer, device, cfg):
    model.train()
    running_loss = 0.0
    y_true_all, y_pred_all = [], []
    lam = cfg.get('tcmc_weight', 0.0)
    for eegs, eyes, labels, lengths, mask in tqdm(dataloader, desc='train', leave=False):
        eegs = eegs.to(device); eyes = eyes.to(device); labels = labels.to(device); mask = mask.to(device)
        optimizer.zero_grad()
        out = model(eegs, eyes, lengths, mask)
        if isinstance(out, tuple):
            logits, aux = out
            tcmc = aux.get('tcmc_loss', torch.tensor(0.0, device=logits.device))
        else:
            logits = out
            tcmc = torch.tensor(0.0, device=logits.device)
        loss = F.cross_entropy(logits, labels) + lam * tcmc
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        y_pred_all.extend(logits.argmax(dim=1).detach().cpu().tolist())
        y_true_all.extend(labels.detach().cpu().tolist())
    epoch_loss = running_loss / len(dataloader.dataset)
    acc, f1, cm = compute_metrics(y_true_all, y_pred_all)
    return epoch_loss, acc, f1, cm

@torch.no_grad()
def eval_one_epoch(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    y_true_all, y_pred_all = [], []
    for eegs, eyes, labels, lengths, mask in tqdm(dataloader, desc='eval', leave=False):
        eegs = eegs.to(device); eyes = eyes.to(device); labels = labels.to(device); mask = mask.to(device)
        logits = model(eegs, eyes, lengths, mask)
        loss = F.cross_entropy(logits, labels)
        running_loss += loss.item() * labels.size(0)
        y_pred_all.extend(logits.argmax(dim=1).cpu().tolist())
        y_true_all.extend(labels.cpu().tolist())
    epoch_loss = running_loss / len(dataloader.dataset)
    acc, f1, cm = compute_metrics(y_true_all, y_pred_all)
    return epoch_loss, acc, f1, cm

def tent_style_eval(model, dataloader, device, cfg):
    model.eval()
    adaptable = []
    if hasattr(model, 'classifier'):
        adaptable += list(model.classifier.parameters())
    if hasattr(model, 'style_gate'):
        adaptable += [model.style_gate.alpha, model.style_gate.beta]
    tta_lr = cfg.get('tta_lr', 1e-4)
    optimizer = torch.optim.Adam(adaptable, lr=tta_lr)

    steps = int(cfg.get('tta_steps', 1))
    reg_w = cfg.get('tta_reg_alpha', 1e-4)

    running_loss = 0.0
    y_true_all, y_pred_all = [], []
    for eegs, eyes, labels, lengths, mask in tqdm(dataloader, desc='eval(TENT-Style)', leave=False):
        eegs = eegs.to(device); eyes = eyes.to(device); labels = labels.to(device); mask = mask.to(device)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            logits = model(eegs, eyes, lengths, mask)  
            prob = F.softmax(logits, dim=1)
            entropy = -(prob * (prob.clamp_min(1e-6).log())).sum(dim=1).mean()
            reg = 0.0
            if hasattr(model, 'style_gate'):
                reg = reg_w * ((model.style_gate.alpha-1.0).pow(2).mean() + (model.style_gate.beta-0.0).pow(2).mean())
            loss = entropy + reg
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            logits = model(eegs, eyes, lengths, mask)
            loss = F.cross_entropy(logits, labels)
        running_loss += loss.item() * labels.size(0)
        y_pred_all.extend(logits.argmax(dim=1).cpu().tolist())
        y_true_all.extend(labels.cpu().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    acc, f1, cm = compute_metrics(y_true_all, y_pred_all)
    return epoch_loss, acc, f1, cm

def fit(model, train_loader, val_loader, cfg):
    device = cfg['device']
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val_loss = float('inf')
    best_model_state = None
    patience_cnt = 0
    for epoch in range(cfg['epochs']):
        t0 = time.time()
        train_loss, train_acc, train_f1, _ = train_one_epoch(model, train_loader, optimizer, device, cfg)
        # 验证：支持 TENT-Style
        if cfg.get('tta_mode','none') == 'tent_style':
            val_loss, val_acc, val_f1, _ = tent_style_eval(model, val_loader, device, cfg)
        else:
            val_loss, val_acc, val_f1, _ = eval_one_epoch(model, val_loader, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{cfg['epochs']}: train_loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} time={(time.time()-t0):.1f}s")
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save({'model': best_model_state, 'cfg': cfg}, cfg['ckpt_path'])
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg['patience']:
                print("Early stopping.")
                break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model
