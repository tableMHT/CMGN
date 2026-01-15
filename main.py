# main.py
import argparse
import torch
import numpy as np
from sklearn.model_selection import KFold
from config import config
from dataset import get_all_subjects, build_loaders_by_subject
from model import DE_FG_FR_Model
from train import fit, eval_one_epoch
from utils import set_seed

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eeg_dir', type=str, default=None)
    ap.add_argument('--eye_dir', type=str, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--num_workers', type=int, default=2)
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = dict(config)
    if args.eeg_dir: cfg['eeg_dir'] = args.eeg_dir
    if args.eye_dir: cfg['eye_dir'] = args.eye_dir
    if args.epochs: cfg['epochs'] = args.epochs
    if args.batch_size: cfg['batch_size'] = args.batch_size

    set_seed(cfg['seed'])
    
    try:
        all_subject_ids = np.array(get_all_subjects(cfg))
    except Exception as e:
        print(f"加载被试列表失败: {e}")
        print("请确保 config.py 中的 'eeg_dir' 路径正确。")
        return

    n_splits = 3
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=cfg['seed'])
    
    fold_results_acc = []
    fold_results_f1 = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_subject_ids)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        

        train_subjects = all_subject_ids[train_idx].tolist()
        val_subjects = all_subject_ids[val_idx].tolist()
        
        print(f"Train subjects: {train_subjects}")
        print(f"Val subjects: {val_subjects}")

        train_loader, val_loader = build_loaders_by_subject(
            cfg, 
            train_subjects, 
            val_subjects,
            batch_size=cfg['batch_size'],
            num_workers=args.num_workers
        )

        model = DE_FG_FR_Model(cfg, num_classes=4)
        
        model = fit(model, train_loader, val_loader, cfg)

        val_loss, val_f1, val_acc, cm = eval_one_epoch(model, val_loader, cfg['device'])
        
        print(f"\nFold {fold+1} evaluate (eval_one_epoch):")
        print("Loss={:.4f} Acc={:.4f} F1={:.4f}".format(val_loss, val_acc, val_f1))
        print(f"Confusion Matrix:\n{cm}")
        
        fold_results_acc.append(val_acc)
        fold_results_f1.append(val_f1)

    print(f"平均验证准确率 (Acc): {np.mean(fold_results_acc):.4f}")
    print(f"平均验证 F1-Score: {np.mean(fold_results_f1):.4f}")

if __name__ == '__main__':
    main()
