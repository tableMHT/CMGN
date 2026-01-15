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
    # 移除 --encoder_type, --tcmc, --tta，因为它们已在 config.py 中固定
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
    
    # --- 3折被试交叉验证 ---
    
    # 1. 获取所有被试ID
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

    print(f"--- 开始 {n_splits}-Fold 被试交叉验证 (使用 Hyena, TCMC, TENT-Style TTA) ---")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_subject_ids)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        
        # 2. 划分训练集和验证集被试
        train_subjects = all_subject_ids[train_idx].tolist()
        val_subjects = all_subject_ids[val_idx].tolist()
        
        print(f"Train subjects: {train_subjects}")
        print(f"Val subjects: {val_subjects}")

        # 3. 构建 DataLoaders
        # 我们使用 build_loaders_by_subject，它内部使用 SeedIVDataset
        train_loader, val_loader = build_loaders_by_subject(
            cfg, 
            train_subjects, 
            val_subjects,
            batch_size=cfg['batch_size'],
            num_workers=args.num_workers
        )

        # 4. 初始化模型 (SEED-IV 是4分类)
        # 编码器、TCMC、TTA配置已在 config.py 中固定
        model = DE_FG_FR_Model(cfg, num_classes=4)
        
        # 5. 训练 (fit 函数内部将使用 TENT-Style TTA 进行验证)
        model = fit(model, train_loader, val_loader, cfg)

        # 6. 在验证集上报告最终的 *非TTA* 评估结果 (作为标准评估)
        val_loss, val_f1, val_acc, cm = eval_one_epoch(model, val_loader, cfg['device'])
        
        print(f"\nFold {fold+1} 最终评估 (eval_one_epoch):")
        print("Loss={:.4f} Acc={:.4f} F1={:.4f}".format(val_loss, val_acc, val_f1))
        print(f"Confusion Matrix:\n{cm}")
        
        fold_results_acc.append(val_acc)
        fold_results_f1.append(val_f1)

    # --- 报告 3-Fold CV 平均结果 ---
    print("\n--- 3-Fold 交叉验证完成 ---")
    print(f"平均验证准确率 (Acc): {np.mean(fold_results_acc):.4f}")
    print(f"平均验证 F1-Score: {np.mean(fold_results_f1):.4f}")

if __name__ == '__main__':
    main()