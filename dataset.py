
import os
import glob
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

def _nan_guard_(t: torch.Tensor, clamp_val: float = 1e3):
    # Replace NaN/Inf and clamp to [-clamp_val, clamp_val].
    t = torch.nan_to_num(t, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
    t.clamp_(-clamp_val, clamp_val)
    return t

def _unify_eeg_trial(arr):
    # Accept EEG trial with axes among (time, 62, 5) and return (T, 5, 62).
    x = np.asarray(arr)
    if x.ndim != 3:
        raise ValueError(f"EEG trial must be 3D, got shape={x.shape}")
    try:
        ch_ax = [i for i, s in enumerate(x.shape) if s == 62][0]
    except IndexError:
        raise ValueError(f"Cannot find channel axis=62 in EEG trial shape={x.shape}")
    try:
        band_ax = [i for i, s in enumerate(x.shape) if s == 5][0]
    except IndexError:
        raise ValueError(f"Cannot find band axis=5 in EEG trial shape={x.shape}")
    time_ax = [i for i in range(3) if i not in (ch_ax, band_ax)][0]
    x = np.transpose(x, (time_ax, band_ax, ch_ax))  # (T,5,62)
    t = torch.tensor(x, dtype=torch.float32)
    return _nan_guard_(t)

def _unify_eye_trial(arr, eye_dim):
    # Accept eye trial with one axis==eye_dim and return (T, eye_dim).
    x = np.asarray(arr)
    if x.ndim not in (2, 3):
        raise ValueError(f"EYE trial must be 2D/3D, got shape={x.shape}")
    if x.ndim == 3 and 1 in x.shape:
        x = np.squeeze(x)
    if x.ndim == 2:
        if x.shape[0] == eye_dim:
            x = x.T
        elif x.shape[1] == eye_dim:
            pass
        else:
            diffs = [abs(s - eye_dim) for s in x.shape]
            feat_ax = int(np.argmin(diffs))
            if x.shape[feat_ax] != eye_dim:
                raise ValueError(f"EYE trial shape {x.shape} does not contain eye_dim={eye_dim}")
            if feat_ax == 0:
                x = x.T
    else:
        if eye_dim in x.shape:
            feat_ax = list(x.shape).index(eye_dim)
            time_axes = [i for i in range(3) if i != feat_ax]
            x = np.transpose(x, (*time_axes, feat_ax)).reshape(-1, eye_dim)
        else:
            raise ValueError(f"EYE trial shape {x.shape} does not contain eye_dim={eye_dim}")
    t = torch.tensor(x, dtype=torch.float32)
    return _nan_guard_(t)

def _match_eeg_keys(d, prefer_prefix, trials_per_session):
    keys = [k for k in d.keys() if not k.startswith('__')]
    if prefer_prefix:
        cand = [k for k in keys if k.startswith(prefer_prefix)]
        if len(cand) >= trials_per_session:
            def _s(k):
                ds = ''.join([c for c in k if c.isdigit()])
                return int(ds) if ds else 0
            return sorted(cand, key=_s)[:trials_per_session]
    de_keys = [k for k in keys if k.startswith('de') or 'de_' in k]
    if len(de_keys) >= trials_per_session:
        def _s(k):
            ds = ''.join([c for c in k if c.isdigit()])
            return int(ds) if ds else 0
        return sorted(de_keys, key=_s)[:trials_per_session]
    return sorted(keys)[:trials_per_session]

def _match_eye_keys(d, trials_per_session):
    keys = [k for k in d.keys() if not k.startswith('__')]
    def _knum(k):
        ds = ''.join([c for c in k if c.isdigit()])
        return int(ds) if ds else 0
    return sorted(keys, key=_knum)[:trials_per_session]

def _normalize_label_matrix(L, trials_per_session, n_subjects_total):
    # Return labels in shape (n_subjects_total, trials_per_session) without remapping.
    L = np.asarray(L)
    if L.ndim == 1:
        if L.shape[0] != trials_per_session:
            raise ValueError(f"Label vector length {L.shape[0]} != trials {trials_per_session}")
        return np.tile(L.reshape(1, -1), (n_subjects_total, 1))
    if L.ndim == 2:
        if 1 in L.shape and max(L.shape) == trials_per_session:
            L = L.reshape(-1)
            return np.tile(L.reshape(1, -1), (n_subjects_total, 1))
        if L.shape[1] == trials_per_session:
            return L
        if L.shape[0] == trials_per_session:
            return L.T
    raise ValueError(f"Unexpected label shape {L.shape}")

def _load_all_label_rows(label_dir, trials_per_session, n_subjects_total):
    l1 = sio.loadmat(os.path.join(label_dir, 'session1_label.mat'))['session1_label'].squeeze()
    l2 = sio.loadmat(os.path.join(label_dir, 'session2_label.mat'))['session2_label'].squeeze()
    l3 = sio.loadmat(os.path.join(label_dir, 'session3_label.mat'))['session3_label'].squeeze()
    l1 = _normalize_label_matrix(l1, trials_per_session, n_subjects_total)
    l2 = _normalize_label_matrix(l2, trials_per_session, n_subjects_total)
    l3 = _normalize_label_matrix(l3, trials_per_session, n_subjects_total)
    return l1.astype(int), l2.astype(int), l3.astype(int)

def _discover_subject_count(eeg_dir, sessions_per_subject):
    files = sorted(glob.glob(os.path.join(eeg_dir, '*.mat')))
    if len(files) % sessions_per_subject != 0:
        raise ValueError(f"EEG .mat count {len(files)} not divisible by sessions_per_subject={sessions_per_subject}")
    return len(files) // sessions_per_subject

def get_all_subjects(cfg):
    eeg_dir = cfg.get('eeg_dir')
    sessions_per_subject = int(cfg.get('sessions_per_subject', 3))
    n = _discover_subject_count(eeg_dir, sessions_per_subject)
    return list(range(1, n + 1))

class SeedIVDataset(Dataset):
    def __init__(self, cfg, subjects=None, mode='train'):
        self.cfg = cfg
        self.eeg_dir = cfg.get('eeg_dir')
        self.eye_dir = cfg.get('eye_dir')
        self.label_dir = cfg.get('label_dir')
        self.eye_dim = int(cfg.get('eye_dim', 24))
        self.use_key_prefix = cfg.get('use_key_prefix', 'de_movingAve')
        self.sessions_per_subject = int(cfg.get('sessions_per_subject', 3))
        self.trials_per_session = int(cfg.get('trials_per_session', 24))
        self.subjects = subjects
        self.mode = mode

        assert os.path.isdir(self.eeg_dir), f"EEG dir not found: {self.eeg_dir}"
        assert os.path.isdir(self.eye_dir), f"EYE dir not found: {self.eye_dir}"
        assert os.path.isdir(self.label_dir), f"Label dir not found: {self.label_dir}"

        self.eegs, self.eyes, self.labels = [], [], []
        self._build()

    def _build(self):
        eeg_files = sorted(glob.glob(os.path.join(self.eeg_dir, '*.mat')))
        eye_files = sorted(glob.glob(os.path.join(self.eye_dir, '*.mat')))
        assert len(eeg_files) % self.sessions_per_subject == 0, f"EEG files not divisible by {self.sessions_per_subject}"
        assert len(eye_files) % self.sessions_per_subject == 0, f"EYE files not divisible by {self.sessions_per_subject}"
        n_subjects_total = len(eeg_files) // self.sessions_per_subject

        if self.subjects is None:
            self.subjects = list(range(1, n_subjects_total + 1))

        l1, l2, l3 = _load_all_label_rows(self.label_dir, self.trials_per_session, n_subjects_total)
        max_sid = min(n_subjects_total, l1.shape[0])

        for sid in self.subjects:
            if not (1 <= sid <= max_sid):
                raise ValueError(f"Subject id {sid} out of range 1..{max_sid}")
            base = (sid - 1) * self.sessions_per_subject
            subj_eeg_files = eeg_files[base:base + self.sessions_per_subject]
            subj_eye_files = eye_files[base:base + self.sessions_per_subject]
            subj_labels = np.concatenate([l1[sid - 1], l2[sid - 1], l3[sid - 1]], axis=0).astype(int)  # (72,)

            for s in range(self.sessions_per_subject):
                eeg_mat = sio.loadmat(subj_eeg_files[s], verify_compressed_data_integrity=False)
                eye_mat = sio.loadmat(subj_eye_files[s], verify_compressed_data_integrity=False)

                eeg_keys = _match_eeg_keys(eeg_mat, self.use_key_prefix, self.trials_per_session)
                eye_keys = _match_eye_keys(eye_mat, self.trials_per_session)

                if len(eeg_keys) < self.trials_per_session:
                    raise ValueError(f"EEG file {os.path.basename(subj_eeg_files[s])} trials {len(eeg_keys)} < {self.trials_per_session}")
                if len(eye_keys) < self.trials_per_session:
                    raise ValueError(f"EYE file {os.path.basename(subj_eye_files[s])} trials {len(eye_keys)} < {self.trials_per_session}")

                for t in range(self.trials_per_session):
                    eeg_arr = eeg_mat[eeg_keys[t]]
                    eye_arr = eye_mat[eye_keys[t]]
                    eeg_T52 = _unify_eeg_trial(eeg_arr)            # (T,5,62)
                    eye_TD  = _unify_eye_trial(eye_arr, self.eye_dim)  # (T,D)
                    T = min(eeg_T52.shape[0], eye_TD.shape[0])
                    if T <= 0:
                        continue
                    self.eegs.append(eeg_T52[:T])
                    self.eyes.append(eye_TD[:T])
                    self.labels.append(int(subj_labels[s * self.trials_per_session + t]))

        if not (len(self.labels) == len(self.eegs) == len(self.eyes)):
            raise AssertionError(f"样本数量不一致：EEG{len(self.eegs)} EYE{len(self.eyes)} Label{len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg = self.eegs[idx]
        eye = self.eyes[idx]
        label = self.labels[idx]
        T = eeg.shape[0]
        mask = torch.ones(T, dtype=torch.bool)
        return {'eeg': eeg, 'eye': eye, 'label': label, 'length': T, 'mask': mask}

def collate_fn(batch):
    B = len(batch)
    lengths = torch.tensor([b['length'] for b in batch], dtype=torch.long)
    maxT = int(lengths.max().item())
    D_eye = batch[0]['eye'].shape[1]
    eegs = torch.zeros(B, maxT, 62, 5, dtype=torch.float32)
    eyes = torch.zeros(B, maxT, D_eye, dtype=torch.float32)
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    mask = torch.zeros(B, maxT, dtype=torch.bool)
    for i, item in enumerate(batch):
        T = item['eeg'].shape[0]
        eegs[i, :T] = item['eeg'].permute(0, 2, 1)  # (T,5,62)->(T,62,5)
        eyes[i, :T] = item['eye']
        mask[i, :T] = True
    eegs = _nan_guard_(eegs)
    eyes = _nan_guard_(eyes)
    return eegs, eyes, labels, lengths, mask

def build_loaders_by_subject(cfg, train_subjects, test_subjects, batch_size=32, num_workers=0, shuffle=True):
    train_ds = SeedIVDataset(cfg, subjects=train_subjects, mode='train')
    test_ds  = SeedIVDataset(cfg, subjects=test_subjects, mode='test')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, drop_last=False)
    return train_loader, test_loader
