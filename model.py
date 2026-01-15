# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----------------------------
# BandFuser
# ----------------------------
class BandFuser(nn.Module):
    """
    BandFuser: 把 (62通道 × 5频段) 转换为一个统一的表示
    - band_fusion=True: 每个通道内 5 个频段通过 MLP 融合到 band_embed_dim，再拼接所有通道
    - band_fusion=False: 直接把所有频段展平
    """
    def __init__(self, n_chan=62, n_band=5, band_embed=4, band_fusion=True):
        super().__init__()
        self.band_fusion = band_fusion
        if band_fusion:
            self.per_band = nn.Sequential(
                nn.Linear(n_band, band_embed),
                nn.ReLU(),
                nn.Linear(band_embed, band_embed)
            )
            self.out_dim = n_chan * band_embed
        else:
            self.out_dim = n_chan * n_band

    def forward(self, x):
        # x: (B, T, 62, 5)
        B, T, C, Bf = x.shape
        if self.band_fusion:
            flat = x.view(B*T*C, Bf)          # (B*T*C,5)
            out = self.per_band(flat)         # (B*T*C, band_embed)
            out = out.view(B, T, C, -1)       # (B,T,62,band_embed)
            out = out.view(B, T, -1)          # (B,T,62*band_embed)
            return out
        else:
            return x.view(B, T, -1)           # (B,T,62*5)

# ----------------------------
# Positional Encoding (for Transformer)
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos*div_term)
        pe[:, 1::2] = torch.cos(pos*div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,max_len,d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ----------------------------
# Transformer-based time encoder
# ----------------------------
class TransformerTimeEncoder(nn.Module):
    def __init__(self, in_dim, d_model=256, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_dim = d_model

    def forward(self, x, mask=None):
        # x: (B,T,D)
        x = self.in_proj(x)
        x = self.pos(x)
        x_t = x.permute(1,0,2)  # (T,B,D)
        if mask is not None:
            key_padding_mask = ~mask  # (B,T), True=pad
            out = self.encoder(x_t, src_key_padding_mask=key_padding_mask)
        else:
            out = self.encoder(x_t)
        return out.permute(1,0,2)  # (B,T,D)

# ----------------------------
# CNN-based time encoder
# ----------------------------
class TimeCNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dims=[128,256], kernel=3):
        super().__init__()
        layers, cur = [], in_dim
        for h in hidden_dims:
            layers += [
                nn.Conv1d(cur, h, kernel_size=kernel, padding=kernel//2),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ]
            cur = h
        self.net = nn.Sequential(*layers)
        self.out_dim = cur

    def forward(self, x, mask=None):
        # x: (B,T,D) -> (B,D,T)
        x = x.permute(0,2,1)
        out = self.net(x)
        return out.permute(0,2,1)  # (B,T',D_out)

# ----------------------------
# FiLM Block
# ----------------------------
class FiLMBlock(nn.Module):
    """
    从 EEG 特征生成 gamma, beta，对 Eye 特征做调制
    """
    def __init__(self, d_eeg, d_eye, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_eeg, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2*d_eye)
        )
        self.d_eye = d_eye

    def forward(self, eeg_feats):
        B,T,D = eeg_feats.shape
        gb = self.net(eeg_feats.reshape(B*T,D)).view(B,T,2*self.d_eye)
        gamma, beta = torch.split(gb, self.d_eye, dim=-1)
        return gamma, beta

# ----------------------------
# FR Module (Feature Reweighting)
# ----------------------------
class FRModule(nn.Module):
    def __init__(self, feat_dim, n_heads=4, d_k=32):
        super().__init__()
        self.n_heads, self.d_k = n_heads, d_k
        self.total = n_heads*d_k
        self.proj_q = nn.Linear(feat_dim, self.total)
        self.proj_k = nn.Linear(feat_dim, self.total)
        self.proj_v = nn.Linear(feat_dim, self.total)
        self.out = nn.Linear(self.total, feat_dim)

    def forward(self, x, mask=None):
        B,T,D = x.shape
        q = self.proj_q(x).view(B,T,self.n_heads,self.d_k).permute(0,2,1,3)
        k = self.proj_k(x).view(B,T,self.n_heads,self.d_k).permute(0,2,1,3)
        v = self.proj_v(x).view(B,T,self.n_heads,self.d_k).permute(0,2,1,3)
        scores = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.d_k)  # (B,h,T,T)
        if mask is not None:
            key_mask = ~mask.unsqueeze(1).unsqueeze(1)  # (B,1,1,T)
            scores = scores.masked_fill(key_mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).permute(0,2,1,3).contiguous().view(B,T,-1)
        return self.out(out)

# ----------------------------
# 主模型
# ----------------------------

class DE_FG_FR_Model(nn.Module):
    def __init__(self, cfg, n_chan=62, n_band=5, eye_dim=None, num_classes=7):
        super().__init__()
        self.cfg = cfg
        if eye_dim is None:
            eye_dim = int(cfg.get('eye_dim', 33))
        # ---- EEG 频段融合 ----
        self.bandfuser = BandFuser(n_chan, n_band,
                                   band_embed=cfg['band_embed_dim'],
                                   band_fusion=cfg['band_fusion'])
        d_eeg_in = self.bandfuser.out_dim

        # ---- Eye 线性投影 ----
        self.eye_proj = nn.Linear(eye_dim, cfg['d_model']//4)

        # ---- 时间编码器：Transformer / TimeCNN / SSM / Hyena ----
        enc_type = cfg.get('encoder_type', 'transformer')
        if enc_type == 'transformer':
            self.eeg_encoder = TransformerTimeEncoder(d_eeg_in, cfg['d_model'], cfg['n_heads'], cfg['n_layers'], cfg['dropout'])
            self.eye_encoder = TransformerTimeEncoder(self.eye_proj.out_features,
                                                      max(1, cfg['d_model']//1), max(1, cfg['n_heads']//2), max(1, cfg['n_layers']), cfg['dropout'])
        elif enc_type == 'timecnn':
            self.eeg_encoder = TimeCNNEncoder(d_eeg_in, cfg['timecnn_hidden'])
            self.eye_encoder = TimeCNNEncoder(self.eye_proj.out_features, [h//2 for h in cfg['timecnn_hidden']])
        elif enc_type == 'ssm':
            self.eeg_encoder = SSMTimeEncoder(d_eeg_in, cfg['d_model'], max(1, cfg['n_layers']), cfg['dropout'])
            self.eye_encoder = SSMTimeEncoder(self.eye_proj.out_features, max(1, cfg['d_model']//1), max(1, cfg['n_layers']), cfg['dropout'])
        elif enc_type == 'hyena':
            self.eeg_encoder = HyenaTimeEncoder(d_eeg_in, cfg['d_model'], max(1, cfg['n_layers']), cfg['dropout'])
            self.eye_encoder = HyenaTimeEncoder(self.eye_proj.out_features, max(1, cfg['d_model']//1), max(1, cfg['n_layers']), cfg['dropout'])
        else:
            raise ValueError(f"Unknown encoder_type: {enc_type}")

        d_eeg = self.eeg_encoder.out_dim
        d_eye = self.eye_encoder.out_dim

        # ---- FiLM & FR ----
        self.film = FiLMBlock(d_eeg, d_eye)
        self.fr = FRModule(d_eye, cfg['fr_heads'], cfg['fr_dk'])

        # ---- T-CMC 对比头（可选）----
        if cfg.get('tcmc_enabled', False):
            self.tcmc = TemporalContrastHead(proj_dim=min(d_eeg, d_eye),
                                             temp=cfg.get('tcmc_temp', 0.07),
                                             pos_window=cfg.get('tcmc_pos_window', 0))
            # 若d不一致，额外投影到相同维度
            self.align_eeg = nn.Linear(d_eeg, min(d_eeg, d_eye), bias=False)
            self.align_eye = nn.Linear(d_eye, min(d_eeg, d_eye), bias=False)
        else:
            self.tcmc = None

        # ---- StyleGate + 分类器 ----
        fusion_dim = d_eeg + d_eye
        self.style_gate = StyleGate(fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, cfg['mlp_dim']),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(cfg['mlp_dim'], num_classes)
        )

    def forward(self, eeg_seq, eye_seq, lengths, mask):
        # 1) 频段融合 & 眼动投影
        eeg_feat = self.bandfuser(eeg_seq)      # (B,T,D_eeg_in)
        eye_feat = self.eye_proj(eye_seq)       # (B,T,D_eye_in)

        # 2) 时间编码
        eeg_enc = self.eeg_encoder(eeg_feat, mask)  # (B,T,D_eeg)
        eye_enc = self.eye_encoder(eye_feat, mask)  # (B,T,D_eye)

        # 3) 时间对齐（若T不同）
        if eeg_enc.size(1) != eye_enc.size(1):
            minT = min(eeg_enc.size(1), eye_enc.size(1))
            eeg_enc, eye_enc, mask = eeg_enc[:, :minT], eye_enc[:, :minT], mask[:, :minT]

        aux = {}
        # 4) 可选：逐时刻跨模态对比损失（训练用）
        if self.training and self.tcmc is not None:
            z_eeg = self.align_eeg(eeg_enc)
            z_eye = self.align_eye(eye_enc)
            aux['tcmc_loss'] = self.tcmc(z_eeg, z_eye, mask)

        # 5) FiLM 调制 + FR
        gamma, beta = self.film(eeg_enc)
        if gamma.size(-1) != eye_enc.size(-1):
            proj = nn.Linear(gamma.size(-1), eye_enc.size(-1)).to(eye_enc.device)
            gamma, beta = proj(gamma), proj(beta)
        eye_mod = gamma * eye_enc + beta
        eye_fr = self.fr(eye_mod, mask)

        # 6) 池化（mask加权平均）
        mask_f = mask.float().unsqueeze(-1)
        eeg_pool = (eeg_enc * mask_f).sum(1) / (mask_f.sum(1) + 1e-8)
        eye_pool = (eye_fr * mask_f).sum(1) / (mask_f.sum(1) + 1e-8)

        # 7) StyleGate + 分类头
        z = torch.cat([eeg_pool, eye_pool], dim=-1)
        z = self.style_gate(z)
        logits = self.classifier(z)

        if self.training:
            return logits, aux
        return logits

        # 时间编码器
        if cfg['encoder_type']=="transformer":
            self.eeg_encoder = TransformerTimeEncoder(d_eeg_in, cfg['d_model'], cfg['n_heads'], cfg['n_layers'], cfg['dropout'])
            self.eye_encoder = TransformerTimeEncoder(self.eye_proj.out_features, cfg['d_model']//2, max(1,cfg['n_heads']//2), max(1,cfg['n_layers']), cfg['dropout'])
        else:
            self.eeg_encoder = TimeCNNEncoder(d_eeg_in, cfg['timecnn_hidden'])
            self.eye_encoder = TimeCNNEncoder(self.eye_proj.out_features, [h//2 for h in cfg['timecnn_hidden']])

        # FiLM
        self.film = FiLMBlock(self.eeg_encoder.out_dim, self.eye_encoder.out_dim)
        # FR
        self.fr = FRModule(self.eye_encoder.out_dim, cfg['fr_heads'], cfg['fr_dk'])

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.eeg_encoder.out_dim+self.eye_encoder.out_dim, cfg['mlp_dim']),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(cfg['mlp_dim'], num_classes)
        )

    def forward(self, eeg_seq, eye_seq, lengths, mask):
        # EEG band fusion
        eeg_feat = self.bandfuser(eeg_seq)         # (B,T,D_eeg_in)
        eye_feat = self.eye_proj(eye_seq)          # (B,T,D_eye_in)
        eeg_enc = self.eeg_encoder(eeg_feat,mask)  # (B,T,D_eeg)
        eye_enc = self.eye_encoder(eye_feat,mask)  # (B,T,D_eye)

        # 对齐时间长度
        if eeg_enc.size(1)!=eye_enc.size(1):
            minT=min(eeg_enc.size(1),eye_enc.size(1))
            eeg_enc,eye_enc,mask=eeg_enc[:,:minT],eye_enc[:,:minT],mask[:,:minT]

        gamma,beta=self.film(eeg_enc)
        if gamma.size(-1)!=eye_enc.size(-1):
            proj=nn.Linear(gamma.size(-1),eye_enc.size(-1)).to(eye_enc.device)
            gamma,beta=proj(gamma),proj(beta)
        eye_mod=gamma*eye_enc+beta
        eye_fr=self.fr(eye_mod,mask)

        mask_f=mask.float().unsqueeze(-1)
        eeg_pool=(eeg_enc*mask_f).sum(1)/(mask_f.sum(1)+1e-8)
        eye_pool=(eye_fr*mask_f).sum(1)/(mask_f.sum(1)+1e-8)

        return self.classifier(torch.cat([eeg_pool,eye_pool],dim=-1))


# ----------------------------
# Tiny SSM / Hyena-like Encoders (轻量实现，接口与Transformer一致)
# ----------------------------
class SSMLayerTiny(nn.Module):
    """
    一个极简的SSM近似：用可学习一维卷积核去逼近长程依赖（深度可堆叠）
    注意：这是工程上可跑的轻量替代，便于快速集成与对比。
    """
    def __init__(self, d_model, kernel_size=25, dropout=0.1):
        super().__init__()
        padding = kernel_size//2
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, groups=d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):  # x: (B,T,D)
        res = x
        x = x.transpose(1,2)                 # (B,D,T)
        x = self.conv(x)
        x = x.transpose(1,2)                 # (B,T,D)
        x = torch.tanh(self.gate(x)) * x     # gated residual
        x = self.dropout(x)
        return self.norm(x + res)

class SSMTimeEncoder(nn.Module):
    def __init__(self, in_dim, d_model=256, n_layers=3, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList([SSMLayerTiny(d_model, kernel_size=25, dropout=dropout) for _ in range(n_layers)])
        self.out_dim = d_model
    def forward(self, x, mask=None):
        x = self.in_proj(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class HyenaBlockTiny(nn.Module):
    """
    极简Hyena风格：使用多尺度空洞卷积近似长卷积核，带残差与归一化。
    """
    def __init__(self, d_model, dilations=(1,2,4,8), dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3*d, dilation=d, groups=d_model)
            for d in dilations
        ])
        self.mix = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):  # x: (B,T,D)
        res = x
        h = x.transpose(1,2)  # (B,D,T)
        out = 0
        for conv in self.convs:
            out = out + conv(h)
        out = out.transpose(1,2)      # (B,T,D)
        out = self.mix(out)
        out = torch.relu(out)
        out = self.dropout(out)
        return self.norm(out + res)

class HyenaTimeEncoder(nn.Module):
    def __init__(self, in_dim, d_model=256, n_layers=3, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList([HyenaBlockTiny(d_model, dropout=dropout) for _ in range(n_layers)])
        self.out_dim = d_model
    def forward(self, x, mask=None):
        x = self.in_proj(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

# ----------------------------
# StyleGate：用于TENT-Style 的轻量风格门控
# ----------------------------
class StyleGate(nn.Module):
    """
    对融合后的向量 z 施加逐通道的仿射变换：y = alpha * z + beta
    - alpha, beta 初始化为 1, 0；训练时与分类头一同学习
    - 测试时自适应（TENT-Style）只解冻 alpha/beta
    """
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))
    def forward(self, z):
        return z * self.alpha + self.beta

# ----------------------------
# T-CMC 逐时刻跨模态对比头
# ----------------------------
class TemporalContrastHead(nn.Module):
    def __init__(self, proj_dim, temp=0.07, pos_window=0):
        super().__init__()
        self.proj_eeg = nn.Linear(proj_dim, proj_dim, bias=False)
        self.proj_eye = nn.Linear(proj_dim, proj_dim, bias=False)
        self.temp = temp
        self.pos_window = pos_window

    def forward(self, eeg_enc, eye_enc, mask):
        """
        eeg_enc, eye_enc: (B,T,D) 经过时间编码后的表示
        mask: (B,T)      有效时间步
        返回：InfoNCE 对比损失 (标量)
        """
        B,T,D = eeg_enc.size()
        eeg = F.normalize(self.proj_eeg(eeg_enc), dim=-1)
        eye = F.normalize(self.proj_eye(eye_enc), dim=-1)

        # 相似度：逐batch独立计算，避免跨样本泄露
        loss = 0.0
        eps = 1e-8
        for b in range(B):
            m = mask[b]  # (T,)
            valid_idx = torch.nonzero(m, as_tuple=False).squeeze(-1)
            if valid_idx.numel() < 2:
                continue
            e = eeg[b, valid_idx]  # (Tv,D)
            v = eye[b, valid_idx]  # (Tv,D)
            # sims: (Tv,Tv)
            sims = torch.matmul(e, v.transpose(0,1)) / self.temp
            # 正样本：主对角线及 +/- pos_window
            Tv = sims.size(0)
            pos_mask = torch.eye(Tv, dtype=torch.bool, device=sims.device)
            if self.pos_window > 0:
                for w in range(1, self.pos_window+1):
                    pos_mask |= torch.diag(torch.ones(Tv-w, dtype=torch.bool, device=sims.device), diagonal=w)
                    pos_mask |= torch.diag(torch.ones(Tv-w, dtype=torch.bool, device=sims.device), diagonal=-w)
            neg_mask = ~pos_mask
            # InfoNCE：每一行以对应正样本作为分子，其他为负样本
            pos_scores = sims[pos_mask].view(Tv, -1).max(dim=1).values  # 取窗口内最高作为正
            # 对数分母：log(sum(exp(·)))，包含正与负
            logsumexp = torch.logsumexp(sims, dim=1)
            loss_b = -(pos_scores - logsumexp).mean()
            loss = loss + loss_b
        return loss / (B + eps)

