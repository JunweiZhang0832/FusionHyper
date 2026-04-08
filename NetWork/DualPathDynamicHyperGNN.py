import torch
import torch.nn as nn

def img_to_patches(x, patch_size):
    B, C, H, W = x.shape
    orig_H, orig_W = H, W
    if H % patch_size != 0 or W % patch_size != 0:
        new_H = (H // patch_size) * patch_size
        new_W = (W // patch_size) * patch_size
        x = x[:, :, :new_H, :new_W]
        H, W = new_H, new_W

    ph = H // patch_size
    pw = W // patch_size
    x = x.view(B, C, ph, patch_size, pw, patch_size)
    patches = x.mean(dim=(3, 5)).view(B, C, ph * pw).permute(0, 2, 1)  # [B, N, C]
    mapping = {'ph': ph, 'pw': pw, 'patch_size': patch_size, 'H': orig_H, 'W': orig_W}
    return patches, mapping

def patches_to_map(nodes, mapping):
    B, N, D = nodes.shape
    ph, pw, ps, H, W = mapping['ph'], mapping['pw'], mapping['patch_size'], mapping['H'], mapping['W']
    x = nodes.permute(0, 2, 1).view(B, D, ph, pw)
    x = x.repeat_interleave(ps, dim=2).repeat_interleave(ps, dim=3)
    if x.shape[2] != H or x.shape[3] != W:
        x = torch.nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
    return x


def build_incidence_sparse(node_feats, ph, pw, topk=9, R=3):
    device = node_feats.device
    N, D = node_feats.shape
    nf = node_feats.float()
    nf = nf / (nf.norm(dim=1, keepdim=True) + 1e-8)
    sim = torch.matmul(nf, nf.t())  # [N,N]
    k = min(topk, N)
    _, topk_idx = torch.topk(sim, k=k, dim=-1, largest=True, sorted=False)
    nodes_cluster = topk_idx.reshape(-1).to(device)
    edge_idx_cluster = (torch.arange(N, device=device).unsqueeze(1).expand(N, k).reshape(-1))

    centers_y = torch.arange(ph, device=device).unsqueeze(1).repeat(1, pw).reshape(-1)  # [N]
    centers_x = torch.arange(pw, device=device).repeat(ph).reshape(-1)  # [N]
    N = centers_y.numel()
    nodes_dilate_list = []
    edge_idx_dilate_list = []
    for r in range(1, R + 1):
        offsets = torch.arange(-r, r + 1, device=device)
        off_y, off_x = torch.meshgrid(offsets, offsets, indexing='ij')
        off_y = off_y.reshape(-1)  # [K]
        off_x = off_x.reshape(-1)  # [K]

        d_inf = torch.maximum(off_y.abs(), off_x.abs())  # [K]
        if r == 1:
            ring_mask = (d_inf <= 1)
        else:
            ring_mask = (d_inf == r)

        center_mask = (off_y == 0) & (off_x == 0)
        ring_mask = ring_mask | center_mask  # [K]

        nbr_y = centers_y.unsqueeze(1) + off_y.unsqueeze(0)  # [N, K]
        nbr_x = centers_x.unsqueeze(1) + off_x.unsqueeze(0)  # [N, K]

        valid = (
                (nbr_y >= 0) & (nbr_y < ph) &
                (nbr_x >= 0) & (nbr_x < pw) &
                ring_mask.unsqueeze(0)
        )  # [N, K]

        if not valid.any():
            continue

        nbr_idx = nbr_y * pw + nbr_x  # [N, K]

        valid_flat = valid.reshape(-1)
        nodes_flat = nbr_idx.reshape(-1)[valid_flat]

        center_repeat = torch.arange(N, device=device).repeat_interleave(
            valid.sum(dim=1)
        )

        base_edge = (r - 1) * N
        edge_ids = base_edge + center_repeat

        nodes_dilate_list.append(nodes_flat)
        edge_idx_dilate_list.append(edge_ids)

    if len(nodes_dilate_list) > 0:
        nodes_dilate = torch.cat(nodes_dilate_list, dim=0)
        edge_idx_dilate = torch.cat(edge_idx_dilate_list, dim=0)
    else:
        nodes_dilate = torch.empty(0, dtype=torch.long, device=device)
        edge_idx_dilate = torch.empty(0, dtype=torch.long, device=device)

    nodes_all = torch.cat([nodes_cluster.to(device), nodes_dilate], dim=0).to(torch.long)
    edge_idx_all = torch.cat([edge_idx_cluster.to(device), edge_idx_dilate], dim=0).to(torch.long)
    L = nodes_all.shape[0]
    indices = torch.stack([nodes_all, edge_idx_all], dim=0)  # [2, L]
    values = torch.ones((L,), device=device, dtype=torch.float32)  # use float32 values
    M = N + R * N
    H = torch.sparse_coo_tensor(indices, values, (N, M), device=device).coalesce()
    return H, M

class HypergraphSpectralConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.theta = nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, X, H):
        device = X.device
        N = X.shape[0]
        H32 = torch.sparse_coo_tensor(H.indices(), H.values().to(torch.float32), H.shape, device=device).coalesce()
        ones_N = torch.ones((N, 1), device=device, dtype=torch.float32)
        De = torch.sparse.mm(H32.t(), ones_N).squeeze(1) + 1e-8    # [M] float32
        ones_M = torch.ones((H32.shape[1],1), device=device, dtype=torch.float32)
        Dv = torch.sparse.mm(H32, ones_M).squeeze(1) + 1e-8       # [N] float32
        Dv_inv_sqrt = (Dv ** -0.5).unsqueeze(1)                  # [N,1] float32
        X32 = X.float()
        S = Dv_inv_sqrt * X32                                     # [N, D] float32
        T = torch.sparse.mm(H32.t(), S)                           # [M, D] float32
        U = T / De.unsqueeze(1)                                   # [M, D] float32
        V = torch.sparse.mm(H32, U)                               # [N, D] float32
        Y_mid32 = Dv_inv_sqrt * V                                 # [N, D] float32
        Y_mid = Y_mid32.to(X.dtype)                                  # if X is fp16 under autocast, Y_mid becomes fp16
        Y = self.theta(Y_mid)                                        # Linear will run under current autocast context
        # [N,out_dim]
        return Y

class ConvFFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pw1 = nn.Conv2d(dim, dim, 1)
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        res = x
        x = self.pw1(x)
        x = self.dw(x) + x
        x = self.act(x)
        x = self.pw2(x)
        return x + res

class MSDVHGNNBlock(nn.Module):
    def __init__(self, node_dim, out_dim):
        super().__init__()
        # 超图谱卷积层，执行节点特征的超图传播
        self.conv = HypergraphSpectralConv(in_dim=node_dim, out_dim=out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.res_proj = nn.Linear(node_dim, out_dim) if node_dim != out_dim else None

    def forward(self, node_feats, H):
        # 超图谱卷积，转换后的节点特征
        Y = self.conv(node_feats, H)   # [N, out_dim]
        # 如果输入输出维度相同: res = node_feats
        res = self.res_proj(node_feats) if self.res_proj is not None else node_feats
        # 将卷积结果与残差连接相加，并归一化
        out = self.ln(Y + res)
        return out

class DVHGNNModule(nn.Module):
    def __init__(self, in_ch=3, embed_dim=64, patch_size=8, topk=9, R=3,
                 num_blocks=2, hyper_out_dim=64, out_ch=32):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.topk = topk
        self.R = R
        self.num_blocks = num_blocks

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.ModuleList([MSDVHGNNBlock(node_dim=embed_dim, out_dim=hyper_out_dim) for _ in range(num_blocks)])
        self.node_proj = nn.Linear(hyper_out_dim, embed_dim) if hyper_out_dim != embed_dim else None
        self.convffn = ConvFFN(dim=embed_dim)
        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, out_ch, kernel_size=1)
        )

    def forward(self, x):
        B, C_in, H, W = x.shape
        device = x.device
        feat = self.stem(x)                                 # [B, embed_dim, H, W]
        patches, mapping = img_to_patches(feat, self.patch_size)  # [B, N, D]
        B, N, D = patches.shape
        out_nodes = torch.zeros((B, N, D), device=device, dtype=patches.dtype)

        for b in range(B):
            node_feats = patches[b]                        # [N, D] on device
            # Build sparse incidence H in a float32-safe way (indices + values)
            H_sparse, M = build_incidence_sparse(node_feats, mapping['ph'], mapping['pw'], topk=self.topk, R=self.R)
            # run block(s)
            nf = node_feats
            for blk in self.blocks:
                nf = blk(nf, H_sparse)
            if self.node_proj is not None:
                nf = self.node_proj(nf)
            out_nodes[b] = nf

        img_feat = patches_to_map(out_nodes, mapping)      # [B, D, H, W]
        img_feat = self.convffn(img_feat)
        out = self.out_conv(img_feat)
        return out

# if __name__ == "__main__":
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     m = DVHGNNModule(in_ch=3, embed_dim=64, patch_size=8, topk=9, R=3, num_blocks=2, hyper_out_dim=64, out_ch=24).to(device)
#     x = torch.randn(1, 3, 224, 224, device=device)
#     # run one forward under autocast (simulate AMP)
#
#     y = m(x)
#     print("out shape:", y.shape)   # expected (1,24,224,224)
