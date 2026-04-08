import torch
import torch.nn as nn
import torch.nn.functional as F
def deform_sample(x, offset):
    B, C, H, W = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device),
        torch.linspace(-1, 1, W, device=x.device),
        indexing='ij'
    )
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    grid = grid + offset
    return F.grid_sample(x, grid, align_corners=True)

# SCMC spatial and channel mixing Convolution
class SCMC(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.spatial = nn.Conv2d(ch, ch, 5, 1, 2, groups=ch)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch, 1),
            nn.Mish(),
        )

    def forward(self, x):
        spatial = self.spatial(x)
        channel = self.channel(x)
        return spatial + channel

# SEDC Scale-Enhanced Deformable Convolution
class SEDCBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.PWConv = nn.Conv2d(in_ch, in_ch, 1, 1, 0)
        self.SCMC = SCMC(in_ch // 4)
        self.offset = nn.Conv2d(in_ch // 4, 2, 3, 1, 1)
        self.conv1 = nn.Conv2d(in_ch // 4, out_ch // 4, 3, 1, 1)

    def forward(self, x):
        # 生成2通道的偏移量
        x = self.PWConv(F.layer_norm(x, x.shape[1:]))
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        off1 = self.offset(x1).permute(0, 2, 3, 1)
        x_def1 = deform_sample(x1, off1)
        out1 = self.conv1(x_def1)
        out1 = self.SCMC(out1)

        off2 = self.offset(x2).permute(0, 2, 3, 1)
        x_def2 = deform_sample(x2, off2)
        out2 = self.conv1(x_def2)
        out2 = self.SCMC(out2)

        off3 = self.offset(x3).permute(0, 2, 3, 1)
        x_def3 = deform_sample(x3, off3)
        out3 = self.conv1(x_def3)
        out3 = self.SCMC(out3)

        off4 = self.offset(x4).permute(0, 2, 3, 1)
        x_def4 = deform_sample(x4, off4)
        out4 = self.conv1(x_def4)
        out4 = self.SCMC(out4)

        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.PWConv(out) + x
        out = F.layer_norm(out, out.shape[1:])
        return out


class FeedForwardNet(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1),
        )

    def forward(self, x):
        return self.net(x)

# SADA Scale-Adaptive Deformable Transformer
class SADA(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        # 1×1卷积映射到3倍维度，分别用于Query,key,value
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        # 3×3卷积，生成每个注意力头的偏移量（每个头2个通道）
        self.offset = nn.Conv2d(dim, 2 * heads, 3, 1, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.FFN = FeedForwardNet(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # 先进行LN
        x0 = F.layer_norm(x, x.shape[1:])
        # 生成Query,Key,Value
        q, k, v = self.qkv(x0).chunk(3, dim=1)
        q = q.reshape(B, self.heads, C // self.heads, H, W)
        k = k.reshape(B, self.heads, C // self.heads, H, W)
        v = v.reshape(B, self.heads, C // self.heads, H, W)
        off = self.offset(x).reshape(B, self.heads, 2, H, W).permute(0, 1, 3, 4, 2)
        out_heads = []
        for h in range(self.heads):
            k_s = deform_sample(k[:, h], off[:, h])
            v_s = deform_sample(v[:, h], off[:, h])
            q_f = F.elu(q[:, h]) + 1  # 保证正值，稳定
            k_f = F.elu(k_s) + 1
            kv = torch.einsum("bchw, bchw -> bc", k_f, v_s)

            k_sum = k_f.sum(dim=(2, 3))  # (B, C')
            den = torch.einsum("bchw, bc -> bchw", q_f, k_sum) + 1e-8
            out = torch.einsum("bchw, bc -> bchw", q_f, kv) / den
            # ----------  替换结束  ----------
            out_heads.append(out)
        out = torch.cat(out_heads, dim=1)
        out = self.proj(out) + x
        out = F.layer_norm(out, out.shape[1:])
        out = self.FFN(out) + out
        return out

