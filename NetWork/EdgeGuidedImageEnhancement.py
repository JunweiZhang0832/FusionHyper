
import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeExtractor(nn.Module):
    def __init__(self, mode='sobel'):
        super().__init__()
        if mode == 'sobel':
            kernel_x = torch.tensor([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]], dtype=torch.float32)
            kernel_y = torch.tensor([[-1, -2, -1],
                                     [0,  0,  0],
                                     [1,  2,  1]], dtype=torch.float32)
            self.weight_x = nn.Parameter(kernel_x.unsqueeze(0).unsqueeze(0), requires_grad=False)
            self.weight_y = nn.Parameter(kernel_y.unsqueeze(0).unsqueeze(0), requires_grad=False)
        elif mode == 'laplacian':
            kernel = torch.tensor([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]], dtype=torch.float32)
            self.weight = nn.Parameter(kernel.unsqueeze(0).unsqueeze(0), requires_grad=False)
        else:
            raise ValueError("mode must be 'sobel' or 'laplacian'")
        self.mode = mode

    def forward(self, x):
        # x: (B, 3, H, W)
        gray = x.mean(dim=1, keepdim=True)
        if self.mode == 'sobel':
            edge_x = F.conv2d(gray, self.weight_x, padding=1)
            edge_y = F.conv2d(gray, self.weight_y, padding=1)
            edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        else:
            edge = F.conv2d(gray, self.weight, padding=1)
        # 边缘强度标准化
        edge = torch.sigmoid(edge)
        return edge  # (B,1,H,W)


class EdgeGuidedEB(nn.Module):
    def __init__(self, channels=3, hidden_dim=64, edge_mode='sobel'):
        super().__init__()
        self.edge_extractor = EdgeExtractor(mode=edge_mode)
        self.conv_in = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
        )
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Conv2d(hidden_dim * 2, channels, kernel_size=3, padding=1)
        self.res_scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        feat_main = self.conv_in(x)
        low_filter = torch.ones((3, 1, 3, 3), device=x.device) / 9.0
        low = F.conv2d(x, low_filter, stride=1, padding=1, groups=3)
        edge_map = self.edge_extractor(x)
        feat_edge = self.edge_conv(edge_map)
        fused = torch.cat([feat_main, feat_edge], dim=1)
        out = self.fuse(fused) + self.alpha * low
        return x + self.res_scale * out


# if __name__ == "__main__":
#     x = torch.randn(1, 3, 224, 224).cuda()
#     model = EdgeGuidedEB(edge_mode='sobel').cuda()
#     y = model(x)
#     print("Input:", x.shape)
#     print("Output:", y.shape)
