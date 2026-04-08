from thop import profile
from NetWork.DualPathDynamicHyperGNN import DVHGNNModule
from utils.CUDA_Check import GPUorCPU
from NetWork.SEDCAndSAD import SEDCBlock, SADA
from NetWork.EdgeGuidedImageEnhancement import EdgeGuidedEB
import warnings
warnings.filterwarnings("ignore", message="operator.*does not have profile information")
DEVICE = GPUorCPU().DEVICE
import torch
import torch.nn as nn

class SimAM(nn.Module):
    def __init__(self, lamda=1e-5):
        super().__init__()
        self.lamda = lamda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        # n 表示每个特征图的像素数减1
        n = h * w - 1
        # 计算通道内均值
        mean = torch.mean(x, dim=[-2, -1], keepdim=True)
        # 计算方差（去中心化后归一化）
        var = torch.sum(torch.pow((x - mean), 2), dim=[-2, -1], keepdim=True) / n
        # 计算 e_t，调制每个像素的权重（形式上类似于能量函数）
        e_t = torch.pow((x - mean), 2) / (4 * (var + self.lamda)) + 0.5
        # 使用 sigmoid 将 e_t 映射为 [0,1]，然后乘以原输入
        out = self.sigmoid(e_t) * x
        return out


class ELA(nn.Module):
    def __init__(self, channels, conv_kernel_size=7, gn_groups=6):
        super().__init__()

        self.channels = channels
        # 用手写的 pooling 代替 None-based AdaPool
        self.pool_h = lambda t: torch.mean(t, dim=3, keepdim=True)  # (B,C,H,1)
        self.pool_w = lambda t: torch.mean(t, dim=2, keepdim=True)  # (B,C,1,W)
        pad = (conv_kernel_size - 1) // 2
        # Conv1d 处理序列特征
        self.conv_h = nn.Conv1d(channels, channels, conv_kernel_size, padding=pad)
        self.conv_w = nn.Conv1d(channels, channels, conv_kernel_size, padding=pad)
        # GroupNorm
        self.gn_h = nn.GroupNorm(gn_groups, channels)
        self.gn_w = nn.GroupNorm(gn_groups, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        # 方向池化
        x_h = self.pool_h(x).squeeze(-1)  # (B,C,H)
        x_w = self.pool_w(x).squeeze(-2)  # (B,C,W)
        # Conv1d + GN
        h = self.sigmoid(self.gn_h(self.conv_h(x_h)))  # (B,C,H)
        w = self.sigmoid(self.gn_w(self.conv_w(x_w)))  # (B,C,W)
        # 恢复空间维度
        h = h.unsqueeze(-1).expand(-1, -1, -1, W)
        w = w.unsqueeze(-2).expand(-1, -1, H, -1)
        attn = h * w
        return x * attn


# out_dim=32
class feature_extraction(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(feature_extraction, self).__init__()
        self.at = nn.ReLU()
        # 对拼接后的通道进行归一化，注意输出通道数为 out_dim*3（3个卷积分支拼接）
        self.bn = nn.BatchNorm2d(out_dim * 3)
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_dim * 3 + in_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )
        # 使用不同尺寸和膨胀系数的卷积核提取多尺度特征
        self.conv3_3 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=2, dilation=2)
        self.conv5_5 = nn.Conv2d(in_dim, out_dim, kernel_size=5, padding=6, dilation=3)
        self.conv7_7 = nn.Conv2d(in_dim, out_dim, kernel_size=7, padding=12, dilation=4)
        self.conv5 = nn.Sequential(nn.Conv2d(in_dim, in_dim * 2, kernel_size=5, padding=2),
                                   nn.BatchNorm2d(in_dim * 2),
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(in_dim * 2, int((in_dim * 4)), kernel_size=7, padding=3),
                                   nn.BatchNorm2d(in_dim * 4),
                                   nn.ReLU())
        self.conv9 = nn.Sequential(nn.Conv2d(in_dim * 4, in_dim, kernel_size=9, padding=4),
                                   nn.BatchNorm2d(in_dim),
                                   nn.ReLU())
    def forward(self, x):
        # 分别计算三条路径的卷积结果
        conv3 = self.conv3_3(x)
        conv5 = self.conv5_5(x)
        conv7 = self.conv7_7(x)
        # 拼接三个卷积结果（在通道维度）
        out = torch.cat([conv3, conv5, conv7], dim=1)
        # 归一化并激活
        out = self.bn(out)
        out = self.at(out)
        #第二分支
        out2 = self.conv5(x)
        out2 = self.conv7(out2)
        out2 = self.conv9(out2) + x
        out = torch.cat([out, out2], dim=1)
        out = self.conv3(out)
        return out


class FEA_Decoder(nn.Module):
    def __init__(self, input_dim=72, out_dim=48):
        super().__init__()
        hidden_dim = input_dim
        self.ELA = ELA(108)
        self.block1 = nn.Sequential(
            nn.Conv2d(input_dim, int(hidden_dim * 1.5), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(hidden_dim * 1.5)),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(int(hidden_dim * 1.5), int(hidden_dim * 3), kernel_size=3, padding=2, stride=1, dilation=2, groups=int(hidden_dim * 1.5)),
            nn.BatchNorm2d(int(hidden_dim * 3)),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(int(hidden_dim * 3), out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        #residual = x
        x = self.block1(x)
        x = self.ELA(x)
        out = self.block2(x)
        out = self.block3(out)
        out = out
        return out


class GCN_Decoder(nn.Module):
    def __init__(self, input_dim=24):
        super().__init__()
        hidden_dim = input_dim
        #self.LSKA = LSKN2()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_dim, int(hidden_dim * 2), kernel_size=1),
            nn.BatchNorm2d(int(hidden_dim * 2)),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(int(hidden_dim * 2), int(hidden_dim * 2), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(hidden_dim * 2)),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(int(hidden_dim * 2), input_dim, kernel_size=1),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        residual = x
        x = self.block1(x)
        out = self.block2(x)
        #out = self.LSKA(out)
        out = self.block3(out)
        out = out + residual
        return out


class All_Decoder(nn.Module):
    def __init__(self, input_dim=72, out_dim=48):
        super().__init__()
        hidden_dim = input_dim
        self.SimAM = SimAM()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_dim, int(hidden_dim * 1.5), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(int(hidden_dim * 1.5), hidden_dim * 3, kernel_size=5, padding=4, dilation=2, groups=int(hidden_dim * 1.5)),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )

    def forward(self, gcn_result, fea_result):
        residual = fea_result
        x = torch.cat([gcn_result, fea_result], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.SimAM(x)
        out = self.block3(x)
        out = out + residual
        return out



class LSKN(nn.Module):
    def __init__(self):
        super().__init__()
        self.BN = nn.BatchNorm2d(72)
        self.relu = nn.ReLU()
        self.ELA = ELA(72)
        self.block = nn.Sequential(
                        nn.Conv2d(24, 72, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(72),
                        nn.ReLU()
                    )
        self.block3 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, groups=24),
            nn.Conv2d(24, 24, kernel_size=5, stride=1, padding=4, dilation=2, groups=24),
            nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=5, stride=1, padding=2, groups=24),
            nn.Conv2d(24, 24, kernel_size=7, stride=1, padding=6, dilation=2, groups=24),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=7, stride=1, padding=3, groups=24),
            nn.Conv2d(24, 24, kernel_size=9, stride=1, padding=8, dilation=2, groups=24),
            nn.Conv2d(24, 24, kernel_size=5, stride=1, padding=2),
        )

    def forward(self, x):
        x = self.block(x)
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
        x1 = self.block1(x1)
        x2 = x2 + x1
        x2 = self.block2(x2)
        x3 = x2 + x3
        x3 = self.block3(x3)
        x = torch.concat([x1, x2, x3], dim=1)
        #x = self.Conv(x)
        x = self.ELA(x)
        x = self.relu(self.BN(x))
        return x


class LSKN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.BN = nn.BatchNorm2d(72)
        self.relu = nn.ReLU()
        self.ELA = ELA(72)
        self.block = nn.Sequential(
                        nn.Conv2d(32, 72, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(72),
                        nn.ReLU()
                    )
        self.block3 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, groups=24),
            nn.Conv2d(24, 24, kernel_size=5, stride=1, padding=4, dilation=2, groups=24),
            nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=5, stride=1, padding=2, groups=24),
            nn.Conv2d(24, 24, kernel_size=7, stride=1, padding=6, dilation=2, groups=24),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=7, stride=1, padding=3, groups=24),
            nn.Conv2d(24, 24, kernel_size=9, stride=1, padding=8, dilation=2, groups=24),
            nn.Conv2d(24, 24, kernel_size=5, stride=1, padding=2),
        )

    def forward(self, x):
        x = self.block(x)
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
        x1 = self.block1(x1)
        x2 = x2 + x1
        x2 = self.block2(x2)
        x3 = x2 + x3
        x3 = self.block3(x3)
        x = torch.concat([x1, x2, x3], dim=1)
        #x = self.Conv(x)
        x = self.ELA(x)
        x = self.relu(self.BN(x))
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.EDGE = EdgeGuidedEB(edge_mode='sobel')
        self.HGNN = DVHGNNModule(in_ch=3, embed_dim=72, patch_size=8, topk=9, R=3, num_blocks=3, hyper_out_dim=72, out_ch=24)
        self.HGNNCoarse = DVHGNNModule(in_ch=3, embed_dim=72, patch_size=28, topk=5, R=3, num_blocks=3, hyper_out_dim=72, out_ch=24)
        self.HGNNMedium = DVHGNNModule(in_ch=3, embed_dim=72, patch_size=14, topk=7, R=3, num_blocks=3, hyper_out_dim=72, out_ch=24)
        self.MSFE = feature_extraction(in_dim=3, out_dim=24)
        self.SEDC1 = SEDCBlock(24, 24)
        self.SEDC2 = SEDCBlock(24, 24)
        self.SEDC0 = SEDCBlock(24, 24)
        self.SADATrans1 = SADA(24, 4)
        self.SADATrans2 = SADA(24, 4)
        self.SADATrans0 = SADA(24, 4)
        self.LSKN = LSKN()
        self.LSKN3 = LSKN3()

        self.gcn_mix = nn.Sequential(
            SimAM(),
            nn.Conv2d(48, 24, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )
        self.gcn_coarse_mix = nn.Sequential(
            SimAM(),
            nn.Conv2d(48, 24, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )
        self.gcn_medium_mix = nn.Sequential(
            SimAM(),
            nn.Conv2d(48, 24, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )
        self.fea_mix = nn.Sequential(
            SimAM(),
            nn.Conv2d(72 * 2, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.gcn_Decoder = GCN_Decoder()
        self.fea_Decoder = FEA_Decoder()
        self.allDecoder = All_Decoder()
        self.Reconstruction = nn.Sequential(
            nn.Conv2d(48, 1, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, A, B):
        # 28*28=784patch
        EA = self.EDGE(A)
        EB = self.EDGE(B)
        gcn_result_a = self.HGNN(EA)
        gcn_result_b = self.HGNN(EB)
        gcn_result = torch.cat([gcn_result_a, gcn_result_b], dim=1)
        gcn_result = self.gcn_mix(gcn_result)
        gcn_result = self.gcn_Decoder(gcn_result)

        # 8*8=64
        coarse_result_a = self.HGNNCoarse(EA)
        coarse_result_b = self.HGNNCoarse(EB)
        coarse_result = torch.cat([coarse_result_a, coarse_result_b], dim=1)
        coarse_result = self.gcn_coarse_mix(coarse_result)

        # 16*16=256
        medium_result_a = self.HGNNMedium(EA)
        medium_result_b = self.HGNNMedium(EB)
        medium_result = torch.cat([medium_result_a, medium_result_b], dim=1)
        medium_result = self.gcn_medium_mix(medium_result)

        fea_a = self.MSFE(EA)
        fea_a = fea_a + gcn_result_b
        fea_a = self.LSKN(fea_a)
        fea_b = self.MSFE(EB)
        fea_b = fea_b + gcn_result_a
        fea_b = self.LSKN(fea_b)
        fea = self.fea_mix(torch.cat([fea_a, fea_b], dim=1))
        fea_result0 = self.LSKN3(fea)#72
        fea1, fea2, fea3 = torch.chunk(fea_result0, 3, dim=1)

        fea_result11 = fea1 + coarse_result
        fea_result1 = self.SEDC0(fea_result11)  # 24
        fea_result1 = self.SADATrans0(fea_result1)
        fea_result1 = self.SEDC0(fea_result1)

        fea_result1 = self.SEDC0(fea_result1 + medium_result)  # 24
        fea_result1 = self.SADATrans0(fea_result1)
        fea_result1 = self.SEDC0(fea_result1)

        fea_result1 = self.SEDC0(fea_result1 + fea_result11)


        fea_result22 = fea2 + medium_result
        fea_result2 = self.SEDC1(fea_result22) # 24
        fea_result2 = self.SADATrans1(fea_result2)
        fea_result2 = self.SEDC1(fea_result2)

        fea_result2 = self.SEDC1(fea_result2 + coarse_result)   # 24
        fea_result2 = self.SADATrans1(fea_result2)
        fea_result2 = self.SEDC1(fea_result2)

        fea_result2 = self.SEDC1(fea_result2 + fea_result22)

        fea_result33 = self.SEDC2(fea3)
        fea_result3 = self.SADATrans2(fea_result33) # 24
        fea_result3 = self.SEDC2(fea_result3)

        fea_result3 = self.SEDC2(fea_result3 + gcn_result)
        fea_result3 = self.SADATrans2(fea_result3)  # 24
        fea_result3 = self.SEDC2(fea_result3)

        fea_result3 = self.SEDC2(fea_result3 + fea3)


        fea_result = torch.cat([fea_result1, fea_result2, fea_result3], dim=1) # 72

        fea_result = self.fea_Decoder(fea_result)

        result = self.allDecoder(gcn_result, fea_result)
        DM = self.Reconstruction(result)
        return DM


if __name__ == '__main__':
    test_tensor_A = torch.rand((1, 3, 224, 224)).to(DEVICE)
    test_tensor_B = torch.rand((1, 3, 224, 224)).to(DEVICE)
    model = Net().to(DEVICE)
    flops, params = profile(model, inputs=(test_tensor_A, test_tensor_B))
    print(f"FLOPs: {flops/1e9:.3f} G, Params: {params/1e6:.3f} M")
    DM = model(test_tensor_A, test_tensor_B)
    print("Decision Map:", DM.shape)