import numpy as np
import torch
from torchvision import transforms as T
from skimage.morphology import remove_small_objects, binary_opening, binary_closing
from utils.CUDA_Check import GPUorCPU
import torch.nn.functional as F
DEVICE = GPUorCPU.DEVICE

def Binarization(img_tensor):
    return torch.where(img_tensor > 0.5, 1., 0.)

# def RemoveSmallArea(img_tensor, size=None, threshold=0.001):
#     if size is None:
#         _, _, H, W = img_tensor.shape
#         size = threshold * H * W
#     img_array = img_tensor.detach().cpu().numpy().astype(np.bool_)
#     tmp_image1 = remove_small_objects(img_array, max_size=size)
#     tmp_image2 = (1 - tmp_image1).astype(np.bool_)
#     tmp_image3 = remove_small_objects(tmp_image2, max_size=size)
#     tmp_image4 = 1 - tmp_image3
#     tmp_image4 = tmp_image4.astype(np.float32)
#
#     if type(img_tensor) is torch.Tensor:
#         tmp_image4 = torch.from_numpy(tmp_image4)
#         tmp_image4 = tmp_image4.to(img_tensor.device)
#     return tmp_image4

def RemoveSmallArea(img_tensor, threshold):
    # 将tensor转换为numpy数组
    img_array = img_tensor.cpu().numpy() if torch.is_tensor(img_tensor) else img_tensor

    # 使用正确的参数名
    cleaned_array = remove_small_objects(img_array.astype(bool), min_size=threshold)

    # 转换回tensor
    return torch.from_numpy(cleaned_array.astype(np.float32))

