import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from utils.CUDA_Check import GPUorCPU
from torchvision.io import read_image, ImageReadMode

#通过GPUorCPU类确定运行设备
DEVICE = GPUorCPU().DEVICE
#模型输入图像的尺寸设定(训练时调整大小)
model_input_image_size_height = 256
model_input_image_size_width = 256
#随机裁剪尺寸（如训练时的随机裁剪区域大小）
random_crop_size = 224

#定义一个简单的归一化类，将像素值归一化到[0,1]范围
class ZeroOneNormalize(object):
    def __call__(self, img):
        #将图像转换为浮点型并除以255
        return img.float().div(255)

#训练数据加载器
class DataLoader_Train(Dataset):
    #定义一系列的数据预处理转换操作（数据增强）
    train_valid_transforms = transforms.Compose(
        [
            # transforms.CenterCrop(224),
            #resize调整图像大小到指定模型输入尺寸
            transforms.Resize((model_input_image_size_height, model_input_image_size_width), antialias=False),
            #随机裁剪出固定尺寸的区域
            transforms.RandomCrop(random_crop_size),
            #随机水平反转，增强数据多样性
            transforms.RandomHorizontalFlip(),
            #随机垂直翻转
            transforms.RandomVerticalFlip(),
            #将数据归一化到[0,1]
            ZeroOneNormalize(),
        ]
    )
#针对数据标准化的转换（后续归一化操作，确保均值和标准差为0.5）
    train_valid_transforms_Norm = transforms.Compose(
        [
            transforms.Normalize(mean=0.5, std=0.5),
        ]
    )
#构造函数，接收四个文件列表路径
    def __init__(self, file_list_A, file_list_B, file_list_GT, file_list_DM):
        self.file_list_A = file_list_A
        self.file_list_B = file_list_B
        self.file_list_GT = file_list_GT
        self.file_list_DM = file_list_DM
        #两个转换操作分别赋值
        self.transform1 = self.train_valid_transforms
        self.transform2 = self.train_valid_transforms_Norm
#数据集长度，如果所有文件列表长度一致，就返回该长度，
    # 源图像A、B，GT真实图、DM决策图
    def __len__(self):
        if len(self.file_list_A) == len(self.file_list_B) == len(self.file_list_GT) == len(self.file_list_DM):
            self.filelength = len(self.file_list_A)
            return self.filelength
#获取单个样本数据
    def __getitem__(self, idx):
        #获取随机种子，保证同一个索引下各图像进行相同的数据增强操作
        seed = torch.random.seed()
        #获取图像A(sourceA)，模式为RGB
        imgA_path = self.file_list_A[idx]
        img_A = read_image(imgA_path, mode=ImageReadMode.RGB).to(DEVICE)
        #为确保相同随机转换，每次重置随机种子
        torch.random.manual_seed(seed)
        img_A = self.transform1(img_A)
        imgA_transformed = self.transform2(img_A)

        #读取图像B(sourceB)
        imgB_path = self.file_list_B[idx]
        img_B = read_image(imgB_path, mode=ImageReadMode.RGB).to(DEVICE)
        torch.random.manual_seed(seed)
        img_B = self.transform1(img_B)
        imgB_transformed = self.transform2(img_B)

        #读取图像ground truth图像
        imgGT_path = self.file_list_GT[idx]
        img_GT = read_image(imgGT_path, mode=ImageReadMode.RGB).to(DEVICE)
        torch.random.manual_seed(seed)
        imgGT_transformed = self.transform1(img_GT)

        #读取decision map图像（灰度模式）
        imgDM_path = self.file_list_DM[idx]
        img_DM = read_image(imgDM_path, mode=ImageReadMode.GRAY).to(DEVICE)
        torch.random.manual_seed(seed)
        imgDM_transformed = self.transform1(img_DM)

        #返回预处理后的四个图像tensor
        return imgA_transformed, imgB_transformed, imgGT_transformed, imgDM_transformed

#评估数据加载器（与训练数据加载器不同，只做归一化）
class Dataloader_Eval(Dataset):
    #定义评估时的转换操作
    eval_transforms = transforms.Compose(
        [
            #transforms.Resize((224, 224), antialias=False),
            # transforms.ToTensor(),
            ZeroOneNormalize(),
            transforms.Normalize(mean=0.5, std=0.5),
        ]
    )

    def __init__(self, file_list_A, file_list_B):
        self.file_list_A = file_list_A
        self.file_list_B = file_list_B
        self.transform1 = self.eval_transforms
        self.transform2 = self.eval_transforms

    def __len__(self):
        if len(self.file_list_A) == len(self.file_list_B):
            self.filelength = len(self.file_list_A)
            return self.filelength

    def __getitem__(self, idx):
        imgA_path = self.file_list_A[idx]
        img_A = read_image(imgA_path, mode=ImageReadMode.RGB).to(DEVICE)
        # img_A = Image.open(imgA_path).convert('RGB')
        imgA_transformed = self.transform1(img_A).to(DEVICE)

        imgB_path = self.file_list_B[idx]
        img_B = read_image(imgB_path, mode=ImageReadMode.RGB).to(DEVICE)
        # img_B = Image.open(imgB_path).convert('RGB')
        imgB_transformed = self.transform1(img_B).to(DEVICE)

        return imgA_transformed, imgB_transformed
#DataLoader_Train 和 Dataloader_Eval 主要负责读取图像文件、应用预处理与数据增强，使得训练与评估时输入数据满足模型需求。
#为确保同一索引下的不同图像进行一致的数据增强操作，采用了相同的随机种子