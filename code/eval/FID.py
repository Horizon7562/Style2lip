import torch
import numpy as np
from torchvision import models, transforms
from scipy.linalg import sqrtm
from PIL import Image

# 预处理 InceptionV3 所需的图片
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载预训练的 InceptionV3 模型
inception_model = models.inception_v3(pretrained=True, transform_input=False)
inception_model.eval()

def get_inception_features(images):
    """
    提取 InceptionV3 特征
    images: Tensor，大小为 [batch * 5, 3, 256, 256]
    """
    features = []
    for i in range(images.size(0)):
        image = images[i].unsqueeze(0)  # [1, 3, 256, 256]
        
        with torch.no_grad():
            feature = inception_model(image)  # [1, 2048] 特征维度
            features.append(feature.cpu().numpy().flatten())
    
    return np.array(features)

def compute_fid(real_images, generated_images):
    """
    计算 Fréchet Inception Distance (FID)
    real_images 和 generated_images 的维度为 [batch, 3, 5, 256, 256]
    """
    # 将数据转换为 [batch * 5, 3, 256, 256]
    real_images = real_images.view(-1, 3, 256, 256)  # 展开 batch 和图像数量维度
    generated_images = generated_images.view(-1, 3, 256, 256)
    
    # 获取真实图像和生成图像的 InceptionV3 特征
    real_features = get_inception_features(real_images)
    generated_features = get_inception_features(generated_images)
    
    # 计算均值和协方差矩阵
    mu_real = np.mean(real_features, axis=0)
    mu_generated = np.mean(generated_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_generated = np.cov(generated_features, rowvar=False)
    
    # 计算 FID
    diff_mu = mu_real - mu_generated
    diff_sigma = sigma_real + sigma_generated - 2 * sqrtm(sigma_real.dot(sigma_generated))
    
    fid = np.sum(diff_mu ** 2) + np.trace(diff_sigma)
    
    return float(fid)

# 示例数据，假设 y 和 y_hat 已经是 [batch, 3, 5, 256, 256] 的 tensor
y = torch.rand(16, 3, 5, 256, 256)  # 真实图像
y_hat = torch.rand(16, 3, 5, 256, 256)  # 生成图像

# 计算 FID
fid_value = compute_fid(y, y_hat)

# 输出 FID 值
print("FID:", fid_value)
