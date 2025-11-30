import torch
import torch.nn as nn

from criteria.lpips.networks import get_network, LinLayers
from criteria.lpips.utils import get_state_dict


class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).
    Arguments:
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type).to("cuda")

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list).to("cuda")
        self.lin.load_state_dict(get_state_dict(net_type, version))

    # def forward(self, x: torch.Tensor, y: torch.Tensor):
    #     feat_x, feat_y = self.net(x), self.net(y)

    #     diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
    #     res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

    #     return torch.sum(torch.cat(res, 0)) / x.shape[0]
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        batch_size, channel, num_images, height, width = x.shape

        # 展平为 [batch * num_images, channel, height, width]
        x_reshaped = x.view(batch_size * num_images, channel, height, width)
        y_reshaped = y.view(batch_size * num_images, channel, height, width)

        # 获取输入 x 和 y 的特征
        feat_x, feat_y = self.net(x_reshaped), self.net(y_reshaped)

        # 计算特征的差异
        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        
        # 通过线性层处理每一层的特征图差异
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        # 计算最终的损失，除以batch_size和num_images
        return torch.sum(torch.cat(res, 0)) / (batch_size * num_images)
