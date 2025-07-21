from torch import nn


class DWConv(nn.Module):
    def __init__(self, dim=64, kernel_size=3):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, 1, 1, bias=True, groups=dim)