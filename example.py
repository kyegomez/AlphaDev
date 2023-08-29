
import torch
from alpha_dev.model import AlphaDev

model =  AlphaDev().cuda()

x = torch.randint(0, 256, (1, 1024)).cuda()

model(x) # (1, 1024, 20000)