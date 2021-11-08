import matplotlib.pyplot as plt
import torch
import numpy as np


state_dict = torch.load('./save/student_exsmaller_stride2/mobilenetv2_best.pth')
weight_a = state_dict['model']['blocks.2.branches.0.0.conv.0.0.weight'].cpu().numpy()[:, :, 0, 0].transpose()
weight_b = state_dict['model']['blocks.2.branches.1.0.conv.0.0.weight'].cpu().numpy()[:, :, 0, 0].transpose()

diff = 2 * np.abs(weight_a - weight_b) / (np.abs(weight_a) + np.abs(weight_b))
plt.matshow(diff)
plt.savefig('diff_small.png')
