import torch
import math
import numpy as np
import torch.nn as nn


class Pos_Embed(nn.Module):
    def __init__(self, channels, num_frames, num_joints):
        super().__init__()

        # 根据帧序和节点序生成位置向量
        pos_list = []
        for tk in range(num_frames):
            for st in range(num_joints):
                pos_list.append(st)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()  # num_frames*num_joints, 1

        pe = torch.zeros(num_frames * num_joints, channels)  # T*N, C

        div_term = torch.exp(torch.arange(0, channels, 2).float() * -(math.log(10000.0) / channels))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列 # 偶数C维度sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列 # 奇数C维度cos
        pe = pe.view(num_frames, num_joints, channels).permute(2, 0, 1).unsqueeze(0)  # T N C -> C T N -> 1 C T N
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv # BCTN
        x = self.pe[:, :, :x.size(2)]
        return x


if __name__ == "__main__":
    B = 2
    C = 4
    T = 120
    N = 25
    x = torch.rand((B, C, T, N))
    
    Pos_embed_1 = Pos_Embed(C, T, N)
    PE = Pos_embed_1(x)
    # print(PE.shape) # 1 C T N
    x = x + PE

    print("All Done !")

