import torch

#一些超参数设置


class Configs:
    def __init__(self):
        pass


configs = Configs()

# trainer related
configs.n_cpu = 0
configs.device = torch.device('cuda:0')
configs.batch_size_test = 16 #
configs.batch_size = 16 #这个应该是训练用的batch szie吧？
configs.lr = 0.0001
configs.weight_decay = 0
configs.display_interval = 120
configs.num_epochs = 300
configs.early_stopping = True
configs.patience = 3
configs.gradient_clipping = False
configs.clipping_threshold = 1.

# lr warmup
# 这里使用了预热学习率，即先用最初的小学习率训练，然后每个step增大一点点，
# 直到达到最初设置的比较大的学习率时（注：此时预热学习率完成），
# 采用最初设置的学习率进行训练（注：预热学习率完成后的训练过程，学习率是衰减的），
# 有助于使模型收敛速度变快，效果更佳。
configs.warmup = 3000

# data related
configs.input_dim = 10 # 相当于input channel?
configs.output_dim = 1 # 相当于output channel?
configs.input_dim2 = 8 # 相当于input channel?

configs.input_length = 5 # 时间的长度
configs.output_length = 1

# configs.input_gap = 1
# configs.pred_shift = 1

# model
configs.d_model = 256 # 嵌入向量的维度，即用多少维来表示一个符号。
configs.patch_size = (4, 4) # 每个图像块的大小
configs.emb_spatial_size = 4 # (x/patch_size)^2
configs.nheads = 8 #注意力头
configs.dim_feedforward = 512 #feedforward层特征的维度大小
configs.dropout = 0.1
configs.num_encoder_layers = 2
configs.num_decoder_layers = 1

# configs.ssr_decay_rate = 0 # 这个是什么？似乎是预测sst的一个参数，这里设置成0不用就行
