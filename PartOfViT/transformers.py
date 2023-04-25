import paddle
import paddle.nn as nn
import numpy as np
from PIL import Image

paddle.set_device('cpu')


class PatchEmbedding(nn.Layer):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, dropout = 0.):
        super().__init__()
        self.patch_embedding = nn.Conv2D(in_channels = in_channels,
                                         out_channels = embed_dim,
                                         kernel_size = patch_size,
                                         stride = patch_size,
                                         weight_attr = paddle.ParamAttr(initializer = nn.initializer.Constant(1.0)),
                                         bias_attr = False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # [n,c,h,w]
        x = self.patch_embedding(x)  # [n,c',h',w]
        x = x.flatten(2)  # [n,c',h'*w']
        x = x.transpose([0, 2, 1])  # [n,h'*w',c]
        x = self.dropout(x)
        return x


class Encoder(nn.Layer):
    def __init__(self, embed_dim):
        super().__init__()
        # 注意力机制未写
        self.attn = Identity()  # TODO
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim)
        self.attn_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = h + x

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h + x

        return x


class ViT(nn.Layer):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding(224, 7, 3, 16)
        layer_list = [Encoder(16) for i in range(5)]
        self.encoders = nn.LayerList(layer_list)
        self.head = nn.Linear(16, 10)  # 10:num_classes
        self.avgpool = nn.AdaptiveAvgPool1D(1)

    def forward(self, x):
        x = self.patch_embed(x)
        for encoder in self.encoders:
            x = encoder(x)
        # layernorm
        # [n,h'*w',c]
        x = x.transpose([0, 2, 1])
        x = self.avgpool(x)  # [0,c,1]
        x = x.flatten(1)  # [n, c]
        x = self.head(x)
        return x


class Mlp(nn.Layer):
    def __init__(self, embed_dim, mlp_ratio = 4.0, dropout = 0.):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Identity(nn.Layer):
    def __int__(self):
        super().__init__()

    def forward(self, x):
        return x


def main():
    # # 1.load image and convert to tensor
    # img = Image.open(r"D:\A.python文件\ViT\img.jpg")
    # img = np.array(img)
    # for i in range(28):
    #     for j in range(28):
    #         print(f'{img[i, j]:03} ', end = '')
    #     print()
    #
    # sample = paddle.to_tensor(img, dtype = 'float32')
    # sample = sample.reshape([1, 1, 28, 28])
    # print(sample.shape)
    # # 2.Patch Embedding
    # patch_embed = PatchEmbedding(image_size = 28, patch_size = 7, in_channels = 1, embed_dim = 1)
    # out = patch_embed(sample)
    # print('out shape = ', out.shape)
    # # 3.Mlp
    # mlp = Mlp(1)
    # out = mlp(out)
    # print(out)
    # print('out shape = ', out.shape)

    t = paddle.randn([4, 3, 224, 224])
    model = ViT()
    out = model(t)
    print(out.shape)


if __name__ == "__main__":
    main()
