from email.policy import strict

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .prompt_pool import Prompt


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # 线性变换层，用于将每个patch转换为embed_dim维度的向量
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )

    def forward(self, x):
        x = self.projection(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_drop=0., proj_drop=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.scale = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.general_prompt = torch.nn.Parameter(torch.randn(4, self.head_dim), requires_grad=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        g_prompt = self.general_prompt.expand(B, -1, -1).unsqueeze(1)
        g_prompt = g_prompt.repeat(1, self.num_heads, 1, 1)
        k = torch.cat([g_prompt, k], dim=2)
        v = torch.cat([g_prompt, v], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, attn_depth=1):
        super(TransformerEncoderLayer, self).__init__()
        self.attn_layers = nn.ModuleList()
        for i in range(attn_depth):
            self.attn_layers.append(nn.Sequential(
                nn.LayerNorm(embed_dim),
                MultiHeadSelfAttention(embed_dim, num_heads),
                nn.Dropout(dropout)
            ))
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for attn_layer in self.attn_layers:
            x = x + attn_layer(x)
        x = x + self.dropout(self.mlp(self.norm(x)))
        return x


class ViT_Prompt(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, num_classes, depth, num_heads, drop_rate=0.1,
                 embedding_key='mean', prompt_init='uniform', top_k=3, prompt_length=5, batchwise_prompt=False,
                 prompt_key_init='uniform', prompt_pool=True, prompt_key=True, pool_size=None,
                 mlp_ratio=4.0, dropout=0.1, attn_depth=1):

        super(ViT_Prompt, self).__init__()

        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)

        if prompt_length is not None and pool_size is not None and prompt_pool:
            self.prompt = Prompt(length=prompt_length, embed_dim=embed_dim, embedding_key=embedding_key,
                                 prompt_init=prompt_init, prompt_pool=prompt_pool, prompt_key=prompt_key,
                                 pool_size=pool_size, top_k=top_k, batchwise_prompt=batchwise_prompt,
                                 prompt_key_init=prompt_key_init)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        num_patches = self.patch_embed.num_patches
        embed_len = num_patches
        if prompt_length is not None and pool_size is not None and prompt_pool:
            embed_len += prompt_length * top_k

        self.pos_embed = nn.Parameter(torch.zeros(1, embed_len + 1+15, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.prompt_pool = prompt_pool

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout, attn_depth) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classifier Head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_prompt(self, x, cls_feature=None):
        x = self.patch_embed(x)
        if hasattr(self, 'prompt'):
            res = self.prompt(x, prompt_mask=None, cls_features=cls_feature)
            self.total_prompt_len = res['total_prompt_len']
            x = res['prompted_embedding']
        else:
            res = dict()
        return x, res

    # def forward_features(self, x, res, general_prompt):
    #
    #     x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    #     x = self.pos_drop(x + self.pos_embed)
    #
    #     for i, layer in enumerate(self.layers):
    #         g_prompt = general_prompt[i].expand(x.shape[0], -1, -1)
    #         embeddings = torch.cat((x, g_prompt), dim=1)
    #         embeddings = layer(embeddings)
    #     embeddings = self.norm(embeddings)
    #     res['feature_medium'] = embeddings[:, :-12]
    #
    #     return res
    def forward_features(self, x, res):

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for i, layer in enumerate(self.layers):
            x = layer(x)
        embeddings = self.norm(x)
        res['feature_medium'] = embeddings[:, 1:]

        return res

    def forward_head(self, res):
        x = res['feature_medium']
        if self.prompt_pool:
            x = x[:, 1:(1 + self.total_prompt_len)]
            res['prompt_feature'] = x
            prompt_mean = x.mean(dim=1)
            res['prompt_feat_mean'] = prompt_mean

            feature = res['feature_medium'][:, (2 * self.total_prompt_len):]

            res['wto_prompt_feature'] = feature

            prompt_head = self.norm(prompt_mean)

            res['logits'] = self.head(prompt_head)

        return res

    def forward(self, x):
        x, res = self.forward_prompt(x)
        res = self.forward_features(x, res)
        res = self.forward_head(res)
        return res

# # 定义模型参数
# image_size = 11
# patch_size = 1
# in_channels = 144
# embed_dim = 768
# num_classes = 6
# depth = 8
# num_heads = 8
# drop_rate = 0.1
# embedding_key = 'cls'
# prompt_init = 'uniform'
# top_k = 5
# prompt_length = 5
# batchwise_prompt = True
# prompt_key_init = 'uniform'
# prompt_pool = True
# pool_size = 100
# mlp_ratio = 4.0
# dropout = 0.1
# attn_depth = 1
#
# # 创建模型实例
# model = ViT_Prompt(
#     image_size=image_size,
#     patch_size=patch_size,
#     in_channels=in_channels,
#     embed_dim=embed_dim,
#     num_classes=num_classes,
#     depth=depth,
#     num_heads=num_heads,
#     drop_rate=drop_rate,
#     embedding_key=embedding_key,
#     prompt_init=prompt_init,
#     top_k=top_k,
#     prompt_length=prompt_length,
#     batchwise_prompt=batchwise_prompt,
#     prompt_key_init=prompt_key_init,
#     prompt_pool=prompt_pool,
#     pool_size=pool_size,
#     mlp_ratio=mlp_ratio,
#     dropout=dropout,
#     attn_depth=attn_depth
# )
#
# # 加载预训练权重
# pretrained_weights_path = 'vit_model.pth'
# pretrained_weights = torch.load(pretrained_weights_path)
# model.load_state_dict(pretrained_weights, strict=False)
#
# # 验证模型前向传播
# batch_size = 2
# x = torch.randn(batch_size, in_channels, image_size, image_size)
# output = model(x)
# print(output['cls_token'].shape)
