import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


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
        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.projection[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.projection[0].bias)

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

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.qkv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.qkv.bias)
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

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

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)

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

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.attn_layers:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.zeros_(module.bias)
        nn.init.kaiming_normal_(self.mlp.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.mlp.fc1.bias)
        nn.init.kaiming_normal_(self.mlp.fc2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.mlp.fc2.bias)

    def forward(self, x):
        for attn_layer in self.attn_layers:
            x = x + attn_layer(x)
        x = x + self.dropout(self.mlp(self.norm(x)))
        return x


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # 取出 cls token
        cls_token = x[:, 0]
        return self.fc(cls_token)


class ViTBackbone(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, depth, num_heads,
                 mlp_ratio=4.0, dropout=0.1, attn_depth=1):

        super(ViTBackbone, self).__init__()

        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches= self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout, attn_depth) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class ViTModel(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, depth, num_heads, num_classes):
        super(ViTModel, self).__init__()

        self.backbone = ViTBackbone(image_size, patch_size, in_channels, embed_dim, depth, num_heads)

        self.classification_head = ClassificationHead(embed_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.classification_head(x)


