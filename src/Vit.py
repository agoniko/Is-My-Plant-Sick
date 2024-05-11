import torch
from torch import nn
import numpy as np
from tqdm import tqdm


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.projector = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.projector(x)
        x = x.flatten(
            2
        )  # [batch, embed_dim, sqrt(num_patches), sqrt(num_patches)] ===> [batch, embed_dim, num_patches]
        x = x.transpose(
            1, 2
        )  # [batch, embed_dim, num_patches] ===> [batch, num_patches, embed_dim]
        return x


class Attention(nn.Module):
    def __init__(self, dim, n_heads=12):
        super().__init__()
        self.dim = dim  # embedding dimension
        self.n_heads = n_heads  # number of heads
        self.head_dim = dim // n_heads  # embedding dimension of every head
        self.scale = self.head_dim**0.5

        self.qkv = nn.Linear(
            dim, dim * 3
        )  # Linear projection to dim*3 because query, key, and value are projected together and then divided (it is possible to do it separately)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x.shape = (n_samples, n_patches + 1, dim) --> n_patches + 1 because of the class token
        """

        n_samples, n_tokens, dim = x.shape

        # from on now n_patches = n_patches + 1
        # create q, k, and v and divide the embedding for the number of heads

        # qkv --> [n_samples, n_patches, dim*3]
        # Insert code here to create qkv
        qkv = self.qkv(x)

        # Create the heads using qkv --> reshape to shape [n_samples, n_patches, 3, n_heads, head_dim]
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)

        # Divide q, k, and v --> [n_samples, n_patches, n_heads, head_dim] for each of them
        q, k, v = qkv.unbind(dim=2)

        # Transpose q, k, and v to shape [n_samples, n_heads, n_patches, head_dim] for attention computation
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Compute self-attention between q, k, and v
        # Attention = Softmax( ( Q @ K.T) / scale) @ V with @ = dot product
        attention_score = torch.matmul(
            q, k.transpose(-2, -1)
        )  # Shape: [n_samples, n_heads, n_patches, n_patches]
        attention_score /= self.scale

        # Compute the weighted sum of the values along the last axis
        attention_weights = torch.softmax(attention_score, dim=-1)
        out = torch.matmul(attention_weights, v)

        out = out.permute(0, 2, 1, 3)  # [n_samples, n_patches, n_heads, head_dim]
        out = out.reshape(
            n_samples, n_tokens, self.n_heads * self.head_dim
        )  # [n_samples, n_patches, dim]

        # Linear projection of the features --> [n_samples, n_patches, dim]
        x = self.proj(out)
        return x, attention_weights


class MLP(nn.Module):  # Multi Layer Perceptron
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        """
        x.shape() = (n_samples, n_patches + 1, in_features)
        """

        #
        x = self.fc1(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches + 1, hidden_features)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim=384, n_heads=12, mlp_ratio=3):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dim)
        self.norm2 = nn.LayerNorm(dim)

        # This is needed to register the forward hook
        self.attention_weights = None

    def forward(self, x):
        x = self.norm1(x)
        out, self.attention_weights = self.attention(x)
        x = x + out

        x = self.norm2(x)
        x = x + self.mlp(x)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_channels=3,
        n_classes=30,
        dim_emb=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.embed = PatchEmbedding(img_size, patch_size, in_channels, dim_emb)
        self.class_token = nn.Parameter(
            torch.randn(1, 1, dim_emb)
        )  # class token capture global information of the image

        self.positional_embedding = nn.Parameter(
            torch.rand(1, self.num_patches + 1, dim_emb)
        )  # positional embedding, num_patches +1 because of the class token

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(dim_emb, n_heads, mlp_ratio) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(dim_emb)
        self.logits = nn.Linear(dim_emb, n_classes)

    def forward(self, x):
        n_samples = x.shape[0]

        patch_embed = self.embed(x)
        class_token = self.class_token.expand(n_samples, -1, -1)

        # Concatenate the class token and the patch embedding
        patch_embed = torch.cat([class_token, patch_embed], dim=1)

        # Add the positional embedding
        patch_embed += self.positional_embedding

        # Pass the patch embedding through the transformer blocks
        patch_embed = self.transformer_blocks(patch_embed)

        # Get the class token
        class_token = patch_embed[:, 0, :]

        logits = self.logits(class_token)

        return logits
