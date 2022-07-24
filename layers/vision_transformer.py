import torch
import torch.nn as nn
from layers import TransformerBlock


class VisionTransformer(nn.Module):

    def __init__(self, n_images, embed_size, n_classes, heads=5, n_transformers=5):
        super(VisionTransformer, self).__init__()

        self.n_transformers = n_transformers
        self.n_classes = n_classes

        self.image_embbeding = nn.Linear(embed_size, embed_size)
        self.position_embbeding = nn.Linear(n_images, embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads)

        self.linear_to_classes = nn.Linear(embed_size*n_images, n_classes)

    def forward(self, x):
        batch, n, embed_size = x.size()

        positions = torch.tensor(range(0, n)).float().cuda()

        position_embed = self.position_embbeding(positions)
        img_embed = self.image_embbeding(x)

        x = position_embed + img_embed

        for i in range(self.n_transformers):
            x = self.transformer_block(x, None)

        x = x.view(batch, n*embed_size)

        x = self.linear_to_classes(x)
        x = torch.softmax(x, dim=1)

        return x
