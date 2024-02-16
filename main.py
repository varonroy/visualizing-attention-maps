import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, vit_b_32, ViT_B_16_Weights, ViT_B_32_Weights
from torchvision.datasets import CIFAR100, ImageFolder
from torchvision.transforms import InterpolationMode
import torchvision.transforms.v2 as v2

import matplotlib.pyplot as plt

# model properties (can be extracted by trial and error or inspecting the model code)
num_channels = 3
patch_size = 16
image_size = 224
num_classes = 1000
num_patches = image_size // patch_size  # patches per row or col
shuffle = True
dataset_source = "cifar100"

# build the NN
# source: https://pytorch.org/vision/main/models/vision_transformer.html
vit = None
if patch_size == 16:
    vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
if patch_size == 32:
    vit = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
assert vit is not None

# sanity check
batch_size = 3
img = torch.randn(batch_size, num_channels, image_size, image_size)
out = vit(img)
print("sanity check")
print(" - input :", img.size())
print(" - output:", out.size())
assert out.size() == torch.Size([batch_size, num_classes])

# load the data
dataset = None
transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize([image_size, image_size]),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

if dataset_source == "cifar100":
    dataset = CIFAR100(
        root="~/datasets/cifar100",
        train=False,
        download=True,
        transform=transform,
    )
if dataset_source == "drive":
    # load image from drive
    dataset = ImageFolder(
        root="./imgs/",
        transform=transform,
    )
assert dataset is not None

loader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
img, _ = next(iter(loader))
print("dataset item dimensions and range")
print(" - image:", img.size())
print(" - - min:", torch.min(img))
print(" - - max:", torch.max(img))


# monkey patch the forward function
class WrappedEncoderBlock(nn.Module):
    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.attn = None

    def forward(self, input):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        x = self.encoder.ln_1(input)

        # forward with attention maps
        x, attn = self.encoder.self_attention(x, x, x)

        # save attention weights
        self.attn = attn.detach().cpu()

        x = self.encoder.dropout(x)
        x = x + input

        y = self.encoder.ln_2(x)
        y = self.encoder.mlp(y)
        return x + y


vit.encoder.layers[-1] = WrappedEncoderBlock(vit.encoder.layers[-1])
attn_layer = vit.encoder.layers[-1]


# visualize
def unnormalize(img):
    imin = img.min()
    imax = img.max()
    return (img - imin) / (imax - imin)


def plot_img(ax, img, scale=None):  # img is a tensor [channels, size, size]
    img = unnormalize(img) * 255
    if scale is not None:
        img = img * unnormalize(scale)
    img = img.to(dtype=torch.int)
    img = img.permute((1, 2, 0))
    ax.imshow(img)


def plot_attn(ax, attn):  # attn is a tensor [num_patches^2]
    attn = attn.reshape([1, num_patches, num_patches])
    plot_img(ax, attn)


def plot_img_with_attn(ax, img, attn):
    attn = attn.reshape([1, num_patches, num_patches])
    attn = v2.functional.resize(
        attn,
        [image_size, image_size],
        interpolation=InterpolationMode.BICUBIC,
    )
    plot_img(ax, img, scale=attn)


# pass the image through the vit
print("plotting...")
_ = vit(img)
attn = attn_layer.attn
# dimensions:
# - img: [batch, channels, image_size, image_size]
# - attn: [batch, num_patches ^ 2 + 1, num_patches ^ 2 + 1]

img_idx = 0

class_token_idx = 0
fig, ax = plt.subplots(nrows=2, ncols=3)
maps = [
    attn[img_idx, class_token_idx, 1:],
    attn[img_idx, 1:, class_token_idx],
]
for i, attn_map in enumerate(maps):
    plot_img(ax[i][0], img[img_idx])
    plot_attn(ax[i][1], attn_map)
    plot_img_with_attn(ax[i][2], img[img_idx], attn_map)
plt.show()
