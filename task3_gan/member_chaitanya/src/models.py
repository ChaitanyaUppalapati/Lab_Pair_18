"""CycleGAN networks.

Generator (ResNet-based, 9 blocks): c7s1-64, d128, d256, R256 x9, u128, u64, c7s1-3 + tanh.
  - reflection padding; InstanceNorm + ReLU after every conv except the last
  - upsampling = nearest-neighbour 2x resize + 3x3 conv (avoids checkerboard artifacts)
Discriminator (70x70 PatchGAN): C64-C128-C256-C512 with 4x4 kernels, then a 4x4
  conv to 1 channel. No norm on C64; InstanceNorm elsewhere; LeakyReLU(0.2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_in_relu(in_ch: int, out_ch: int, kernel: int, stride: int = 1, reflect_pad: int = 0, pad: int = 0) -> list:
    layers = [nn.ReflectionPad2d(reflect_pad)] if reflect_pad else []
    layers += [nn.Conv2d(in_ch, out_ch, kernel, stride, padding=pad), nn.InstanceNorm2d(out_ch), nn.ReLU(inplace=True)]
    return layers


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch),
        )

    def forward(self, x):
        return x + self.body(x)


class ResnetGenerator(nn.Module):
    def __init__(self, ngf: int = 64, n_blocks: int = 9, n_down: int = 2, attention: bool = False):
        super().__init__()
        layers = conv_in_relu(3, ngf, 7, reflect_pad=3)                         # c7s1-64
        ch = ngf
        for _ in range(n_down):                                                   # d128, d256
            layers += conv_in_relu(ch, ch * 2, 3, stride=2, pad=1)
            ch *= 2
        layers += [ResidualBlock(ch) for _ in range(n_blocks)]                    # R256 x9
        if attention:
            layers.append(SelfAttention(ch))                                      # optional: on the 64x64x256 map
        for _ in range(n_down):                                                   # u128, u64
            layers += [nn.Upsample(scale_factor=2, mode="nearest")] + conv_in_relu(ch, ch // 2, 3, reflect_pad=1)
            ch //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ch, 3, 7), nn.Tanh()]        # c7s1-3
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, ndf: int = 64, attention: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(3, ndf, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),                                   # C64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1), nn.InstanceNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),  # C128
        ]
        if attention:
            layers.append(SelfAttention(ndf * 2))                                                            # optional: 64x64x128
        layers += [
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1), nn.InstanceNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),  # C256
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1), nn.InstanceNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True),  # C512
            nn.Conv2d(ndf * 8, 1, 4, 1, 1),                                                                  # 1-channel patch map
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    """SAGAN self-attention (Zhang et al., 2019): every position attends to every other position.

    out = x + gamma * (softmax(f(x)^T g(x)) applied to h(x)); gamma starts at 0, so the block begins
    as an identity and the network learns how much global context to mix in.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.query = nn.Conv2d(ch, ch // 8, 1)
        self.key = nn.Conv2d(ch, ch // 8, 1)
        self.value = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).flatten(2).transpose(1, 2)            # (B, N, C/8)
        k = self.key(x).flatten(2)                               # (B, C/8, N)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)            # (B, N, N)
        v = self.value(x).flatten(2)                             # (B, C, N)
        out = torch.bmm(v, attn.transpose(1, 2)).view(B, C, H, W)
        return x + self.gamma * out


def add_spectral_norm(module: nn.Module) -> nn.Module:
    """Wrap every Conv2d in spectral normalisation (Miyato et al., 2018)."""
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, nn.utils.parametrizations.spectral_norm(child))
        else:
            add_spectral_norm(child)
    return module


def make_generator(m: dict) -> nn.Module:
    """Build a generator from the model config (attention / spectral norm optional, default off)."""
    g = ResnetGenerator(m["ngf"], m["n_res_blocks"], m["n_downsampling"], attention=m.get("g_attention", False))
    g.apply(lambda mod: init_weights(mod, m.get("init_std", 0.02)))
    return add_spectral_norm(g) if m.get("spectral_norm_g", False) else g


def make_discriminator(m: dict) -> nn.Module:
    d = PatchDiscriminator(m["ndf"], attention=m.get("d_attention", False))
    d.apply(lambda mod: init_weights(mod, m.get("init_std", 0.02)))
    return add_spectral_norm(d) if m.get("spectral_norm_d", False) else d


def init_weights(module: nn.Module, std: float = 0.02) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class ImagePool:
    """History buffer of generated images (Shrivastava et al.): D sees a mix of current and past fakes."""

    def __init__(self, size: int):
        self.size, self.images = size, []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return images
        out = []
        for img in images.detach():
            img = img.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(img)
                out.append(img)
            elif torch.rand(1).item() < 0.5:
                idx = int(torch.randint(0, self.size, (1,)).item())
                out.append(self.images[idx].clone())
                self.images[idx] = img
            else:
                out.append(img)
        return torch.cat(out, 0)


def diff_augment(x: torch.Tensor, translation: float = 0.125, cutout: float = 0.5) -> torch.Tensor:
    """DiffAugment (Zhao et al., 2020) translation + cutout; differentiable, applied to real and fake before D."""
    B, _, H, W = x.shape
    # translation: shift by up to `translation` of the size, zero fill
    sx, sy = int(H * translation + 0.5), int(W * translation + 0.5)
    tx = torch.randint(-sx, sx + 1, (B, 1, 1), device=x.device)
    ty = torch.randint(-sy, sy + 1, (B, 1, 1), device=x.device)
    gb, gx, gy = torch.meshgrid(torch.arange(B, device=x.device), torch.arange(H, device=x.device),
                                torch.arange(W, device=x.device), indexing="ij")
    gx = torch.clamp(gx + tx + 1, 0, H + 1)
    gy = torch.clamp(gy + ty + 1, 0, W + 1)
    x = F.pad(x, [1, 1, 1, 1]).permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)
    # cutout: zero one square of side `cutout` x size at a random location
    ch, cw = int(H * cutout + 0.5), int(W * cutout + 0.5)
    ox = torch.randint(0, H + (1 - ch % 2), (B, 1, 1), device=x.device)
    oy = torch.randint(0, W + (1 - cw % 2), (B, 1, 1), device=x.device)
    gb, gx, gy = torch.meshgrid(torch.arange(B, device=x.device), torch.arange(ch, device=x.device),
                                torch.arange(cw, device=x.device), indexing="ij")
    gx = torch.clamp(gx + ox - ch // 2, 0, H - 1)
    gy = torch.clamp(gy + oy - cw // 2, 0, W - 1)
    mask = torch.ones(B, H, W, dtype=x.dtype, device=x.device)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1)
