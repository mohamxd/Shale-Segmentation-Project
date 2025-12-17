import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except Exception:
    HAS_SMP = False

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=6, base=64):
        super().__init__()
        feats = [base, base*2, base*4, base*8]
        self.encs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for f in feats:
            self.encs.append(DoubleConv(prev, f))
            self.pools.append(nn.MaxPool2d(2))
            prev = f
        self.bottleneck = DoubleConv(prev, prev*2)
        # decoder
        self.upconvs = nn.ModuleList()
        self.decs = nn.ModuleList()
        rev = feats[::-1]
        cur = prev*2
        for f in rev:
            self.upconvs.append(nn.ConvTranspose2d(cur, f, 2, 2))
            self.decs.append(DoubleConv(cur, f))
            cur = f
        self.final = nn.Conv2d(cur, num_classes, 1)
    def forward(self, x):
        enc = []
        z = x
        for e, p in zip(self.encs, self.pools):
            z = e(z); enc.append(z); z = p(z)
        z = self.bottleneck(z)
        for up, dec, skip in zip(self.upconvs, self.decs, reversed(enc)):
            z = up(z)
            if z.shape[-2:] != skip.shape[-2:]:
                z = nn.functional.interpolate(z, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            z = torch.cat([z, skip], dim=1)
            z = dec(z)
        return self.final(z)

def build_model(arch: str, encoder: str, in_channels: int, classes: int, encoder_weights=None):
    arch_lower = (arch or "").lower()
    if arch_lower in {"simple", "simple_unet"}:
        return SimpleUNet(in_channels=in_channels, num_classes=classes)

    if not HAS_SMP:
        raise ImportError(
            "segmentation_models_pytorch is required for architecture "
            f"'{arch}'. Install it or set model.arch to 'simple'."
        )

    if arch_lower in {"unetpp", "unet++"}:
        return smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    if arch_lower in {"deeplabv3plus", "deeplabv3+"}:
        return smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    if arch_lower == "unet":
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    raise ValueError(f"Unsupported architecture '{arch}'.")


def load_checkpoint_strict(
    model: nn.Module, checkpoint_path: str, device: torch.device, arch: str | None = None, encoder: str | None = None
) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state

    cleaned = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith("module.") else k
        cleaned[new_key] = v

    load_result = model.load_state_dict(cleaned, strict=False)
    missing, unexpected = load_result.missing_keys, load_result.unexpected_keys
    if missing or unexpected:
        def _summarize(keys):
            head = keys[:20]
            suffix = "..." if len(keys) > 20 else ""
            return head, suffix

        miss_head, miss_suffix = _summarize(missing)
        unexp_head, unexp_suffix = _summarize(unexpected)
        summary = [
            f"Checkpoint mismatch for arch={arch or model.__class__.__name__}, encoder={encoder or 'n/a'}",
            f"  path: {checkpoint_path}",
            f"  missing keys: {len(missing)} -> {', '.join(miss_head)}{miss_suffix if missing else ''}",
            f"  unexpected keys: {len(unexpected)} -> {', '.join(unexp_head)}{unexp_suffix if unexpected else ''}",
        ]
        raise RuntimeError("\n".join(summary))
