"""Strictly load CheXWorld's published target ViT-B encoder."""
from pathlib import Path
import importlib
import sys
import types

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'chexworld_weights' / 'chexworld_pretrained.tar'


def load_encoder(checkpoint=WEIGHTS):
    import torch
    # Load the upstream package without importing unrelated optional decoders.
    name = 'chexworld_upstream_models'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT / 'chexworld_vendor' / 'models')]
        sys.modules[name] = package
    module = importlib.import_module(name + '.jepa_vit')
    model = module.vit_base(img_size=224, patch_size=16)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model']
    state = {k.removeprefix('target_encoder.'): v for k, v in state.items()
             if k.startswith('target_encoder.')}
    if not state:
        raise ValueError('Published target_encoder weights are absent')
    model.load_state_dict(state, strict=True)
    return model.requires_grad_(False).eval()


def transform():
    from torchvision.transforms import Compose, Resize, CenterCrop, Grayscale, ToTensor, Normalize, InterpolationMode
    return Compose([Resize(256, interpolation=InterpolationMode.BICUBIC), CenterCrop(224),
                    Grayscale(num_output_channels=3), ToTensor(),
                    Normalize([.485, .456, .406], [.229, .224, .225])])


def feature_map(model, pixels):
    tokens = model(pixels)
    if tokens.shape[1:] != (196, 768):
        raise ValueError(f'Unexpected official CheXWorld output: {tokens.shape}')
    return tokens.transpose(1, 2).reshape(-1, 768, 14, 14)
