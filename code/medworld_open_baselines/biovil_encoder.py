"""Pinned public BioViL-T image model; single-image (missing-prior) interface."""
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'biovil_weights' / 'biovil_t_image_model_proj_size_128.pt'
URL = 'https://huggingface.co/microsoft/BiomedVLP-BioViL-T/resolve/v1.0/biovil_t_image_model_proj_size_128.pt'
MD5 = 'a83080e2f23aa584a4f2b24c39b1bb64'


def load_encoder(checkpoint=WEIGHTS):
    import torch
    checkpoint = Path(checkpoint)
    if hashlib.md5(checkpoint.read_bytes()).hexdigest() != MD5:
        raise ValueError('BioViL-T official checkpoint MD5 mismatch')
    sys.path.insert(0, str(ROOT / 'biovil_vendor' / 'hi-ml-multimodal' / 'src'))
    from health_multimodal.image.model.model import ImageModel
    from health_multimodal.image.model.types import ImageEncoderType
    model = ImageModel(img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
                       joint_feature_size=128, freeze_encoder=True)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    return model.requires_grad_(False).eval()


def transform():
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Grayscale
    # Matches upstream 512 short-edge / 448 center crop and [0,1] intensities.
    return Compose([Grayscale(num_output_channels=3), Resize(512), CenterCrop(448), ToTensor()])


def load_image(path):
    # Upstream inference first remaps the full image intensity range to uint8.
    # This happens before resizing/cropping and must not be omitted.
    from health_multimodal.image.data.io import load_image as official_load_image
    return official_load_image(Path(path))


def feature_map(model, pixels):
    # f_static concatenated with the learned missing-prior representation.
    # Never pass a follow-up image into the source encoder.
    return model(pixels).patch_embeddings
