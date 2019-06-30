dependencies = ['torch', 'torchvision', 'pretrainedmodels', 'efficientnet-pytorch']

from torchvision.models import mobilenet_v2, shufflenet_v2_x1_0
from pretrainedmodels.models import pnasnet5large


# See also:
# https://paperswithcode.com/sota/image-classification-on-imagenet
# https://github.com/Cadene/pretrained-models.pytorch
# https://github.com/osmr/imgclsmob



def object_recognition(pretrained=False, **kwargs):
    """Image recognition model balancing good performance and model size"""
   
    model_name = kwargs.pop('model_name', 'efficientnet-b2')
    if pretrained is True:
        model = EfficientNet.from_pretrained(model_name, **kwargs)
    else:
        model = EfficientNet.from_name(model_name, **kwargs)
    return model


def object_recognition_sota(pretrained=False, **kwargs):
    """State of the art object recognition model"""

    if pretrained is True:
        pretrained_dataset = 'imagenet'
        num_classes=1000
        model = pnasnet5large(pretrained='imagenet', num_classes=num_classes, **kwargs)
    else:
        model = pnasnet5large(**kwargs)
    return model


def object_recognition_faster(pretrained=False, **kwargs):
    """Faster image recognition suitable for mobile devices"""
   
    return shufflenet_v2_x1_0(pretrained=pretrained, **kwargs)
