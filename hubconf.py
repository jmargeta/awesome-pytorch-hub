dependencies = ['mxnet', 'torch', 'torchvision', 'pretrainedmodels', 'efficientnet_pytorch']

from efficientnet_pytorch import EfficientNet
from gluoncv.model_zoo import mask_rcnn_fpn_resnet101_v1d_coco as mask_rcnn_fpn_resnet101_v1d_coco_mxnet
from pretrainedmodels.models import pnasnet5large
from torchvision.models import mobilenet_v2, shufflenet_v2_x1_0
from mxnet.gluon.model_zoo.vision import mobilenet_v2_1_0 as mobilenet_v2_1_0_mxnet 

# See also:
# https://paperswithcode.com/sota/image-classification-on-imagenet
# https://github.com/Cadene/pretrained-models.pytorch
# https://github.com/osmr/imgclsmob
# https://gluon-cv.mxnet.io/model_zoo/index.html


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

    num_classes = 1000
    if pretrained is True:
        pretrained_dataset = 'imagenet'
        model = pnasnet5large(pretrained='imagenet', num_classes=num_classes, **kwargs)
    else:
        model = pnasnet5large(num_classes=num_classes, **kwargs)
    return model


def object_recognition_faster(pretrained=False, **kwargs):
    """Faster image recognition suitable for mobile devices"""
   
    return shufflenet_v2_x1_0(pretrained=pretrained, **kwargs)


def object_recognition_faster_mxnet(pretrained=True, **kwargs):
    """Faster object recognition with mxnet mobilenet_v2_1_0 backend"""

    return mxnet.gluon.model_zoo.vision.mobilenet_v2_1_0_mxnet(pretrained=pretrained, **kwargs)


def instance_segmentation_mxnet(pretrained=True, **kwargs):
    """Instance segmentation with mxnet mobilenet_v2_1_0 backend"""

    return mask_rcnn_fpn_resnet101_v1d_coco_mxnet(pretrained=pretrained, **kwargs)
