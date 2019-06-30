## A somewhat curated repo for pytorch-hub model definitions

[PyTorch Hub](https://pytorch.org/docs/stable/hub.html) definitions with stable entrypoints aliasing the state of the art machine learning models.

**PRs welcome**


## Why?
The state of the art changes all the time and we want to be able to be able to use the latest and greatest models. Sometimes we care about the very best accuracy, other times we just need a faster model that is ready for some fine-tuning.

PyTorch Hub enables us to have a consistent loading experience for the models from multiple repositories / libraries. Plus this repo is a nice opportunity to have a PyTorch Hub playground.


## Usage

```py
import torch

from PIL import Image
from torchvision import transforms

# for generic use
object_recognition = torch.hub.load('jmargeta/awesome-pytorch-hub:master', 'object_recognition', pretrained=False)

# for mobile devices
object_recognition_faster = torch.hub.load('jmargeta/awesome-pytorch-hub:master', 'object_recognition', pretrained=False)

# when that extra accuracy is needed
object_recognition_sota = torch.hub.load('jmargeta/awesome-pytorch-hub:master', 'object_recognition', pretrained=False)

# load and process an image, e.g.
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# do the object recognition
im = transform(Image.open('test_image.jpg'))
pred = object_recognition(im.unsqueeze(0))

top_5_probas, top_5_indices = pred[0].topk(5)
print(f'Predicted class indices: {top_5_indices.tolist()}')
```

