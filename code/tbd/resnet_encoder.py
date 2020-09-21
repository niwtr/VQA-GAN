import torch
import numpy as np
from scipy.misc import imread, imresize
from torchvision.models import resnet101
import torch
import torch.nn.functional as F
import torch.nn as nn

def load_resnet_image_encoder(model_stage=2):
    """ Load the appropriate parts of ResNet-101 for feature extraction.

    Parameters
    ----------
    model_stage : Integral
        The stage of ResNet-101 from which to extract features.
        For 28x28 feature maps, this should be 2. For 14x14 feature maps, 3.

    Returns
    -------
    torch.nn.Sequential
        The feature extractor (ResNet-101 at `model_stage`)

    Notes
    -----
    This function will download ResNet-101 if it is not already present through torchvision.
    """
    
    print('Load pretrained ResNet 101.')
    model = resnet101(pretrained=True)
    layers = [model.conv1, model.bn1, model.relu, model.maxpool]
    layers += [getattr(model, 'layer{}'.format(i+1)) for i in range(model_stage)]
    model = torch.nn.Sequential(*layers)
    if torch.cuda.is_available():
        model.cuda()

    for p in model.parameters():
        p.requires_grad = False
    return model.eval()

# def extract_image_feats(img, model, gpus):
#     img_hres = F.interpolate(img, size = (224, 224), mode = 'bicubic', align_corners=False)
#     imin = img_hres.min()
#     imax = img_hres.max()
#     img_hres = (img_hres - imin) / (imax - imin)
#     mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
#     std = torch.FloatTensor([0.229, 0.224, 0.224]).view(1, 3, 1, 1).cuda()
#     img_hres = (img_hres - mean) / std
#     return nn.parallel.data_parallel(model, (img_hres), gpus)


def extract_image_feats(img, model, gpus):
    img_hres = F.interpolate(img, size = (224, 224), mode = 'bicubic', align_corners=False)
    img_hres = (img_hres + 1.) / 2.
    mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.FloatTensor([0.229, 0.224, 0.224]).view(1, 3, 1, 1).cuda()
    img_hres = (img_hres - mean) / std
    return nn.parallel.data_parallel(model, (img_hres), gpus)


def extract_image_feats_(img_path, model):
    """ Extract image features from the image at `img_path` using `model`.

    Parameters
    ----------
    img_path : Union[pathlib.Path, str]
        The path to the image file.

    model : torch.nn.Module
        The feature extractor to use.

    Returns
    -------
    Tuple[numpy.ndarray, torch.Tensor]
        The image and image features extracted from `model`
    """
    # read in the image and transform it to shape (1, 3, 224, 224)
    path = str(img_path) # to handle pathlib
    img = imread(path, mode='RGB')
    img = imresize(img, (224, 224), interp='bicubic')
    img = img.transpose(2, 0, 1)[None]

    # use ImageNet statistics to transform the data
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
    img_tensor = torch.FloatTensor((img / 255 - mean) / std)

    # push to the GPU if possible
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    return (img.squeeze().transpose(1, 2, 0), model(img_tensor))
