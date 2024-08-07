# You may be able to use a lot of the code in my original utils library,
# but I recommend trying to implement them yourself first to practice.


import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2


def loadImage(filename, asTensor=False):
    '''
    Load an image from a file

    inputs
    filename: path to image file
    asTensor: if True, return image as PyTorch Tensor, else return as NumPy array

    output
    image: image as PyTorch Tensor or NumPy array
    '''
    
    image = Image.open(filename).convert('RGB')
    image = np.array(image)
    if asTensor:
        image = imageToTensor(image)
        #image = image.unsqueeze(0)
    return image



def saveImage(image, filename, isTensor=False):
    '''
    Save an image to a file

    inputs
    image: image as PyTorch Tensor or NumPy array
    filename: path to save image file
    isTensor: if True, image is a PyTorch Tensor, else image is a NumPy array
    '''
    
    if isTensor:
        image = tensorToImage(image)
    image = Image.fromarray(image)
    image.save(filename)



def loadMask(filename, asTensor=False):
    '''
    Load a mask from a file

    inputs
    filename: path to mask file
    asTensor: if True, return mask as PyTorch Tensor, else return as NumPy array

    output
    mask: mask as PyTorch Tensor or NumPy array
    '''
    
    mask = Image.open(filename).convert('L')
    mask = np.array(mask)
    if asTensor:
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
    return mask



def saveMask(mask, filename, isTensor=False):
    '''
    Save a mask to a file

    inputs
    mask: mask as PyTorch Tensor or NumPy array
    filename: path to save mask file
    isTensor: if True, mask is a PyTorch Tensor, else mask is a NumPy array
    '''
    
    if isTensor:
        mask = tensorToMask(mask)
    mask = Image.fromarray(mask)
    mask.save(filename)



def imageToTensor(image,addBatch=True):
    '''
    Convert an image from a NumPy array to a PyTorch Tensor

    inputs
    image: image as NumPy array

    output
    tensor: image as PyTorch Tensor
    '''
    
    image = np.transpose(image, (2, 0, 1))  # Convert HxWxC to CxHxW
    if np.amax(image) > 1.0:
        image = image/255.0
    
    result = torch.from_numpy(image).float()
    if addBatch:
        result = result.unsqueeze(0)
    return result



def tensorToImage(tensor):
    '''
    Convert an image from a PyTorch Tensor to a NumPy array

    inputs
    tensor: image as PyTorch Tensor

    output
    image: image as NumPy array
    '''
    
    image = tensor.detach().cpu().numpy() * 255.0
    image = np.clip(image,0,255)
    image = image.squeeze(0)
    image = np.transpose(image, (1, 2, 0)).astype(np.uint8)  # Convert CxHxW to HxWxC
    return image



def toTensor(image):
    transform = transforms.ToTensor()
    return transform(image)



def toNumpy(tensor):
    if tensor.dim() == 4:  # Tensor with shape (B, C, H, W)
        tensor = tensor.squeeze(0)  # Remove the batch dimension if it exists
    if tensor.dim() == 3:  # Tensor with shape (C, H, W)
        return tensor.detach().cpu().numpy().transpose(1, 2, 0)
    else:
        raise ValueError("Unsupported tensor shape: {}".format(tensor.shape))



def maskToTensor(mask):
    '''
    Convert a mask from a NumPy array to a PyTorch Tensor

    inputs
    mask: mask as NumPy array

    output
    tensor: mask as PyTorch Tensor
    '''
    
    mask = torch.from_numpy(mask).unsqueeze(0).float()

    if torch.amax(mask) > 1.0:
        mask = mask/255.0

    return mask



def tensorToMask(tensor):
    '''
    Convert a mask from a PyTorch Tensor to a NumPy array

    inputs
    tensor: mask as PyTorch Tensor

    output
    mask: mask as NumPy array
    '''
    
    mask = tensor.detach().cpu().squeeze().numpy() * 255.0
    return mask.astype(np.uint8)



#def applyMask(image, mask):

    #alpha = np.mean(mask_im[:, :, :3], axis=2)  # Convert RGBA to grayscale
    #alpha[alpha > 0] = 1.0  # Set non-transparent regions to fully opaque
    #alpha = np.expand_dims(alpha, axis=2)  
    #return alpha

    #mask_temp = mask.squeeze()
    #mask_temp = cv2.resize(mask_temp.numpy(),(result.shape[-1],result.shape[-2]))
    #mask = torch.from_numpy(mask_temp).unsqueeze(0)
       
    #return image*mask



def applyMaskTensor(result, mask):

    # Check that mask is passed in
    if mask is None:
        return result

    # Check that mask and result are the same size
    if result.shape[-2] != mask.shape[-2] or result.shape[-1] != mask.shape[-1]:
        resizer = transforms.Resize((result.shape[-2],result.shape[-1]),transforms.InterpolationMode.NEAREST_EXACT,antialias=False)
        mask = resizer(mask)

    # Check that they have the same dimesions
    if mask.ndim < result.ndim:
        mask = mask.unsqueeze(-3)

    return result*mask



def alphaBlend(image, mask, background):
    '''
    Alpha blend an image with a mask

    inputs
    image: image as PyTorch Tensor or NumPy array
    mask: mask as PyTorch Tensor or NumPy array
    background: background image as PyTorch Tensor or NumPy array

    output
    blended: image alpha blended with background as PyTorch Tensor or NumPy array
    '''

    alpha = mask / mask.max()

    if isinstance(image, np.ndarray):
        alpha = cv2.resize(alpha,(image.shape[1],image.shape[0]))
        background = cv2.resize(background,(image.shape[1],image.shape[0]))

    # This makes sure the mask and image have the same number of dimensions
    if alpha.ndim < image.ndim:
        alpha = alpha[...,None]


    blended = image * alpha + background * (1 - alpha)
    return blended
