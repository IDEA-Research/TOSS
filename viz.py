import torch
import cv2
 
 
def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    Save a tensor to a file as an image.
    :param input_tensor: tensor to save [C, H, W]
    :param filename: file to save to
    """
    assert (len(input_tensor.shape) == 3)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)
