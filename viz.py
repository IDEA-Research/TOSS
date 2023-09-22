import torch
import cv2
 
 
def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为cv2格式 c,h,w
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 3)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)
