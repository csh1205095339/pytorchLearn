from torchvision import transforms
import cv2 as cv
from torch.utils.tensorboard import SummaryWriter
# python的用法 -> tensor数据类型
# 通过 transforms.ToTensor去看两个问题
# 1、 transforms应该如何使用
"""
 2、 为什么我们需要Tensor的数据类型
 
 
"""
writer = SummaryWriter("logsTrans")


image = cv.imread(r"D:\program\python\pytorchLearn\dataSet\testdataset\train\ants\5650366_e22b7e1065.jpg")
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(image)
writer.add_image("Tensor_img", tensor_img)
writer.add_image("image", image, dataformats="HWC")
writer.close()
print(image)
print(tensor_img)