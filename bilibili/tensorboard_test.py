from torch.utils.tensorboard import SummaryWriter
import cv2 as cv
writer = SummaryWriter("logs")
image = cv.imread(r"D:\program\python\pytorchLearn\dataSet\testdataset\train\ants\5650366_e22b7e1065.jpg")
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
writer.add_image("test", image, 2, dataformats="HWC")

writer.close()