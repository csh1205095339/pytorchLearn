from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
img = Image.open(r"D:\program\python\pytorchLearn\dataSet\testdataset\train\ants\5650366_e22b7e1065.jpg")
print(img)
writer = SummaryWriter("logsTrans")
# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("Tensor_img", img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
writer.add_image("img_norm", img_norm)
print(img_norm[0][0][0])


# Resize
print(img.size)
trans_size = transforms.Resize((512, 512))
img_resize = trans_size(img)
img_resize = trans_totensor(img_resize)
writer.add_image("img_resize", img_resize)

writer.close()