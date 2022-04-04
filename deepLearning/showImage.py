from matplotlib import pyplot as plt
from matplotlib.image import imread

if __name__ == '__main__':
    img = imread('D:\pictures\lena.png')  # 读入图像（设定合适的路径！）
    plt.imshow(img)
    plt.show()