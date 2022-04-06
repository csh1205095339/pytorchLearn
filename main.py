# This is a sample Python script.
import torch
import torchvision
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x = np.arange(0, 6, 0.1)
    print(x)
    y1 = np.sin(x)
    y2 = np.cos(x)
    plt.plot(x, y1, label="sin")
    plt.plot(x, y2, linestyle="--", label="cos")  # 用虚线绘制
    plt.xlabel("x")  # x轴标签
    plt.ylabel("y")  # y轴标签
    plt.title('sin & cos')  # 标题
    plt.legend()
    plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
