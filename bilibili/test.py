from torch.utils.data import Dataset
import cv2 as cv
import os

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_path_list = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path_list[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = cv.imread(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path_list)



if __name__ == "__main__":
    root_dir = r"D:\program\python\pytorchLearn\dataSet\testdataset\train"
    ants_label_dir = "ants"
    bees_label_dir = "bees"
    ants_dataset = MyData(root_dir, ants_label_dir)
    bees_dataset = MyData(root_dir, bees_label_dir)
    ant, ant_label = ants_dataset[0]
    bee, bee_label = bees_dataset[0]
    cv.imshow(ant_label, ant)
    cv.imshow(bee_label, bee)
    cv.waitKeyEx(0)
    cv.destroyAllWindows()