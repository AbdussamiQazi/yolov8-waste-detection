from ultralytics import YOLO
import os
import math
import torch



def main():
    print(torch.cuda.is_available())

    path = "D:\\Nanomod\\TrashClass-2"
    try:
        contents = os.listdir(path)
        print(contents)
    except OSError as e:
        print(f"Error: {e}")

    model = YOLO("D:\\Nanomod\\runs\detect\\train8\\weights\\last.pt")
    results = model.train(data = "D:\\Nanomod\\TrashClass-2\\data.yaml",device = 0,epochs = 50, save = True, save_period = 2)

if __name__ == '__main__':
    main()
