# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import datetime
import os

import cv2


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name} ^_^y')  # Press Ctrl+F8 to toggle the breakpoint.


def cpr_colormap(img_path):
    h1 = []
    h2 = []
    origin_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h1.append(origin_img)
    # cv2.imshow('origin_image', origin_img)
    img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2GRAY)
    i = 0
    for k, v in dict.items():
        dst = cv2.applyColorMap(img, v)
        if i < 3:
            h1.append(dst)
        else:
            h2.append(dst)
        i += 1
        # cv2.imshow(k, dst)
    hc1 = cv2.hconcat(h1)
    hc2 = cv2.hconcat(h2)
    full_img = cv2.vconcat([hc1, hc2])
    cv2.imshow('ORIGIN BONE OCEAN PINK VIRIDIS CIVIDIS DEEPGREEN', full_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # hhhhhhh^__*


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # img = cv2.imread('./dataset/testset/BKAI-IGH-NEOPOLYP/masks/0a22abd004c33abf3ae2136cd9dd77ae.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('./dataset/testset/CVC-300/masks/208.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('gray_scale', img)
    cv2.waitKey(0)
    # print_hi('ChocolateNet')
    # num_mask = 100
    # for i in tqdm(range(num_mask), colour='#e946ef'):
    #     time.sleep(1/10)
    # fig = plt.figure(required_interactive_framework='TkAgg')
    # img = cv2.imread('D://Study/pyspace/PraNet/data/TrainDataset/images/200.png', cv2.IMREAD_GRAYSCALE)
    # pseudo1 = cv2.applyColorMap(img, colormap=cv2.COLORMAP_BONE)  # it looks indicarmine
    colormaps = (
        'COLORMAP_AUTUMN',  # 0
        'COLORMAP_BONE',  # 1
        'COLORMAP_JET',  # 2
        'COLORMAP_WINTER',  # 3
        'COLORMAP_RAINBOW',  # 4
        'COLORMAP_OCEAN',  # 5
        'COLORMAP_SUMMER',  # 6
        'COLORMAP_SPRING',  # 7
        'COLORMAP_COOL',  # 8
        'COLORMAP_HSV',  # 9
        'COLORMAP_PINK',  # 10
        'COLORMAP_HOT',  # 11
        'COLORMAP_PARULA',  # 12
        'COLORMAP_MAGMA',  # 13
        'COLORMAP_INFERNO',  # 14
        'COLORMAP_PLASMA',  # 15
        'COLORMAP_VIRIDIS',  # 16
        'COLORMAP_CIVIDIS',  # 17
        'COLORMAP_TWILIGHT',  # 18
        'COLORMAP_TWILIGHT_SHIFTED',  # 19
        'COLORMAP_TURBO',  # 20
        'COLORMAP_DEEPGREEN'  # 21
    )
    # chosen color
    dict = {
        'COLORMAP_BONE': 1,
        # 'COLORMAP_WINTER': 3,  # canceled
        'COLORMAP_OCEAN': 5,
        # 'COLORMAP_SUMMER': 6,
        'COLORMAP_PINK': 10,
        'COLORMAP_VIRIDIS': 16,
        'COLORMAP_CIVIDIS': 17,
        'COLORMAP_DEEPGREEN': 21
    }
    base_path = 'D://Study/pyspace/PraNet/data/TrainDataset/images/'
    img_paths = [base_path + img for img in os.listdir(base_path) if img.endswith('.png')]
    for img_path in img_paths:
        cpr_colormap(img_path)
