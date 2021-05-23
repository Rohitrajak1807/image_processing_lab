import cv2
import matplotlib.pyplot as plt
import numpy as np

import question_2


def calc_hist(img: np.ndarray):
    x = img[:, ]
    plt.hist(x)
    plt.savefig('../outputs/Assignment1/question_5.png')


def main():
    img = question_2.read_image('../res/img.png', cv2.IMREAD_GRAYSCALE)
    calc_hist(img)


if __name__ == '__main__':
    main()
