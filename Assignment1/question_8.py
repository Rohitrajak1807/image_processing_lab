import cv2
import numpy as np
from question_2 import read_image


def main():
    log_transform(read_image('../res/img.png'))


def log_transform(image):
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(image + 1))
    log_image = np.array(log_image, dtype=np.uint8)
    cv2.imshow('log img', log_image)
    cv2.imwrite('../outputs/Assignment1/question_8.png', log_image)


if __name__ == '__main__':
    main()
