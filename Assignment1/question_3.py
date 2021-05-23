import cv2
import question_2
import numpy as np


def to_greyscale(path):
    img = question_2.read_image(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_grey(path: str):
    img = question_2.read_image(path)
    kernel = [0.1140, 0.5870, 0.2989]
    img = np.dot(img[..., :3], kernel)
    img = np.asarray(img, dtype=np.uint8)
    return img


def main():
    img = to_greyscale('../res/img.png')
    question_2.display_image(img)
    question_2.write_image(img, '../outputs/Assignment1/question_3_greyscale.png')
    grey = to_grey('../res/img.png')
    question_2.display_image(grey, 'custom function')
    question_2.write_image(grey, '../outputs/Assignment1/question_3_greyscale_custom.png')


if __name__ == '__main__':
    main()
