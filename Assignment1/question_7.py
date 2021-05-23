import cv2
from question_2 import read_image


def negative(image):
    neg = 1 - image
    cv2.imwrite('../outputs/Assignment1/question_7.png', neg)
    cv2.imshow('Negative', neg)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()


def main():
    negative(read_image('../res/img.png'))


if __name__ == '__main__':
    main()
