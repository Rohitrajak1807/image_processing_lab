import numpy as np
from question_2 import read_image, write_image, display_image


def pixel_val(pix, r1, s1, r2, s2):
    if 0 <= pix <= r1:
        return (s1 / r1) * pix
    elif r1 < pix <= r2:
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2) / (255 - r2)) * (pix - r2) + s2


def contrast_stretch(image):
    r1 = 70
    s1 = 0
    r2 = 140
    s2 = 255
    pixel_vec = np.vectorize(pixel_val)
    contrast_stretched = pixel_vec(image, r1, s1, r2, s2)
    display_image(image, win_name='original')
    display_image(contrast_stretched, win_name='output')
    write_image(contrast_stretched, '../outputs/Assignment1/question_10.png')


def main():
    contrast_stretch(read_image('../res/img.png'))


if __name__ == '__main__':
    main()
