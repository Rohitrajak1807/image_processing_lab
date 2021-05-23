import numpy as np
from question_2 import read_image, write_image


def gamma_correction(image):
    for gamma in [0.1, 0.5, 1.2, 2.2]:
        gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype=np.uint8)
        write_image(gamma_corrected, f'../outputs/Assignment1/question_9-{gamma}.png')


def main():
    gamma_correction(read_image('../res/img.png'))


if __name__ == '__main__':
    main()
