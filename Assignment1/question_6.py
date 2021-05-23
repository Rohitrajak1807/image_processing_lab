import cv2
import numpy as np

from question_2 import read_image, write_image, display_image


def calc_hist(image, bins):
    histogram = np.zeros(bins)
    for pixel in image:
        histogram[pixel] += 1
    return histogram


def cumulative_freq(histogram):
    b = [histogram[0]]
    for i in histogram:
        b.append(b[-1] + i)
    return np.array(b)


def equalize_hist(img, flat, cs):
    n = (cs - cs.min()) * 255
    N = cs.max() - cs.min()
    cs = n / N
    cs = cs.astype('uint8')
    equalized = cs[flat]
    equalized = np.reshape(equalized, img.shape)
    return equalized


def main():
    img = read_image('../res/img.png', cv2.IMREAD_GRAYSCALE)
    flat_img_array = img.flatten()
    hist = calc_hist(flat_img_array, 256)
    cumulative = cumulative_freq(hist)
    equalized = equalize_hist(img, flat_img_array, cumulative)
    display_image(img, 'original')
    display_image(equalized, 'equalized')
    cv2.waitKey()
    cv2.destroyAllWindows()
    write_image(equalized, '../outputs/Assignment1/question_6.png')


if __name__ == '__main__':
    main()
