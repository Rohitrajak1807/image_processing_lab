import cv2
import numpy as np


def question_1():
    path = '../res/house.jpg'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    sigmas = [
        1, 2, 3
    ]
    laplacian = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ], dtype=np.int8)
    the_other_laplacian = np.array([
        [-1, 2, -1],
        [2, -4, 2],
        [-1, 2, -1]
    ], dtype=np.int8)

    for sigma in sigmas:
        kernel_2d = cv2.getGaussianKernel(3, sigma)
        kernel = kernel_2d * kernel_2d.T
        gaussed = cv2.filter2D(src=img, kernel=kernel, ddepth=-1)
        laplaced = cv2.filter2D(src=gaussed, ddepth=-1, kernel=laplacian)
        the_other_laplaced = cv2.filter2D(src=gaussed, ddepth=-1, kernel=the_other_laplacian)
        cv2.imshow('log', laplaced)
        cv2.imshow('the other logged', the_other_laplaced)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()
        cv2.imwrite(f'../outputs/Assignment3/question_1/laplaced_sigma_{sigma}.jpg', laplaced)
        cv2.imwrite(f'../outputs/Assignment3/question_1/the_other_laplaced_sigma_{sigma}.jpg', the_other_laplaced)


def get_pixel(image, center, x, y, t):
    val = 0
    try:
        if image[x, y] > center + t:
            val = 1
        elif center + t > image[x, y] > center - t:
            val = 0
        else:
            val = -1
    except Exception:
        pass
    return val


def ltp(image, x, y, t):
    center = image[x, y]
    values = [get_pixel(image, center, x - 1, y - 1, t=5), get_pixel(image, center, x - 1, y, t=5),
              get_pixel(image, center, x - 1, y + 1, t=5), get_pixel(image, center, x, y + 1, t=5),
              get_pixel(image, center, x + 1, y + 1, t=5), get_pixel(image, center, x + 1, y, t=5),
              get_pixel(image, center, x + 1, y - 1, t=5), get_pixel(image, center, x, y - 1, t=5)]
    powers = [1, 2, 4, 8, 16, 32, 64, 128]
    out_val = 0
    for idx in range(len(values)):
        out_val += powers[idx] * values[idx]
    return out_val


def question_2(t=5):
    res = '../res/img3.jpg'
    img = cv2.imread(res, cv2.IMREAD_GRAYSCALE)
    out = np.zeros(img.shape, dtype=np.uint8)

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = ltp(img, i, j, t)

    cv2.imshow('ltp', out)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    cv2.imwrite('../outputs/Assignment3/question_2/ltp.jpg', out)


if __name__ == '__main__':
    question_1()
    question_2()
