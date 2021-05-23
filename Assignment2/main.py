import cv2
import numpy as np


def question_1():
    path = '../res/image1.jpg'
    img = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
    sobel_horizontal = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 2]
    ], dtype=np.int8)
    img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    sobel_vertical = sobel_horizontal.transpose()

    out_horizontal_small = cv2.filter2D(src=img_small, kernel=sobel_horizontal, ddepth=-1)
    out_vertical_small = cv2.filter2D(src=img_small, kernel=sobel_vertical, ddepth=-1)
    sbs = np.concatenate((img_small, out_horizontal_small, out_vertical_small), axis=1)
    cv2.imshow('Sobel filter', sbs)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

    out_horizontal = cv2.filter2D(src=img, kernel=sobel_horizontal, ddepth=-1)
    out_vertical = cv2.filter2D(src=img, kernel=sobel_vertical, ddepth=-1)
    cv2.imshow('orig', out_horizontal)
    cv2.imshow('orig2', out_vertical)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    cv2.imwrite('../outputs/Assignment2/question_1/sobel_h.jpg', out_horizontal)
    cv2.imwrite('../outputs/Assignment2/question_1/sobel_v.jpg', out_vertical)

    v = np.array(out_horizontal, dtype=np.uint64)
    h = np.array(out_vertical, dtype=np.uint64)
    magnitude = v ** 2 + h ** 2
    magnitude = np.sqrt(magnitude)
    magnitude = np.clip(magnitude, 0, 255)
    out = np.array(magnitude, dtype=np.uint8)
    cv2.imshow('mag', out)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    cv2.imwrite('../outputs/Assignment2/question_1/magnitude.jpg', out)

    _, out_thresh_1 = cv2.threshold(out, 1, 255, cv2.THRESH_BINARY)
    _, out_thresh_2 = cv2.threshold(out, 2, 255, cv2.THRESH_BINARY)
    _, out_thresh_4 = cv2.threshold(out, 4, 255, cv2.THRESH_BINARY)
    _, out_thresh_8 = cv2.threshold(out, 8, 255, cv2.THRESH_BINARY)
    _, out_thresh_16 = cv2.threshold(out, 16, 255, cv2.THRESH_BINARY)
    _, out_thresh_32 = cv2.threshold(out, 32, 255, cv2.THRESH_BINARY)
    _, out_thresh_64 = cv2.threshold(out, 64, 255, cv2.THRESH_BINARY)
    _, out_thresh_128 = cv2.threshold(out, 128, 255, cv2.THRESH_BINARY)
    cv2.imwrite('../outputs/Assignment2/question_1/magnitude_1.jpg', out_thresh_1)
    cv2.imwrite('../outputs/Assignment2/question_1/magnitude_2.jpg', out_thresh_2)
    cv2.imwrite('../outputs/Assignment2/question_1/magnitude_4.jpg', out_thresh_4)
    cv2.imwrite('../outputs/Assignment2/question_1/magnitude_8.jpg', out_thresh_8)
    cv2.imwrite('../outputs/Assignment2/question_1/magnitude_16.jpg', out_thresh_16)
    cv2.imwrite('../outputs/Assignment2/question_1/magnitude_32.jpg', out_thresh_32)
    cv2.imwrite('../outputs/Assignment2/question_1/magnitude_64.jpg', out_thresh_64)
    cv2.imwrite('../outputs/Assignment2/question_1/magnitude_128.jpg', out_thresh_128)

    h_float = h + 0.0000001
    tan_inv = np.arctan(v / h_float)
    tan_inv = np.clip(tan_inv, 0, 255)
    tan_inv = tan_inv / tan_inv.max(initial=0)
    tan_inv = tan_inv * 255
    out_dir = np.array(tan_inv, dtype=np.uint8)
    cv2.imshow('dir', out_dir)
    cv2.waitKey(10000)
    cv2.imwrite('../outputs/Assignment2/question_1/dir_norm.jpg', out_dir)


def get_pixel(image, center, x, y):
    val = 0
    try:
        if image[x, y] >= center:
            val = 1
    except Exception:
        pass
    return val


def lbp(image, x, y):
    center = image[x, y]
    values = [get_pixel(image, center, x - 1, y - 1), get_pixel(image, center, x - 1, y),
              get_pixel(image, center, x - 1, y + 1), get_pixel(image, center, x, y + 1),
              get_pixel(image, center, x + 1, y + 1), get_pixel(image, center, x + 1, y),
              get_pixel(image, center, x + 1, y - 1), get_pixel(image, center, x, y - 1)]
    powers = [1, 2, 4, 8, 16, 32, 64, 128]
    out_val = 0
    for idx in range(len(values)):
        out_val += powers[idx] * values[idx]
    return out_val


def question_2():
    res = '../res/image1.jpg'
    img = cv2.imread(res, cv2.IMREAD_GRAYSCALE)
    out = np.zeros(img.shape, dtype=np.uint8)

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = lbp(img, i, j)

    cv2.imshow('lbp', out)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    cv2.imwrite('../outputs/Assignment2/question_2/lbp.jpg', out)


if __name__ == '__main__':
    question_1()
    question_2()
