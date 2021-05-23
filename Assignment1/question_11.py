import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('../res/img.png', cv2.IMREAD_GRAYSCALE)
    for i in range(8):
        bit_plane = np.array(((img & (1 << i)) >> i) * 255, dtype=np.uint8)
        cv2.imwrite(f'../outputs/Assignment1/question_11/bit_plane{i}.png', bit_plane)
