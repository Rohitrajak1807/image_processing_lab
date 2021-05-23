import cv2


def read_image(path, flags=None):
    img = cv2.imread(path, flags=flags)
    return img


def display_image(image, win_name='image'):
    cv2.imshow(win_name, image)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()


def write_image(img, path):
    cv2.imwrite(path, img)


def main():
    image = read_image('../res/img.png')
    display_image(image)
    write_image(image, '../outputs/Assignment1/question2.png')


if __name__ == '__main__':
    main()
