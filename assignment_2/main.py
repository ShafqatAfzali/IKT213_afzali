import cv2 as cv

def main(img_path,function):
    img=cv.imread(img_path)

    if function=="padding":
        cv.imshow("lena image with border",padding(img,20))
        cv.waitKey(0)
        cv.destroyWindow("lena image with border")
    elif function=="crop":
        cv.imshow("lena image with border",crop(img,80,130,80,130))
        cv.waitKey(0)



def padding(image, border_width):
    img_with_border = cv.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv.BORDER_CONSTANT)
    return img_with_border


def crop(image, x_0, x_1,  y_0, y_1):
    height, width, channels = image.shape
    cropped = image[y_0:(height-y_1), x_0:(width-x_1)]
    return cropped


main("lena-1.png","padding")