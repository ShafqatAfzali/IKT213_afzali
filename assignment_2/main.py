import cv2 as cv
import numpy as np

#to run each function just run the main function with the image path as the first -
# and the function name as the second variable in string format.
#example i want to run crop function:
    #main("lena-1.png", "crop")

#for the rotation function the name of the function is rotation but for main -
# to run it you have to either write main("lena-1.png", "rotation_90") or
# main("lena-1.png", "rotation_180") to specify how much you want to rotate the image.

def main(img_path,function):
    img=cv.imread(img_path)

    if function=="padding":
        cv.imshow("lena image with border",padding(img,20))
        cv.waitKey(0)
        cv.destroyWindow("lena image with border")
    elif function=="crop":
        cv.imshow("lena image cropped",crop(img,80,130,80,130))
        cv.waitKey(0)
        cv.destroyWindow("lena image cropped")
    elif function=="resize":
        cv.imshow("lena image resized",resize(img,200,200))
        cv.waitKey(0)
        cv.destroyWindow("lena image resized")
    elif function=="copy":
        height, width, channels = img.shape
        emptyPictureArray = np.zeros((height, width, channels), dtype=np.uint8)
        cv.imshow("lena image manual copy",copy(img,emptyPictureArray))
        cv.waitKey(0)
        cv.destroyWindow("lena image manual copy")
    elif function=="grayscale":
        cv.imshow("grayscale of lena image",grayscale(img))
        cv.waitKey(0)
        cv.destroyWindow("grayscale of lena image")
    elif function=="hsv":
        cv.imshow("hsv of lena image",hsv(img))
        cv.waitKey(0)
        cv.destroyWindow("hsv of lena image")
    elif function=="hue_shifted":
        height, width, channels = img.shape
        emptyPictureArray = np.zeros((height, width, channels), dtype=np.uint8)
        cv.imshow("lena image hue shifted by 50",hue_shifted(img,emptyPictureArray, 50))
        cv.waitKey(0)
        cv.destroyWindow("lena image hue shifted by 50")
    elif function=="smoothing":
        cv.imshow("smoothing filter applied to lena image",smoothing(img))
        cv.waitKey(0)
        cv.destroyWindow("smoothing filter applied to lena image")
    elif function=="rotation_90":
        cv.imshow("lena rotated 90 degrees",rotation(img,90))
        cv.waitKey(0)
        cv.destroyWindow("lena rotated 90 degrees")
    elif function=="rotation_180":
        cv.imshow("lena rotated 180 degrees",rotation(img,180))
        cv.waitKey(0)
        cv.destroyWindow("lena rotated 180 degrees")

def padding(image, border_width):
    img_with_border = cv.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv.BORDER_REFLECT)
    cv.imwrite("lena_padded.png", img_with_border)
    return img_with_border

def crop(image, x_0, x_1,  y_0, y_1):
    height, width, channels = image.shape
    cropped = image[y_0:(height-y_1), x_0:(width-x_1)]
    cv.imwrite("lena_croped.png", cropped)
    return cropped

def resize(image, width, height):
    img_resized = cv.resize(image, (width, height), interpolation=cv.INTER_CUBIC)
    cv.imwrite("lena_resized.png", img_resized)
    return img_resized

def copy(image, emptyPictureArray):
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y, x, c] = image[y, x, c]
    cv.imwrite("lena_manual_copy.png", emptyPictureArray)
    return emptyPictureArray

def grayscale(image):
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imwrite("lena_gray.png", gray_img)
    return gray_img

def hsv(image):
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imwrite("lena_hsv.png", hsv_image)
    return hsv_image


def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                current_value = int(image[y, x, c])
                new_value = current_value + hue

                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                else:
                    new_value = new_value
                emptyPictureArray[y, x, c] = np.uint8(new_value)

    cv.imwrite("shifted_lena.png", emptyPictureArray)
    return emptyPictureArray

def smoothing(image):
    smooth_lena = cv.GaussianBlur(image, (15, 15), cv.BORDER_DEFAULT)
    cv.imwrite("lena_smooth.png", smooth_lena)
    return smooth_lena

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated_lena = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
        cv.imwrite("lena_rotatetd_90.png", rotated_lena)
    elif rotation_angle == 180:
        rotated_lena = cv.rotate(image, cv.ROTATE_180)
        cv.imwrite("lena_rotatetd_180.png", rotated_lena)
    return rotated_lena




main("lena-1.png", "hue_shifted")