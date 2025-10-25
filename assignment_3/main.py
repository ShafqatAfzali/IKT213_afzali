import cv2 as cv
import numpy as np


def main(function_name):
    img=cv.imread("lambo.png")
    img2 = cv.imread("shapes-1.png")
    template=cv.imread("shapes_template.jpg",0)
    if function_name == "sobel_edge_detection":
        cv.imshow("lambos detected sobel edge",sobel_edge_detection(img))
        cv.waitKey(0)
        cv.destroyWindow("lambos detected sobel edge")
    elif function_name == "canny_edge_detection":
        cv.imshow("lambos detected canny edge",canny_edge_detection(img,50,50))
        cv.waitKey(0)
        cv.destroyWindow("lambos detected canny edge")
    elif function_name == "template_match":
        cv.imshow("matched shapes",template_match(img2,template))
        cv.waitKey(0)
        cv.destroyWindow("matched shapes")
    elif function_name == "resize":
        cv.imshow("resized lambo, scaled up",resize(img,2,"up"))
        cv.waitKey(0)
        cv.destroyWindow("resized lambo, scaled up")


def sobel_edge_detection(image):
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (3, 3), 0)
    img_sobel = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=1)
    cv.imwrite("lambo_sobel.png", img_sobel)
    return img_sobel


def canny_edge_detection(image, threshold_1, threshold_2):
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (3, 3), 0)
    img_canny = cv.Canny(image=img_blur, threshold1=threshold_1, threshold2=threshold_2)
    cv.imwrite("lambo_canny.png", img_canny)
    return img_canny


def template_match(image, template):
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    width, height = template.shape[::-1]
    result = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(result >= threshold)

    for i in zip(*loc[::-1]):
        cv.rectangle(image, i, (i[0] + width, i[1] + height), (0, 0, 255), 2)

    cv.imwrite("matched_shapes.png", image)

    return image


def resize(image, scale_factor:int, up_or_down:str):

    resized = image.copy()

    if up_or_down == "up":
        for i in range(scale_factor):
            resized = cv.pyrUp(resized, dstsize=(resized.shape[1] * 2, resized.shape[0] * 2))

    elif up_or_down == "down":
        for i in range(scale_factor):
            resized = cv.pyrDown(resized, dstsize=(resized.shape[1] // 2, resized.shape[0] // 2))

    cv.imwrite("lambo_resized_up.png", resized)
    return resized


#main("sobel_edge_detection")
#main("canny_edge_detection")
main("resize")
