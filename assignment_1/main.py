import cv2 as cv


def print_image_information(image):
    my_img = cv.imread(image)
    height, width, channels = my_img.shape
    print("height:", height)
    print("width:", width)
    print("channels:", channels)
    print("size:", my_img.size)
    print("data type:", my_img.dtype)
    cv.imshow("lena-1 displayed", my_img)
    cv.waitKey(0)
    cv.destroyWindow("lena-1 displayed")
    return 0

def SaveMyWeb(filepath):
    webcam=cv.VideoCapture(0)
    fps = webcam.get(cv.CAP_PROP_FPS)
    width = int(webcam.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(webcam.get(cv.CAP_PROP_FRAME_HEIGHT))

    with open(filepath, "a") as file:
        file.write("fps: \t" + str(fps) + "\n")
        file.write("width: \t" + str(width) + "\n")
        file.write("height: \t" + str(height) + "\n")

    webcam.release()



print_image_information('lena-1.png')
SaveMyWeb('camera_outputs.txt')
