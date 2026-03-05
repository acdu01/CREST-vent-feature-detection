import cv2

image = cv2.imread("img_test/cyber_rabbit_depthmap.jpg")
if image is None:
    raise FileNotFoundError("Image not found or path is wrong")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("img_test/greyscale_depth/greyscale_depth.jpg", gray_image)


