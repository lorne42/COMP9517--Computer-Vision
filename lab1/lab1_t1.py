import cv2
import numpy as np

# 读取彩色图像
image = cv2.imread('./Task1.jpg')

# 检查图像是否成功加载
if image is None:
    print("Error: Could not read the image.")
else:
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 强度反转
    inverted_image = 255 - gray_image

    # 伽马校正
    gamma = 5  # 你可以调整这个值
    gamma_corrected_image = np.power(inverted_image / 255.0, gamma) * 255.0
    gamma_corrected_image = np.uint8(gamma_corrected_image)

    # 创建窗口
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original Image', 900, 600)



    cv2.namedWindow('Gamma Corrected Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gamma Corrected Image', 900, 600)
    # 显示图像
    cv2.imshow('Original Image', gray_image)
    cv2.imshow('Gamma Corrected Image', gamma_corrected_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()