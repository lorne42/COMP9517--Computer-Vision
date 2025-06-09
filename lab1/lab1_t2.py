import cv2

# 读取彩色图像
image = cv2.imread('./Task2.jpg')

# 检查图像是否成功加载
if image is None:
    print("Error: Could not read the image.")
else:
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 中值滤波
    median_filtered_image = cv2.medianBlur(gray_image, 5)  # 5是滤波器的大小，必须是奇数

    # 创建窗口
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original Image', 600, 700)

    cv2.namedWindow('Median Filtered Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Median Filtered Image', 600, 700)

    # 显示图像
    cv2.imshow('Original Image', gray_image)
    cv2.imshow('Median Filtered Image', median_filtered_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()