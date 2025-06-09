

import cv2

# 加载图像

cv2.namedWindow('Input', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Input', 900, 600)
cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Output', 900, 600)

# 对图像应用高斯模糊
image = cv2.imread('./Task3.jpg', cv2.IMREAD_GRAYSCALE)

# 对图像应用高斯模糊，生成平滑版本
blurred_image = cv2.GaussianBlur(image, (9, 9), 10.0)

# 计算高频细节
high_pass = cv2.addWeighted(image, 1.5, blurred_image, -0.5, 0)

# 显示原始模糊图像
cv2.imshow('Input', image)

# 显示锐化后的图像
cv2.imshow('Output', high_pass)

# 等待按键并关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
