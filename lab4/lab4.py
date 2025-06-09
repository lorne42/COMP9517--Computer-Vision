import cv2
import numpy as np
from skimage import measure, morphology, segmentation
import matplotlib.pyplot as plt

# 加载图像
image_path = './Leaves.png'
image = cv2.imread(image_path)

# 转换为HSV颜色空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义浅蓝色的颜色范围
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([100, 255, 255])

# 颜色过滤，保留浅蓝色区域
mask = cv2.inRange(hsv, lower_blue, upper_blue)
result = cv2.bitwise_and(image, image, mask=mask)

# 转换为灰度图像
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# 二值化处理
_, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# 应用二值形态学操作
# 定义结构元素
selem = morphology.disk(1)

# 开运算 (先腐蚀后膨胀)
opened = morphology.opening(binary, selem)

# 闭运算 (先膨胀后腐蚀)
closed = morphology.closing(opened, selem)

# 填充对象内部的孔洞
closed_bool = closed.astype(bool)
filled = morphology.remove_small_holes(closed_bool, area_threshold=0)

# 移除边界上的对象
boundary_cleared = segmentation.clear_border(filled)

# 移除小对象以确保最终结果
size_threshold = 0  # 根据需要调整
final_cleaned = morphology.remove_small_objects(boundary_cleared, min_size=size_threshold)

# 标记对象
labeled_image = measure.label(final_cleaned)

# 统计对象数量
num_objects = labeled_image.max()

# 显示结果
print(f"对象数量: {num_objects}")

# 保存结果
result_path = '/mnt/data/segmented_result.png'
cv2.imwrite(result_path, (final_cleaned * 255).astype(np.uint8))
print(f"分割结果已保存到 {result_path}")

# 可视化分割结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Result')
plt.imshow(final_cleaned, cmap='gray')
plt.axis('off')

plt.show()
