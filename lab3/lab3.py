import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# 读取CSV文件
data_dir = './data/data/'  # 替换为你的图片文件夹路径
csv_file = './chinese_mnist.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(csv_file)

# 加载图像数据
images = []
labels = df['character'].values  # 获取标签并转换为NumPy数组

for _, row in df.iterrows():
    # 根据suite_id和sample_id构建文件名
    filename = f"input_{row['suite_id']}_{row['sample_id']}_{row['code']}.jpg"
    img_path = os.path.join(data_dir, filename)

    if os.path.exists(img_path):
        image = imread(img_path)
        # 读取图像并转换为灰度图
        image = resize(image, (64, 64))  # 确保图像大小为64x64
        images.append(image)
    else:
        print(f"Image {img_path} not found.")

images = np.array(images)
labels = np.array(labels)
label_map = {char: idx for idx, char in enumerate(np.unique(labels))}
labels = np.array([label_map[char] for char in labels])
unique_labels, counts = np.unique(labels, return_counts=True)
print(f"Unique labels: {unique_labels}")
print(f"Counts: {counts}")
# 将图像数据展平以适应分类器输入
n_samples, height, width = images.shape
X = images.reshape((n_samples, height * width))

# 使用train_test_split进行分层抽样
X_train, X_test, y_train, y_test = train_test_split(
    X, labels,
    train_size=5000,
    test_size=1000,
    stratify=labels,
    random_state=42
)

# 验证每个类别的数量
train_counts = np.bincount(y_train)
test_counts = np.bincount(y_test)
print(f"Training set class distribution: {train_counts}")
print(f"Test set class distribution: {test_counts}")
# 初始化KNN分类器
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# 初始化决策树分类器
dt_classifier = DecisionTreeClassifier()

# 初始化SGD分类器
sgd_classifier = SGDClassifier(max_iter=250)

# 输出分类器对象，确保参数设置正确
print("KNN Classifier:", knn_classifier)
print("Decision Tree Classifier:", dt_classifier)
print("SGD Classifier:", sgd_classifier)
# 对KNN分类器进行拟合
knn_classifier.fit(X_train, y_train)

# 对决策树分类器进行拟合
dt_classifier.fit(X_train, y_train)

# 对SGD分类器进行拟合
sgd_classifier.fit(X_train, y_train)



# 定义函数用于评估模型性能
def evaluate_model(classifier, X_test, y_test):
    # 预测测试集数据
    y_pred = classifier.predict(X_test)

    # 计算各种评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, f1, cm


# 评估KNN分类器
knn_accuracy, knn_precision, knn_recall, knn_f1, knn_cm = evaluate_model(knn_classifier, X_test, y_test)

# 评估决策树分类器
dt_accuracy, dt_precision, dt_recall, dt_f1, dt_cm = evaluate_model(dt_classifier, X_test, y_test)

# 评估SGD分类器
sgd_accuracy, sgd_precision, sgd_recall, sgd_f1, sgd_cm = evaluate_model(sgd_classifier, X_test, y_test)

# 打印评估结果
print("KNN Classifier Performance:")
print("Accuracy:", knn_accuracy)
print("Precision:", knn_precision)
print("Recall:", knn_recall)
print("F1 Score:", knn_f1)
print("Confusion Matrix:\n", knn_cm)
print("\n")

print("Decision Tree Classifier Performance:")
print("Accuracy:", dt_accuracy)
print("Precision:", dt_precision)
print("Recall:", dt_recall)
print("F1 Score:", dt_f1)
print("Confusion Matrix:\n", dt_cm)
print("\n")

print("SGD Classifier Performance:")
print("Accuracy:", sgd_accuracy)
print("Precision:", sgd_precision)
print("Recall:", sgd_recall)
print("F1 Score:", sgd_f1)
print("Confusion Matrix:\n", sgd_cm)
