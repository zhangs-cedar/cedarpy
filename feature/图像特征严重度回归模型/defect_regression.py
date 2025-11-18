import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, filters, morphology
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from functools import partial
import cv2

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_defect_image(size=(50, 50), defect_type='crack', severity=0.5):
    """创建人工缺陷图像

    Args:
        size: 图像尺寸
        defect_type: 缺陷类型 ('crack', 'hole', 'scratch', 'stain')
        severity: 严重度 (0-1)

    Returns:
        image: 缺陷图像
    """
    img = np.ones(size, dtype=np.float32) * 0.8  # 基础背景

    if defect_type == 'crack':
        # 裂纹：从中心向外的直线
        center = (size[0] // 2, size[1] // 2)
        length = int(20 * severity)
        angle = np.random.uniform(0, 2 * np.pi)
        end_x = int(center[0] + length * np.cos(angle))
        end_y = int(center[1] + length * np.sin(angle))

        # 画裂纹
        cv2.line(img, center, (end_x, end_y), 0.2, thickness=max(1, int(3 * severity)))

    elif defect_type == 'hole':
        # 孔洞：圆形缺陷
        center = (size[0] // 2, size[1] // 2)
        radius = int(8 * severity)
        cv2.circle(img, center, radius, 0.1, -1)

    elif defect_type == 'scratch':
        # 划痕：多条平行线
        for i in range(int(3 * severity)):
            y = size[0] // 2 + i * 3
            cv2.line(img, (5, y), (size[1] - 5, y), 0.3, thickness=1)

    elif defect_type == 'stain':
        # 污渍：不规则形状
        center = (size[0] // 2, size[1] // 2)
        radius = int(12 * severity)
        # 添加噪声模拟污渍
        noise = np.random.normal(0, 0.1 * severity, size)
        img += noise
        # 圆形污渍
        cv2.circle(img, center, radius, 0.4, -1)

    return np.clip(img, 0, 1)


def extract_features(img):
    """提取图像特征用于回归

    Args:
        img: 输入图像

    Returns:
        features: 特征向量
    """
    features = []

    # 1. 基础统计特征
    features.extend([np.mean(img), np.std(img), np.min(img), np.max(img), np.median(img)])

    # 2. 纹理特征 (LBP)
    lbp = feature.local_binary_pattern(img, 8, 1, method='uniform')
    features.extend(
        [
            np.mean(lbp),
            np.std(lbp),
            np.histogram(lbp, bins=10)[0].sum(),  # LBP直方图
        ]
    )

    # 3. 边缘特征
    edges = filters.sobel(img)
    features.extend(
        [
            np.mean(edges),
            np.std(edges),
            np.sum(edges > 0.1) / edges.size,  # 边缘密度
        ]
    )

    # 4. 形态学特征
    binary = img < 0.5
    if np.any(binary):
        labeled = morphology.label(binary)
        features.extend(
            [
                np.max(labeled),  # 连通组件数
                np.sum(binary) / binary.size,  # 缺陷面积比
            ]
        )
    else:
        features.extend([0, 0])

    # 5. 多尺度特征
    for sigma in [1, 2, 4]:
        blurred = filters.gaussian(img, sigma=sigma)
        features.extend([np.mean(blurred), np.std(blurred)])

    return np.array(features)


def generate_dataset(n_samples=1000):
    """生成训练数据集"""
    X = []
    y = []

    defect_types = ['crack', 'hole', 'scratch', 'stain']

    for i in range(n_samples):
        # 随机选择缺陷类型和严重度
        defect_type = np.random.choice(defect_types)
        severity = np.random.uniform(0.1, 1.0)

        # 生成图像
        img = create_defect_image(defect_type=defect_type, severity=severity)

        # 提取特征
        features = extract_features(img)
        X.append(features)
        y.append(severity)

    return np.array(X), np.array(y)


def train_regression_model():
    """训练回归模型"""
    print('生成数据集...')
    X, y = generate_dataset(n_samples=1000)

    print('分割训练测试集...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('训练随机森林回归模型...')
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)

    rf.fit(X_train, y_train)

    # 预测
    y_pred = rf.predict(X_test)

    # 评估
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'模型性能:')
    print(f'MSE: {mse:.4f}')
    print(f'R²: {r2:.4f}')

    return rf, X_test, y_test, y_pred


def visualize_results(rf, X_test, y_test, y_pred):
    """可视化结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 预测vs真实值
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', lw=2)
    axes[0, 0].set_xlabel('True Severity')
    axes[0, 0].set_ylabel('Predicted Severity')
    axes[0, 0].set_title('Predicted vs True')
    axes[0, 0].grid(True)

    # 2. 残差图
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True)

    # 3. 特征重要性
    feature_names = [f'Feature_{i}' for i in range(len(rf.feature_importances_))]
    importance = rf.feature_importances_
    sorted_idx = np.argsort(importance)[::-1][:10]  # 前10个重要特征

    axes[1, 0].barh(range(len(sorted_idx)), importance[sorted_idx])
    axes[1, 0].set_yticks(range(len(sorted_idx)))
    axes[1, 0].set_yticklabels([feature_names[i] for i in sorted_idx])
    axes[1, 0].set_xlabel('Importance')
    axes[1, 0].set_title('Feature Importance (Top 10)')

    # 4. 预测分布
    axes[1, 1].hist(y_test, alpha=0.7, label='True', bins=20)
    axes[1, 1].hist(y_pred, alpha=0.7, label='Predicted', bins=20)
    axes[1, 1].set_xlabel('Severity')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Severity Distribution')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def test_on_new_images(rf):
    """在新图像上测试模型"""
    print('\n测试新图像...')

    defect_types = ['crack', 'hole', 'scratch', 'stain']
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, defect_type in enumerate(defect_types):
        # 生成测试图像
        true_severity = np.random.uniform(0.2, 0.8)
        img = create_defect_image(defect_type=defect_type, severity=true_severity)

        # 预测
        features = extract_features(img)
        pred_severity = rf.predict([features])[0]

        # 显示结果
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'{defect_type}\nTrue: {true_severity:.3f}')
        axes[0, i].axis('off')

        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f'Pred: {pred_severity:.3f}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 训练模型
    rf, X_test, y_test, y_pred = train_regression_model()

    # 可视化结果
    visualize_results(rf, X_test, y_test, y_pred)

    # 测试新图像
    test_on_new_images(rf)
