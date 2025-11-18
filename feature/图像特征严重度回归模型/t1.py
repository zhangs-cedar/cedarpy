# ------------------------------------------------------------
# 1. 必备库
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib   # 比 pickle 更快，用法一样

# ------------------------------------------------------------
# 2. 造一份非线性回归数据
#    y = sin(4πx) · x + ε,  x∈[0,1], ε~N(0,0.05²)
# ------------------------------------------------------------
np.random.seed(42)
X = np.linspace(0, 1, 500).reshape(-1, 1)          # 500×1
y = np.sin(4 * np.pi * X).ravel() * X.ravel() + np.random.normal(0, 0.05, size=500)

# ------------------------------------------------------------
# 3. 训练集 / 测试集 拆分
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ------------------------------------------------------------
# 4. 建模
# ------------------------------------------------------------
rf_reg = RandomForestRegressor(
    n_estimators=500,   # 树的数量
    max_depth=None,     # 不限制深度
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1           # 用满 CPU
)
rf_reg.fit(X_train, y_train)

# ------------------------------------------------------------
# 5. 预测与评估
# ------------------------------------------------------------
y_pred = rf_reg.predict(X_test)
mse  = mean_squared_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
print(f"MSE : {mse:.4f}")
print(f"R²  : {r2:.4f}")

# ------------------------------------------------------------
# 6. 可视化：真实 vs 预测
# ------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.scatter(X_test, y_test, color='tab:blue', label='True')
plt.scatter(X_test, y_pred, color='tab:red', alpha=0.6, label='RF predict')
plt.title("RandomForestRegressor: True vs Predicted")
plt.legend(); plt.show()

# ------------------------------------------------------------
# 7. 特征重要性（本例仅 1 个特征，仅作演示）
# ------------------------------------------------------------
importances = rf_reg.feature_importances_
print("Feature importances:", importances)

# ------------------------------------------------------------
# 8. 模型持久化
# ------------------------------------------------------------
joblib.dump(rf_reg, "rf_regressor.pkl")
# 下次使用：
# loaded = joblib.load("rf_regressor.pkl")
# new_pred = loaded.predict(new_X)