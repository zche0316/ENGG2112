from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import numpy as np


def train_and_evaluate_models(X_train, X_val, y_train, y_val):
    """
    训练并评估逻辑回归和随机森林模型
    """
    # 1. 逻辑回归
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_val)
    lr_prob = lr_model.predict_proba(X_val)[:, 1]
    
    # 2. 随机森林
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_val)
    rf_prob = rf_model.predict_proba(X_val)[:, 1]
    
    # 评估结果
    models = {
        'Logistic Regression': (lr_pred, lr_prob, lr_model),
        'Random Forest': (rf_pred, rf_prob, rf_model)
    }
    
    # 绘制ROC曲线比较
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red']
    for (name, (pred, prob, model)), color in zip(models.items(), colors):
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_val, prob)
        roc_auc = auc(fpr, tpr)
        
        # 绘制ROC曲线
        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    # 添加随机猜测的基准线
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    # 打印每个模型的评估结果
    for name, (pred, prob, model) in models.items():
        print(f"\n{name} 评估结果:")
        print("\n分类报告:")
        print(classification_report(y_val, pred))
        print("\nROC-AUC得分:", roc_auc_score(y_val, prob))
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_val, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # 显示特征重要性/权重
        plt.figure(figsize=(10, 6))
        if name == 'Random Forest':
            # 随机森林特征重要性
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
            plt.title('Random Forest Feature Importance (Top 10)')
            plt.xlabel('Importance Score')
            
        else:
            # 逻辑回归系数
            feature_weights = pd.DataFrame({
                'feature': X_train.columns,
                'coefficient': np.abs(model.coef_[0])  # 使用绝对值便于比较
            }).sort_values('coefficient', ascending=False)
            
            sns.barplot(data=feature_weights.head(10), x='coefficient', y='feature')
            plt.title('Logistic Regression Absolute Coefficients (Top 10)')
            plt.xlabel('|Coefficient|')
        
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    
    return models

# 确保函数可以被导出
__all__ = ['train_and_evaluate_models']

if __name__ == "__main__":
    # 导入数据处理模块
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_processor import prepare_humid_fire_data, split_dataset, preprocess_features

    # 1. 数据准备
    print("正在准备数据...")
    merged_data, stations = prepare_humid_fire_data()
    
    # 2. 数据集划分
    print("\n正在划分数据集...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(merged_data, stations)
    
    # 3. 特征预处理
    print("\n正在预处理特征...")
    X_train_processed, X_val_processed, X_test_processed = preprocess_features(X_train, X_val, X_test)
    
    # 4. 模型训练和评估
    print("\n开始训练和评估模型...")
    models = train_and_evaluate_models(X_train_processed, X_val_processed, y_train, y_val)
