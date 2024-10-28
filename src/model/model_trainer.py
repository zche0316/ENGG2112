import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
        plt.title(f'{name} 混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.show()
        
        # 对随机森林显示特征重要性
        if name == 'Random Forest':
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
            plt.title('Random Forest 特征重要性 (Top 10)')
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
