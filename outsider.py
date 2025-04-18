import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

def find_outliers_in_mlb_data(file_path='data.csv', threshold=1000):
    """
    找出MLB数据集中的离群值，特别是薪资高但预测值低的球员
    
    参数:
    file_path: MLB数据集的文件路径
    threshold: 实际值与预测值差异的阈值，用于识别显著离群值
    
    返回:
    离群值的详细信息
    """
    # 加载数据
    print("加载MLB数据...")
    mlb_data = pd.read_csv(file_path)
    
    # 检查是否有名称/ID列
    player_id_column = None
    for col in mlb_data.columns:
        if 'name' in col.lower() or 'player' in col.lower() or 'id' in col.lower():
            player_id_column = col
            break
    
    # 准备特征和目标变量
    print("准备数据...")
    if player_id_column:
        X = mlb_data.drop(['Salary', player_id_column], axis=1)
        players = mlb_data[player_id_column]
    else:
        X = mlb_data.drop('Salary', axis=1)
        players = pd.Series(range(len(mlb_data)), name='PlayerIndex')
    
    y = mlb_data['Salary']
    
    # 拟合普通线性回归模型
    print("拟合线性回归模型...")
    model = LinearRegression()
    model.fit(X, y)
    
    # 预测整个数据集
    y_pred = model.predict(X)
    
    # 计算残差
    residuals = y - y_pred
    
    # 创建包含所有数据的DataFrame
    print("分析残差...")
    results_df = pd.DataFrame({
        'PlayerID': players,
        'Actual_Salary': y,
        'Predicted_Salary': y_pred,
        'Residual': residuals,
        'Abs_Residual': np.abs(residuals)
    })
    
    # 添加原始数据的所有列
    for col in X.columns:
        results_df[col] = X[col].values
    
    # 找出显著的离群值，特别是实际值高但预测值低的
    print("识别离群值...")
    outliers = results_df[
        (results_df['Abs_Residual'] > threshold) & 
        (results_df['Actual_Salary'] > 1000)  # 薪资高于1000k
    ].sort_values('Abs_Residual', ascending=False)
    
    # 找出与描述匹配的特定离群值
    target_outlier = results_df[
        (results_df['Actual_Salary'] > 1900) &  # 接近2000k
        (results_df['Actual_Salary'] < 2100) &  # 接近2000k
        (results_df['Predicted_Salary'] < 500)   # 预测值很低
    ]
    
    # 可视化所有点和离群值
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, y, alpha=0.5)
    plt.scatter(outliers['Predicted_Salary'], outliers['Actual_Salary'], color='red', s=100)
    if not target_outlier.empty:
        plt.scatter(target_outlier['Predicted_Salary'], target_outlier['Actual_Salary'], 
                   color='green', s=150, marker='*')
    
    plt.plot([0, max(y)], [0, max(y)], 'k--')
    plt.xlabel('Predicted Salary (thousand USD)')
    plt.ylabel('Actual Salary (thousand USD)')
    plt.title('Actual vs Predicted Salary with Outliers Highlighted')
    plt.grid(True, alpha=0.3)
    
    # 添加标注
    for idx, row in outliers.head(5).iterrows():
        plt.annotate(f"Player {row['PlayerID']}", 
                    (row['Predicted_Salary'], row['Actual_Salary']),
                    xytext=(10, 10), textcoords='offset points')
    
    if not target_outlier.empty:
        for idx, row in target_outlier.iterrows():
            plt.annotate(f"Target: Player {row['PlayerID']}", 
                        (row['Predicted_Salary'], row['Actual_Salary']),
                        xytext=(10, -15), textcoords='offset points', 
                        fontweight='bold', color='green')
    
    plt.tight_layout()
    plt.savefig('outliers_analysis.png')
    plt.show()
    
    # 分析这个特定的离群值
    if not target_outlier.empty:
        print("\n找到目标离群值!")
        print("详细信息:")
        print(target_outlier)
        
        # 计算平均值以进行比较
        avg_stats = X.mean()
        
        print("\n与平均水平比较:")
        comparison = pd.DataFrame({
            '目标球员': target_outlier.iloc[0, 5:].values,
            '数据集平均值': avg_stats.values,
            '差异百分比': ((target_outlier.iloc[0, 5:].values - avg_stats.values) / avg_stats.values * 100)
        }, index=X.columns)
        print(comparison)
        
        return target_outlier
    else:
        print("\n未找到完全匹配的目标离群值。")
        print("显示最显著的离群值:")
        print(outliers.head())
        return outliers.head()

# 执行函数
if __name__ == "__main__":
    outlier_data = find_outliers_in_mlb_data()
    
    # 保存结果到CSV
    outlier_data.to_csv('identified_outliers.csv', index=False)
    print("\n结果已保存到'identified_outliers.csv'")