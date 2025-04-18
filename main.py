# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
import statsmodels.formula.api as smf
import os


def create_output_directories():
    """Create directory structure for storing results"""
    # Define directories to create
    directories = [
        'results',
        'results/stepwise',
        'results/linear',
        'results/lasso',
        'results/common'
    ]
    
    # Create directories
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def load_and_explore_data(file_path='data.csv'):
    """Load and explore the MLB data"""
    # Load data
    mlb_data = pd.read_csv(file_path)

    # View basic information about the dataset
    print("Dataset shape:", mlb_data.shape)
    print("\nFirst 5 rows:")
    print(mlb_data.head())

    # Check basic statistics
    print("\nStatistical summary:")
    print(mlb_data.describe())

    # Check for missing values
    print("\nMissing values by column:")
    print(mlb_data.isnull().sum())

    # Handle missing values (if any)
    mlb_data = mlb_data.dropna()

    return mlb_data


def visualize_correlations(mlb_data):
    """Create and save correlation heatmap"""
    correlation_matrix = mlb_data.corr()
    plt.figure(figsize=(8, 7))  # 缩小图表尺寸
    # 移除错误的fontsize参数
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.savefig('results/common/correlation_heatmap.png', dpi=200, bbox_inches='tight')  # 降低DPI
    plt.close()


def prepare_data(mlb_data):
    """Prepare features and target and split data"""
    # Prepare features and target variable
    X = mlb_data.drop('Salary', axis=1)  # All features
    y = mlb_data['Salary']  # Target variable: salary

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X, y, X_train, X_test, y_train, y_test


def stepwise_regression(X, y, X_train, X_test, y_train, y_test):
    """Perform stepwise regression analysis"""
    # Create a DataFrame that includes both X_train and y_train for statsmodels
    train_data = pd.DataFrame(X_train)
    train_data['Salary'] = y_train
    
    # Get list of feature names
    feature_names = X_train.columns.tolist()
    
    # Start with an empty model
    current_features = []
    remaining_features = feature_names.copy()
    best_model = None
    best_aic = float('inf')
    
    print("\nPerforming Forward Stepwise Regression:")
    
    # Forward selection process
    while remaining_features:
        best_new_feature = None
        best_new_aic = float('inf')
        
        # Try adding each remaining feature
        for feature in remaining_features:
            # Create formula for the current model
            formula = "Salary ~ " + " + ".join(current_features + [feature])
            
            # Fit the model
            model = smf.ols(formula=formula, data=train_data).fit()
            
            # Evaluate the model
            aic = model.aic
            
            # If this feature improves the model, remember it
            if aic < best_new_aic:
                best_new_aic = aic
                best_new_feature = feature
        
        # If adding the best feature improves the overall model, add it
        if best_new_aic < best_aic:
            best_aic = best_new_aic
            current_features.append(best_new_feature)
            remaining_features.remove(best_new_feature)
            best_model = smf.ols(formula="Salary ~ " + " + ".join(current_features), 
                               data=train_data).fit()
            print(f"Added feature: {best_new_feature}, AIC: {best_aic:.2f}")
        else:
            # No improvement, so we're done
            break
    
    print(f"\nFinal model includes {len(current_features)} features:")
    print(", ".join(current_features))
    
    # Get summary of the best model
    print("\nDetailed model summary:")
    print(best_model.summary())
    
    # Make predictions on test set
    # Need to add intercept to X_test for prediction
    X_test_for_pred = X_test[current_features].copy()
    y_pred = best_model.predict(pd.DataFrame({feat: X_test_for_pred[feat] for feat in current_features}))
    
    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\nStepwise Regression Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Coefficient of Determination (R²): {r2:.4f}")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'Feature': current_features,
        'Coefficient': best_model.params[1:],  # Skip the intercept
        'P-value': [best_model.pvalues[i+1] for i in range(len(current_features))]  # Skip the intercept
    })
    feature_importance['Absolute Coefficient'] = abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Absolute Coefficient', ascending=False)
    print("\nFeature Importance (sorted by absolute coefficient value):")
    print(feature_importance)
    
    # Visualize feature importance
    plt.figure(figsize=(7, 6))  # 缩小图表尺寸
    sns.barplot(x='Absolute Coefficient', y='Feature', data=feature_importance)
    plt.title('Features Selected by Stepwise Regression', fontsize=10)
    plt.xlabel('Absolute Coefficient', fontsize=9)
    plt.ylabel('Feature', fontsize=9)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig('results/stepwise/feature_importance.png', dpi=200, bbox_inches='tight')  # 降低DPI
    plt.close()
    
    visualize_results(y_test, y_pred, 'stepwise')
    
    return best_model, current_features


def regular_linear_regression(X, y, X_train, X_test, y_train, y_test):
    """Perform regular linear regression analysis"""
    # Train linear regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Make predictions
    y_pred = lr_model.predict(X_test)

    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nRegular Linear Regression Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Coefficient of Determination (R²): {r2:.4f}")

    # Print feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lr_model.coef_
    })
    feature_importance['Absolute Coefficient'] = abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Absolute Coefficient', ascending=False)
    print("\nFeature Importance (sorted by absolute coefficient value):")
    print(feature_importance)

    # Visualize feature importance
    plt.figure(figsize=(7, 6))  # 缩小图表尺寸
    sns.barplot(x='Absolute Coefficient', y='Feature', data=feature_importance.head(10))
    plt.title('Top 10 Important Features (Regular Linear Regression)', fontsize=10)
    plt.xlabel('Absolute Coefficient', fontsize=9)
    plt.ylabel('Feature', fontsize=9)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig('results/linear/feature_importance.png', dpi=200, bbox_inches='tight')  # 降低DPI
    plt.close()

    # Use statsmodels for more detailed statistical analysis
    X_train_sm = sm.add_constant(X_train)  # Add constant term
    sm_model = sm.OLS(y_train, X_train_sm).fit()
    print("\nDetailed Statistical Analysis Results:")
    print(sm_model.summary())

    visualize_results(y_test, y_pred, 'linear')
    
    return lr_model


def lasso_regression(X, y, X_train, X_test, y_train, y_test):
    """Perform LASSO regression analysis with hyperparameter tuning"""
    # Define the parameter grid for alpha
    param_grid = {'alpha': np.logspace(-4, 1, 30)}
    
    # Create LASSO model
    lasso = Lasso(max_iter=10000, random_state=42)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Get the best alpha
    best_alpha = grid_search.best_params_['alpha']
    print(f"\nBest LASSO alpha: {best_alpha}")
    
    # Train LASSO model with best alpha
    lasso_model = Lasso(alpha=best_alpha, max_iter=10000, random_state=42)
    lasso_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lasso_model.predict(X_test)
    
    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\nLASSO Regression Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Coefficient of Determination (R²): {r2:.4f}")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lasso_model.coef_
    })
    feature_importance['Absolute Coefficient'] = abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Absolute Coefficient', ascending=False)
    
    # Count non-zero coefficients
    non_zero_coefs = np.sum(lasso_model.coef_ != 0)
    print(f"\nNumber of features selected by LASSO: {non_zero_coefs} out of {X.shape[1]}")
    
    # Show only non-zero coefficients
    non_zero_features = feature_importance[feature_importance['Coefficient'] != 0]
    print("\nFeatures selected by LASSO (non-zero coefficients):")
    print(non_zero_features)
    
    # Visualize feature importance
    plt.figure(figsize=(7, 6))  # 缩小图表尺寸
    sns.barplot(x='Absolute Coefficient', y='Feature', data=non_zero_features)
    plt.title('Important Features Selected by LASSO', fontsize=10)
    plt.xlabel('Absolute Coefficient', fontsize=9)
    plt.ylabel('Feature', fontsize=9)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig('results/lasso/feature_importance.png', dpi=200, bbox_inches='tight')  # 降低DPI
    plt.close()
    
    visualize_results(y_test, y_pred, 'lasso')
    
    return lasso_model


def visualize_results(y_test, y_pred, model_type):
    """Visualize model results"""
    # Set output directory
    output_dir = f'results/{model_type}'
    
    # 创建包含左右两个子图的图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))  # 一行两列的子图布局
    
    # 左侧子图: Actual vs Predicted scatter plot
    ax1.scatter(y_test, y_pred, alpha=0.7, s=30)  # 减小点的大小
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax1.set_xlabel('Actual Salary (thousand USD)', fontsize=9)
    ax1.set_ylabel('Predicted Salary (thousand USD)', fontsize=9)
    ax1.set_title('Actual vs Predicted Salary', fontsize=10)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    
    # 右侧子图: Residual distribution
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, bins=15, ax=ax2)  # 减少bin数量
    ax2.set_xlabel('Residuals', fontsize=9)
    ax2.set_ylabel('Frequency', fontsize=9)
    ax2.set_title('Residual Distribution', fontsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    
    # 设置整体标题
    fig.suptitle(f'{model_type.capitalize()} Regression Results', fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)  # 为整体标题留出空间
    
    # 保存组合图
    plt.savefig(f'{output_dir}/combined_results.png', dpi=200, bbox_inches='tight')  # 降低DPI
    plt.close()

    # 单独的残差分析图仍然保留
    plt.figure(figsize=(6, 5))  # 缩小图表尺寸
    plt.scatter(y_pred, residuals, alpha=0.7, s=30)  # 减小点的大小
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Salary (thousand USD)', fontsize=9)
    plt.ylabel('Residuals', fontsize=9)
    plt.title(f'Residual Analysis ({model_type.capitalize()})', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/residuals.png', dpi=200, bbox_inches='tight')  # 降低DPI
    plt.close()


def main():
    """Main function to control the analysis flow"""
    print("MLB Player Salary Prediction Analysis")
    print("=====================================")
    
    # Create output directories
    create_output_directories()
    
    # Load and explore data
    mlb_data = load_and_explore_data()
    
    # Visualize correlations
    visualize_correlations(mlb_data)
    
    # Prepare data
    X, y, X_train, X_test, y_train, y_test = prepare_data(mlb_data)
    
    # Choose regression method
    print("\nChoose regression method:")
    print("1: Stepwise Regression")
    print("2: Regular Linear Regression")
    print("3: LASSO Regression")
    print("4: All methods")
    
    choice = input("Enter your choice (1/2/3/4): ")
    
    if choice == '1' or choice == '4':
        print("\n--- Running Stepwise Regression ---")
        stepwise_model, selected_features = stepwise_regression(X, y, X_train, X_test, y_train, y_test)
    
    if choice == '2' or choice == '4':
        print("\n--- Running Regular Linear Regression ---")
        linear_model = regular_linear_regression(X, y, X_train, X_test, y_train, y_test)
    
    if choice == '3' or choice == '4':
        print("\n--- Running LASSO Regression ---")
        lasso_model = lasso_regression(X, y, X_train, X_test, y_train, y_test)
    
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()