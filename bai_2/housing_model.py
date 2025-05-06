import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest, f_regression
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
print("Loading processed data...")

# Load data
df = pd.read_csv('housing_processed_clean.csv')

# 1. Kiểm tra dữ liệu
print("\nData shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nCorrelation with MEDV:\n", df.corr()['MEDV'].sort_values(ascending=False))

# Kiểm tra trùng lặp
print("\nDuplicate rows:", df.duplicated().sum())
if df.duplicated().sum() > 0:
    df = df.drop_duplicates()
    print("Removed duplicates, new shape:", df.shape)

# 2. Loại bỏ đặc trưng có tương quan cao với MEDV (>0.95) để tránh rò rỉ dữ liệu
corr_with_medv = df.corr()['MEDV'].abs()
high_corr_features = corr_with_medv[corr_with_medv > 0.95].index.tolist()
if 'MEDV' in high_corr_features:
    high_corr_features.remove('MEDV')
if high_corr_features:
    print("\nDropping high-correlation features:", high_corr_features)
    df = df.drop(high_corr_features, axis=1)

# 3. Chia dữ liệu
X = df.drop(['MEDV', 'is_outlier'], axis=1, errors='ignore')
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# 4. Chọn đặc trưng quan trọng
selector = SelectKBest(score_func=f_regression, k=10)  # Chọn 10 đặc trưng tốt nhất
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
selected_features = X_train.columns[selector.get_support()].tolist()
print("\nSelected features:", selected_features)

# Cập nhật X_train và X_test
X_train = pd.DataFrame(X_train_selected, columns=selected_features)
X_test = pd.DataFrame(X_test_selected, columns=selected_features)


# 5. Pipeline builder
def create_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])


# 6. Optuna objective function với giới hạn độ phức tạp
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),  # Giảm số cây
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),  # Giảm learning rate
        'max_depth': trial.suggest_int('max_depth', 3, 6),  # Giới hạn độ sâu
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),  # Tăng ngưỡng chia
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),  # Tăng số mẫu tối thiểu
        'subsample': trial.suggest_float('subsample', 0.7, 1.0)  # Sử dụng mẫu phụ
    }
    model = GradientBoostingRegressor(**params, random_state=42)
    pipeline = create_pipeline(model)

    try:
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        return -scores.mean()  # Minimize MSE
    except Exception as e:
        print(f"Error in trial: {e}")
        return float('inf')


# 7. Optimize Gradient Boosting parameters
print("\nOptimizing hyperparameters with Optuna...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20, show_progress_bar=True)
best_params = study.best_params
print("\nBest hyperparameters found:", best_params)

# 8. Define models với điều chuẩn
lr = Ridge(alpha=1.0)  # Sử dụng Ridge thay vì Linear Regression
gb = GradientBoostingRegressor(**best_params, random_state=42)
nn = MLPRegressor(
    hidden_layer_sizes=(50, 25),  # Giảm kích thước mạng
    max_iter=1000,
    learning_rate_init=0.001,
    early_stopping=True,
    alpha=0.01,  # Thêm L2 regularization
    random_state=42
)

# 9. Stacking model
estimators = [
    ('lr', create_pipeline(lr)),
    ('gb', create_pipeline(gb)),
    ('nn', create_pipeline(nn))
]
stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),  # Sử dụng Ridge làm meta-learner
    cv=5,
    n_jobs=-1
)


# 10. Evaluation function với cross-validation
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return {
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred),
        'CV_MSE': -cv_scores.mean(),
        'CV_MSE_STD': cv_scores.std()
    }


# 11. Evaluate all models
print("\nEvaluating models...")
models = {
    'Ridge Regression': create_pipeline(lr),
    'Gradient Boosting': create_pipeline(gb),
    'Neural Network': create_pipeline(nn),
    'Stacking': stacking
}

results = {}
for name, model in models.items():
    print(f"\n{name}")
    results[name] = evaluate_model(model, X_train, y_train, X_test, y_test)
    for metric, val in results[name].items():
        print(f"{metric}: {val:.4f}")

# 12. Residual plot for best model (Gradient Boosting)
print("\nCreating residual plot...")
best_model = models['Gradient Boosting']
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted MEDV')
plt.ylabel('Residuals')
plt.title('Residual Plot - Gradient Boosting')
plt.grid(True)
plt.tight_layout()
plt.savefig('residual_plot.png', dpi=300)
plt.close()

# 13. SHAP analysis
print("\nPerforming SHAP analysis...")
gb_model = best_model.named_steps['model']
X_test_scaled = best_model.named_steps['scaler'].transform(X_test)
X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

explainer = shap.Explainer(gb_model)
shap_values = explainer(X_test_df)

plt.figure()
shap.summary_plot(shap_values, X_test_df, plot_type='bar', show=False)
plt.title('SHAP Feature Importance')
plt.tight_layout()
plt.savefig('shap_importance.png', dpi=300)
plt.close()

# 14. Save model results
results_df = pd.DataFrame(results).T
results_df.to_csv('model_evaluation_results.csv')

# 15. Feature importance from GB
feature_importance = pd.DataFrame({
    'feature': X_test.columns,
    'importance': gb_model.feature_importances_
}).sort_values(by='importance', ascending=False)
feature_importance.to_csv('feature_importance.csv', index=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))

# 16. Plot model comparison
plt.figure(figsize=(10, 6))
results_df['RMSE'].plot(kind='bar', color='steelblue')
plt.ylabel('RMSE')
plt.title('Model Performance Comparison (Test Set)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.close()

print("\nModel training and evaluation completed successfully!")