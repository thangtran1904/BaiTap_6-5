import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Đọc dữ liệu
df = pd.read_csv('Data_Number_7.csv')

# 1. Tạo chỉ số nguy cơ biến chứng
def calculate_complication_risk(row):
    # Chuẩn hóa các chỉ số
    bmi_risk = (row['bmi'] - 18.5) / (40 - 18.5)  # BMI bình thường: 18.5-25
    glucose_risk = (row['blood_glucose'] - 70) / (200 - 70)  # Đường huyết bình thường: 70-140
    admission_risk = row['hospitalizations'] / 10  # Chuẩn hóa số lần nhập viện
    
    # Tính toán nguy cơ tổng hợp
    risk_score = (0.4 * bmi_risk + 0.4 * glucose_risk + 0.2 * admission_risk)
    return min(max(risk_score, 0), 1)  # Chuẩn hóa về [0,1]

# 2. Phân nhóm tuổi
def age_group(age):
    if age < 40:
        return '<40'
    elif age <= 60:
        return '40-60'
    else:
        return '>60'

# 3. Tạo đặc trưng xu hướng đường huyết (giả định)
def generate_glucose_trend(row):
    # Giả định xu hướng dựa trên mức đường huyết hiện tại
    if row['blood_glucose'] > 180:
        return 'tăng'
    elif row['blood_glucose'] < 70:
        return 'giảm'
    else:
        return 'ổn định'

# 4. Tạo đặc trưng mức độ nghiêm trọng
def calculate_severity(row):
    # Chuẩn hóa các chỉ số
    glucose_severity = (row['blood_glucose'] - 70) / (200 - 70)
    admission_severity = row['hospitalizations'] / 10
    
    # Tính toán mức độ nghiêm trọng
    severity = (0.6 * glucose_severity + 0.4 * admission_severity)
    return min(max(severity, 0), 1)

# Áp dụng các tính toán
df['complication_risk'] = df.apply(calculate_complication_risk, axis=1)
df['age_group'] = df['age'].apply(age_group)
df['glucose_trend'] = df.apply(generate_glucose_trend, axis=1)
df['severity'] = df.apply(calculate_severity, axis=1)

# 5. Kiểm định chi-squared
def perform_chi_square_test():
    contingency_table = pd.crosstab(df['age_group'], df['complication'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    print("\nKết quả kiểm định Chi-squared:")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"p-value: {p_value:.4f}")
    print("Kết luận: Biến chứng", 
          "phụ thuộc vào nhóm tuổi" if p_value < 0.05 else "không phụ thuộc vào nhóm tuổi")

# 6. Xây dựng và đánh giá mô hình
def build_and_evaluate_models():
    # Chuẩn bị dữ liệu
    X = df[['age', 'bmi', 'blood_glucose', 'hospitalizations', 'complication_risk', 'severity']]
    y = df['complication']
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Áp dụng SMOTE để xử lý mất cân bằng
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Mô hình Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_balanced, y_train_balanced)
    lr_pred = lr.predict(X_test)
    
    print("\nKết quả Logistic Regression:")
    print(classification_report(y_test, lr_pred))
    
    # Mô hình Random Forest với tối ưu siêu tham số
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_balanced, y_train_balanced)
    
    print("\nKết quả tối ưu Random Forest:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    rf_pred = grid_search.predict(X_test)
    print("\nKết quả Random Forest trên tập test:")
    print(classification_report(y_test, rf_pred))

# 7. Vẽ biểu đồ phân tích
def plot_analysis():
    plt.figure(figsize=(15, 10))
    
    # Biểu đồ 1: Phân bố nguy cơ biến chứng theo nhóm tuổi
    plt.subplot(2, 2, 1)
    sns.boxplot(x='age_group', y='complication_risk', data=df)
    plt.title('Phân bố nguy cơ biến chứng theo nhóm tuổi')
    
    # Biểu đồ 2: Tương quan giữa BMI và đường huyết
    plt.subplot(2, 2, 2)
    sns.scatterplot(x='bmi', y='blood_glucose', hue='complication', data=df)
    plt.title('Tương quan giữa BMI và đường huyết')
    
    # Biểu đồ 3: Phân bố mức độ nghiêm trọng
    plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='severity', bins=30)
    plt.title('Phân bố mức độ nghiêm trọng')
    
    # Biểu đồ 4: Tỷ lệ biến chứng theo nhóm tuổi
    plt.subplot(2, 2, 4)
    sns.barplot(x='age_group', y='complication', data=df)
    plt.title('Tỷ lệ biến chứng theo nhóm tuổi')
    
    plt.tight_layout()
    plt.savefig('health_analysis.png')
    plt.close()

if __name__ == "__main__":
    # Thực hiện các phân tích
    perform_chi_square_test()
    build_and_evaluate_models()
    plot_analysis()
    
    # In thống kê mô tả
    print("\nThống kê mô tả:")
    print(df[['complication_risk', 'severity']].describe()) 