import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu
df = pd.read_csv('Data_Number_5.csv')

# 1. Tạo chỉ số hiệu suất học tập tổng hợp
def calculate_performance_score(row):
    # Trọng số cho các môn học
    weights = {'math_score': 0.3, 'literature_score': 0.3, 'science_score': 0.4}
    # Tính điểm trung bình có trọng số
    weighted_score = (row['math_score'] * weights['math_score'] + 
                     row['literature_score'] * weights['literature_score'] + 
                     row['science_score'] * weights['science_score'])
    # Thêm ảnh hưởng của số giờ tự học (tối đa 20% bonus)
    study_hours_bonus = min(row['study_hours'] / 40, 1) * 20
    return weighted_score + study_hours_bonus

# 2. Tạo đặc trưng cân bằng học tập
def calculate_learning_balance(row):
    scores = [row['math_score'], row['literature_score'], row['science_score']]
    return np.std(scores)  # Độ lệch chuẩn càng thấp thì càng cân bằng

# 3. Tạo đặc trưng rủi ro học tập
def calculate_learning_risk(row):
    # Rủi ro tăng khi số buổi vắng mặt cao và số giờ tự học thấp
    absence_risk = row['absences'] / 20  # Chuẩn hóa số buổi vắng mặt
    study_risk = 1 - (row['study_hours'] / 40)  # Chuẩn hóa số giờ tự học
    return (absence_risk + study_risk) / 2

# Áp dụng các tính toán
df['performance_score'] = df.apply(calculate_performance_score, axis=1)
df['learning_balance'] = df.apply(calculate_learning_balance, axis=1)
df['learning_risk'] = df.apply(calculate_learning_risk, axis=1)

# 4. Kiểm định ANOVA cho mức độ tham gia ngoại khóa
def perform_anova():
    groups = [group for _, group in df.groupby('extracurricular')['performance_score']]
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"\nKết quả kiểm định ANOVA:")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print("Kết luận: Mức độ tham gia ngoại khóa", 
          "có ảnh hưởng đáng kể" if p_value < 0.05 else "không có ảnh hưởng đáng kể",
          "đến hiệu suất học tập")

# 5. Xây dựng mô hình SVM
def build_svm_model():
    # Chuẩn bị dữ liệu
    X = df[['performance_score', 'learning_balance', 'learning_risk']]
    y = (df['performance_score'] < df['performance_score'].median()).astype(int)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Tìm siêu tham số tối ưu
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 1]
    }
    
    svm = SVC()
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print("\nKết quả tối ưu siêu tham số:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Đánh giá mô hình
    y_pred = grid_search.predict(X_test)
    print("\nBáo cáo phân loại:")
    print(classification_report(y_test, y_pred))

# 6. Vẽ biểu đồ phân tích
def plot_analysis():
    plt.figure(figsize=(15, 10))
    
    # Biểu đồ 1: Phân bố điểm hiệu suất theo mức độ ngoại khóa
    plt.subplot(2, 2, 1)
    sns.boxplot(x='extracurricular', y='performance_score', data=df)
    plt.title('Phân bố điểm hiệu suất theo mức độ ngoại khóa')
    
    # Biểu đồ 2: Tương quan giữa số giờ học và điểm hiệu suất
    plt.subplot(2, 2, 2)
    sns.scatterplot(x='study_hours', y='performance_score', hue='extracurricular', data=df)
    plt.title('Tương quan giữa số giờ học và điểm hiệu suất')
    
    # Biểu đồ 3: Phân bố rủi ro học tập
    plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='learning_risk', bins=30)
    plt.title('Phân bố rủi ro học tập')
    
    # Biểu đồ 4: Tương quan giữa cân bằng học tập và điểm hiệu suất
    plt.subplot(2, 2, 4)
    sns.scatterplot(x='learning_balance', y='performance_score', hue='extracurricular', data=df)
    plt.title('Tương quan giữa cân bằng học tập và điểm hiệu suất')
    
    plt.tight_layout()
    plt.savefig('education_analysis.png')
    plt.close()

if __name__ == "__main__":
    # Thực hiện các phân tích
    perform_anova()
    build_svm_model()
    plot_analysis()
    
    # In thống kê mô tả
    print("\nThống kê mô tả:")
    print(df[['performance_score', 'learning_balance', 'learning_risk']].describe()) 