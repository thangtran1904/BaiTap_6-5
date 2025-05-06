import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Đọc dữ liệu
df = pd.read_csv('HousingData.csv')

# 1. Vẽ pairplot
print("Generating pairplot...")
plt.figure(figsize=(15, 10))
sns.pairplot(df, diag_kind='kde')
plt.savefig('pairplot.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Tính correlation với price
print("\nCalculating correlations...")
correlations = df.corr()['MEDV'].sort_values(ascending=False)
print("\nCorrelation with price:")
print(correlations)

# Lưu correlation vào file
correlations.to_csv('price_correlations.csv')

# 3. Xử lý outliers bằng Isolation Forest
print("\nDetecting outliers...")
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers = iso_forest.fit_predict(df.select_dtypes(include=[np.number]))
df['is_outlier'] = outliers
df_clean = df[df['is_outlier'] == 1].copy()

# 4. Tính VIF và xử lý đa cộng tuyến
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

X = df_clean.select_dtypes(include=[np.number]).drop(['MEDV', 'is_outlier'], axis=1)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())

vif_results = calculate_vif(X)
print("\nVIF Results:")
print(vif_results)

high_vif_vars = vif_results[vif_results['VIF'] > 5]['Variable'].tolist()
print("\nFeatures with high VIF (>5):", high_vif_vars)

# 5. Tạo biến mới
print("\nCreating new features...")
df_clean['room_per_crime'] = df_clean['RM'] / (df_clean['CRIM'] + 1)
tax_mean = df_clean['TAX'].mean()
df_clean['high_tax'] = (df_clean['TAX'] > tax_mean).astype(int)
df_clean['RM_LSTAT'] = df_clean['RM'] * df_clean['LSTAT']
df_clean['NOX_RM'] = df_clean['NOX'] * df_clean['RM']
df_clean['TAX_RM'] = df_clean['TAX'] * df_clean['RM']

# 6. Tạo đặc trưng đa thức (Polynomial Features) với xử lý NaN
print("\nGenerating polynomial features...")

numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

# Xử lý NaN trước khi tạo đặc trưng đa thức
imputer = SimpleImputer(strategy='mean')
numeric_data_imputed = imputer.fit_transform(df_clean[numeric_cols])

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(numeric_data_imputed)
poly_feature_names = poly.get_feature_names_out(numeric_cols)

df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)

# Gộp lại
df_final = pd.concat([df_clean.reset_index(drop=True), df_poly], axis=1)

# Lưu dữ liệu
print("\nSaving processed data...")
df_final.to_csv('housing_processed.csv', index=False)

# In thống kê mô tả
print("\nDescriptive statistics after processing:")
print(df_final.describe())

# Vẽ heatmap cho một số đặc trưng quan trọng
plt.figure(figsize=(15, 10))
important_features = ['RM', 'LSTAT', 'PTRATIO', 'NOX', 'TAX', 'CRIM', 'MEDV']
sns.heatmap(df_final[important_features].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Important Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nData processing completed! Check the output files for detailed results.")


