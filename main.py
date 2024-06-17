import pandas as pd
import numpy as np

# File paths of NSL-KDD dataset
name_data_path = 'names.csv'
attack_data_path = 'attack.csv'
train_data_path = 'train.csv'
test_data_path = 'test.csv'

# Load data
column_names = pd.read_csv(name_data_path, names=['names'])
train_data = pd.read_csv(train_data_path, names=column_names['names'])
test_data = pd.read_csv(test_data_path, names=column_names['names'])

# Encode
categorical_columns = ['protocol_type', 'service', 'flag']  # Liệt kê các cột phân loại
train_data = pd.get_dummies(train_data, columns=categorical_columns)
test_data = pd.get_dummies(test_data, columns=categorical_columns)

# Integrity check
train_data, test_data = train_data.align(test_data, join='inner', axis=1)

# Spliting data
from sklearn.model_selection import train_test_split

label_column = 'class'

# Split data into features (X) and labels (y)
X_train = train_data.drop(label_column, axis=1)
y_train = train_data[label_column]
X_test = test_data.drop(label_column, axis=1)
y_test = test_data[label_column]


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from imblearn.over_sampling import SMOTE
# Áp dụng SMOTE để cân bằng dữ liệu
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Định nghĩa lưới tham số cho Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30],
    'criterion': ['gini', 'entropy']
}

# Khởi tạo đối tượng GridSearchCV
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1)

# Huấn luyện với dữ liệu SMOTE
grid_search_rf.fit(X_train_sm, y_train_sm)

# Lấy siêu tham số tốt nhất
best_params_rf = grid_search_rf.best_params_
print(f"Best parameters for Random Forest: {best_params_rf}")

# Huấn luyện mô hình với siêu tham số tốt nhất
rf_best = RandomForestClassifier(**best_params_rf)
rf_best.fit(X_train_sm, y_train_sm)

# Dự đoán và đánh giá
y_pred_rf_best = rf_best.predict(X_test)
print("Random Forest Best Classifier:\n", classification_report(y_test, y_pred_rf_best))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf_best))



# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# # Decision Tree
# from sklearn.tree import DecisionTreeClassifier
# dt = DecisionTreeClassifier()
# dt.fit(X_train, y_train)
# y_pred_dt = dt.predict(X_test)
# # print("Decision Tree Classifier:\n", classification_report(y_test, y_pred_dt))
# # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))


# # SVM
# from sklearn.svm import SVC
# svm = SVC()
# svm.fit(X_train, y_train)
# y_pred_svm = svm.predict(X_test)
# # print("Support Vector Machine:\n", classification_report(y_test, y_pred_svm))
# # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))


# #Random Forest
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# y_pred_rf = rf.predict(X_test)
# # print("Random Forest Classifier:\n", classification_report(y_test, y_pred_rf))
# # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))


# # Comparison
# results = {
#     "Model": ["Decision Tree", "SVM", "Random Forest"],
#     "Accuracy": [accuracy_score(y_test, y_pred_dt), accuracy_score(y_test, y_pred_svm), accuracy_score(y_test, y_pred_rf)],
#     "Precision": [precision_score(y_test, y_pred_dt, average='weighted'), precision_score(y_test, y_pred_svm, average='weighted'), precision_score(y_test, y_pred_rf, average='weighted')],
#     "Recall": [recall_score(y_test, y_pred_dt, average='weighted'), recall_score(y_test, y_pred_svm, average='weighted'), recall_score(y_test, y_pred_rf, average='weighted')],
#     "F1 Score": [f1_score(y_test, y_pred_dt, average='weighted'), f1_score(y_test, y_pred_svm, average='weighted'), f1_score(y_test, y_pred_rf, average='weighted')]
# }

# results_df = pd.DataFrame(results)
# print(results_df)
