from helpers.eda import *
from helpers.data_prep import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, classification_report, plot_roc_curve

df = pd.read_csv("diabetes.csv")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

df.head(20)
df.shape

check_df(df)

# aykırı değer var mı?
def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df, col))

# aykırı değerleri baskıla

for col in num_cols:
    replace_with_thresholds(df, col)

# baskıladıktan sonra kontrol
for col in num_cols:
    print(col, check_outlier(df, col))

# eksik değerler

df.isnull().values.any()
df.head()

# 0 değerler için median ile değiştirdim.

def zeros_values(df):
    median = df[num_cols].median()
    df[num_cols] = df[num_cols].replace(to_replace = 0, value = median)
zeros_values(df)
df.head(20)


# feature engineering

# yaş değişkenini kategorilere ayırma
df.loc[(df["Age"] < 18), "NEW_AGE"] = "Young"
df.loc[(df["Age"] > 18) & (df["Age"] < 56), "NEW_AGE"] = "Mature"
df.loc[(df["Age"] > 56), "NEW_AGE"] = "Old"

#insülin değerlerinin normallikleri
df.loc[(df["Insulin"] >=120), "NEW_INSULIN"] = "Anormal"
df.loc[(df["Insulin"] < 120), "NEW_INSULIN"] = "normal"
df.head()

# BMI değerlerinin analizi
df.loc[(df["BMI"] < 18.5), "NEW_BMI"] = "under"
df.loc[(df["BMI"] >= 18.5) & (df["BMI"] <= 24.9) ,"NEW_BMI"] = "healthy"
df.loc[(df["BMI"] >= 25) & (df["BMI"] <= 29.9) ,"NEW_BMI"]= "over"
df.loc[(df["BMI"] >= 30), "NEW_BMI"] = "obese"

# bloodpressure analizi
df.loc[(df["BloodPressure"] < 79), "NEW_BLOODPRESSURE"] = "Normal"
df.loc[(df["BloodPressure"] > 79) & (df["BloodPressure"] < 89), "NEW_BLOODPRESSURE"] = "Hypertension_S1"
df.loc[(df["BloodPressure"] > 89) & (df["BloodPressure"] < 123), "NEW_BLOODPRESSURE"] = "Hypertension_S2"

# glikoz analizi
df.loc[(df["Glucose"] < 70), "NEW_GLUCOSE"] = "Low"
df.loc[(df["Glucose"] >= 70) & (df["Glucose"] < 99), "NEW_GLUCOSE"] = "Normal"
df.loc[(df["Glucose"] > 99) & (df["Glucose"] <= 125), "NEW_GLUCOSE"] = "Secret"
df.loc[(df["Glucose"] >= 126) & (df["Glucose"] < 200), "NEW_GLUCOSE"] = "High"

# hastalık genetik mi değil mi olasılığı
df["DiaPedFunc"] = pd.qcut(df["DiabetesPedigreeFunction"], 3, labels=["Low", "Medium", "High"])

# hamilelik durumu arttıkça diyabet ihtimali arttığı için
df.loc[df['Pregnancies'] == 0, "NEW_PREGNANCIES"] = "NoPregnancy"
df.loc[((df['Pregnancies'] > 0) & (df['Pregnancies'] <= 4)), "NEW_PREGNANCIES"] = "NormalPregnancy"
df.loc[(df['Pregnancies'] > 4), "NEW_PREGNANCIES"] = "OverPregnancy"

# df["BMI_INS"] = df["BMI"] * df["Insulin"]
# df["AGE_PRE"] = df["Age"]* df["Pregnancies"]
df.head()
df.shape

# encode
# label encoder
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col].astype(str))
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O"
               and len(df[col].unique()) == 2]

for col in df.columns:
    label_encoder(df, col)

# one hot encoder

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]

one_hot_encoder(df, ohe_cols, drop_first=True)

# scale

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

# model

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=1)

log_model = LogisticRegression().fit(X_train, y_train)

# Başarı değerlendirme

# Train Accuracy
y_pred = log_model.predict(X_train)
accuracy_score(y_train, y_pred)
#0.747

# Test
# AUC Score için y_prob
y_prob = log_model.predict_proba(X_test)[:, 1]

# Diğer metrikler için y_pred
y_pred = log_model.predict(X_test)


# CONFUSION MATRIX
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y_test, y_pred)

# ACCURACY
accuracy_score(y_test, y_pred)
#0.746

# PRECISION
precision_score(y_test, y_pred)
#0.643

# RECALL
recall_score(y_test, y_pred)
#0.654

# F1
f1_score(y_test, y_pred)
#0.648

# ROC CURVE
plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_test, y_prob)
#0.809


# Classification report
print(classification_report(y_test, y_pred))