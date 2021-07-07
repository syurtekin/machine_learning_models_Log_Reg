import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, classification_report, plot_roc_curve

df = pd.read_csv("../input/cardiovascular-disease-dataset/cardio_train.csv", sep = ";")


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x :'% 3f' % x)

df.head()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

df = df.drop('id', axis=1)
df.head()

#convert it to age by years
df["age"] = round(df["age"] / 365)

###### DATA PREP & EDA ###########

def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name, q1=0.10, q3=0.90):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df, col))

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

df.isnull().values.any()

######## Feature Engineering #######

df.loc[(df["age"] < 18), "NEW_AGE"] = "Young"
df.loc[(df["age"] > 18) & (df["age"] < 56), "NEW_AGE"] = "Mature"
df.loc[(df["age"] >= 56), "NEW_AGE"] = "Old"

cols1 = df["weight"]
cols2 = df["height"] / 100

df["bmi"] = (cols1) / (cols2)**2

df.loc[(df["bmi"] < 18.5), "NEW_BMI"] = "under"
df.loc[(df["bmi"] >= 18.5) & (df["bmi"] <= 24.99) ,"NEW_BMI"] = "healthy"
df.loc[(df["bmi"] >= 25) & (df["bmi"] <= 29.99) ,"NEW_BMI"]= "over"
df.loc[(df["bmi"] >= 30), "NEW_BMI"] = "obese"

df.loc[(df["ap_lo"])<=89, "BLOOD_PRESSURE"] = "normal"
df.loc[(df["ap_lo"])>=90, "BLOOD_PRESSURE"] = "hyper"
df.loc[(df["ap_hi"])<=120, "BLOOD_PRESSURE"] = "normal"
df.loc[(df["ap_hi"])>120, "BLOOD_PRESSURE"] = "normal"
df.loc[(df["ap_hi"])>=140, "BLOOD_PRESSURE"] = "hyper"

df.head()

df.groupby('age')['cardio'].mean()

df.groupby("smoke")["cardio"].mean()

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

# label encoder
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col].astype(str))
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O"
               and len(df[col].unique()) == 2]

for col in df.columns:
    label_encoder(df, col)

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]

one_hot_encoder(df, ohe_cols, drop_first=True).head()

# scale

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

y = df["cardio"]
X = df.drop(["cardio"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=1)

log_model = LogisticRegression().fit(X_train, y_train)

# Train Accuracy
y_pred = log_model.predict(X_train)
accuracy_score(y_train, y_pred)

y_prob = log_model.predict_proba(X_test)[:, 1]

y_pred = log_model.predict(X_test)

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)

precision_score(y_test, y_pred)

recall_score(y_test, y_pred)

f1_score(y_test, y_pred)

plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

roc_auc_score(y_test, y_prob)

print(classification_report(y_test, y_pred))