from helpers.eda import *
from helpers.data_prep import *

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split

df = pd.read_csv("titanic.csv")
df.columns = [col.upper() for col in df.columns]
df.shape
df.head()

def titanic_data_prep(df):
    #############################################
    # 1. Feature Engineering
    #############################################
    # Cabin bool
    df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
    # Name count
    df["NEW_NAME_COUNT"] = df["NAME"].str.len()
    # name word count
    df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
    # name dr
    df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
    # name title
    df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
    # family size
    df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
    # age_pclass
    df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
    # is alone
    df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
    df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
    # age level
    df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
    # sex x age
    df.loc[(df['SEX'] == 'male') & (df['NEW_AGE_CAT'] == 'young'), 'NEW_SEX_CAT'] = 'youngmale'
    df.loc[(df['SEX'] == 'male') & (df['NEW_AGE_CAT'] == 'mature'), 'NEW_SEX_CAT'] = 'maturemale'
    df.loc[(df['SEX'] == 'male') & (df['NEW_AGE_CAT'] == 'senior'), 'NEW_SEX_CAT'] = 'seniormale'
    df.loc[(df['SEX'] == 'female') & (df['NEW_AGE_CAT'] == 'young'), 'NEW_SEX_CAT'] = 'youngfemale'
    df.loc[(df['SEX'] == 'female') & (df['NEW_AGE_CAT'] == 'mature'), 'NEW_SEX_CAT'] = 'maturefemale'
    df.loc[(df['SEX'] == 'female') & (df['NEW_AGE_CAT'] == 'senior'), 'NEW_SEX_CAT'] = 'seniorfemale'

    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    #############################################
    # 2. Outliers (Aykırı Değerler)
    #############################################

    for col in num_cols:
        replace_with_thresholds(df, col)
    #############################################
    # 3. Missing Values (Eksik Değerler)
    #############################################
    df.drop("CABIN", inplace=True, axis=1)
    df.drop(["TICKET", "NAME"], inplace=True, axis=1)
    df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))
    df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

    df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

    df.loc[(df['SEX'] == 'male') & (df['NEW_AGE_CAT'] == 'young'), 'NEW_SEX_CAT'] = 'youngmale'
    df.loc[(df['SEX'] == 'male') & (df['NEW_AGE_CAT'] == 'mature'), 'NEW_SEX_CAT'] = 'maturemale'
    df.loc[(df['SEX'] == 'male') & (df['NEW_AGE_CAT'] == 'senior'), 'NEW_SEX_CAT'] = 'seniormale'
    df.loc[(df['SEX'] == 'female') & (df['NEW_AGE_CAT'] == 'young'), 'NEW_SEX_CAT'] = 'youngfemale'
    df.loc[(df['SEX'] == 'female') & (df['NEW_AGE_CAT'] == 'mature'), 'NEW_SEX_CAT'] = 'maturefemale'
    df.loc[(df['SEX'] == 'female') & (df['NEW_AGE_CAT'] == 'senior'), 'NEW_SEX_CAT'] = 'seniorfemale'
    df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

    #############################################
    # 4. Label Encoding
    #############################################

    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
                   and df[col].nunique() == 2]

    for col in binary_cols:
        df = label_encoder(df, col)

    #############################################
    # 5. Rare Encoding
    #############################################
    df = rare_encoder(df, 0.01)

    #############################################
    # 6. One-Hot Encoding
    #############################################
    ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

    df = one_hot_encoder(df, ohe_cols)
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    num_cols = [col for col in num_cols if "PASSENGERID" not in col]
    useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                    (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
    df.drop(useless_cols, axis=1, inplace=True)
    #############################################
    # 7. Standart Scaler
    #############################################
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

df.head()

my_df = titanic_data_prep(df)

my_df.head()

# model

y = my_df["SURVIVED"]
X = my_df.drop(["PASSENGERID","SURVIVED"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42)

log_model = LogisticRegression().fit(X_train, y_train)

log_model.intercept_
log_model.coef_

# tahmin

y_pred = log_model.predict(X_train)
# tahmin edilen ve gerçek olan değerlerin ilk 10'unu gözlemledik.
y_pred[0:10]
y_train[0:10]

# başarı değerlendirme
# train accuracy

accuracy_score(y_train, y_pred)
#0.84

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

# accuracy
accuracy_score(y_test, y_pred)
#0.826

# precision
precision_score(y_test, y_pred)
#0.794

# recall
recall_score(y_test, y_pred)
#0.783

# F1
f1_score(y_test, y_pred)
#0.789

# ROC CURVE
plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()
