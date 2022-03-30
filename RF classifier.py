import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import os
from typing import Counter

file_path = r"C:\Users\12445\Desktop\magnetite\fill.xlsx"#Please enter the path to the Supplementary table S4
data = pd.read_excel(file_path)
df = data.loc[:, ["mtype", "Ti", "V", "Mg", "Mn", "Al", "Si", "Zn", "Ga"]]

X = df.copy(deep=True)
y = X.pop('mtype')
print(y.value_counts())  # number of samples in each class
y_int, index = pd.factorize(y, sort=True)
y = y_int
index
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
X_compare = np.log(X_train)
X_compare = StandardScaler().fit_transform(X_compare)
models = (RandomForestClassifier(), )

for clf in models:
    scores = cross_val_score(clf, X_compare, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
    print(f'{scores.mean():2.2f}' + '±' + f'{scores.std():2.2f}')
    
log_transformer = FunctionTransformer(np.log, validate=True)
pipe_clf = make_pipeline(log_transformer, StandardScaler(), RandomForestClassifier(oob_score=True, random_state=10, class_weight='balanced'))
pipe_clf

param_grid={"randomforestclassifier__n_estimators": [220, 240, 260, 280, 300], "randomforestclassifier__max_depth": [10, 15, 20, 25, 30]}
grid = GridSearchCV(
    pipe_clf, param_grid=param_grid, cv=10, scoring="f1_macro", n_jobs=-1, refit=True
)
grid.fit(X_train, y_train)
print(
    "The best parameters are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_)
)

y_test_pred = grid.predict(X_test)
t_train_pred = grid.predict(X_train)
# train set report
print(classification_report(y_train, t_train_pred))

#feature importence
features = list(X_test.columns)
feature_importances = grid.best_estimator_._final_estimator.feature_importances_
indices = np.argsort(feature_importances)[::-1]
num_features = len(feature_importances)
plt.rc('font',family='Times New Roman',size=12)
plt.figure()
plt.title("Feature importances")
plt.bar(range(num_features), feature_importances[indices], color="grey", align="center")
plt.xticks(range(num_features), [features[i] for i in indices], rotation='45')
plt.xlim([-1, num_features])
plt.show()
for i in indices:
    print ("{0} - {1:.3f}".format(features[i], feature_importances[i]))

# final evaluation
print(classification_report(y_test, y_test_pred, output_dict=False))
print(confusion_matrix(y_test, y_test_pred))

#Confusion matrix
label_order = [
    "skarn",
    "volcanic",
    "IOA",
    "IOCG",
    "magmatic",
    "porphry",
    "BIF",
    "orogenic",
]
for i in range(len(label_order)):
    if label_order[i] == "IOA":
        pass
    elif label_order[i] == "IOCG":
        pass
    elif label_order[i] == "BIF":
        pass
    else:
        label_order[i] = label_order[i].capitalize()
index = list(index)
for i in range(len(index)):
    if index[i] == "IOA":
        pass
    elif index[i] == "IOCG":
        pass
    elif index[i] == "BIF":
        pass
    else:
        tem = index[i].capitalize()
        index[i] = tem
cm = pd.DataFrame(
    confusion_matrix(y_test, y_test_pred), columns=index, index=index
)
plt.rc('font',family='Times New Roman',size=8)
f, ax = plt.subplots(figsize=(3.15, 2.94))
sns.heatmap(cm, annot=True, fmt="d", linewidths=0.5, ax=ax, cmap="YlGn")
ax.set_title("Confusion matrix of test set(RF)", fontsize=8)
ax.set_xlabel("Predictions", fontsize=8)
ax.set_ylabel("True labels", fontsize=8)
plt.xticks(rotation=45)
plt.tight_layout()
f.savefig(r'C:\Users\12445\Desktop\./confusion_matrix_RF.tiff', dpi=600)
f.savefig(r'C:\Users\12445\Desktop\./confusion_matrix_RF.pdf', dpi=600)

print(
    """
RF classifier to predict the genetic classes of the magnetite source with "Ti", "V", "Mg", "Mn", "Al", "Si", "Zn", "Ga" values,
Please enter the path of the .xlsx data file.(for example: /path/to/file/example_data.xlsx )
The data are supposed to contain all the 8 features above for prediction.
If any one of the features is missing in a sample, that sample will be discarded.
The columns' names of Ti, V, Mg, Mn, Al, Si, Zn, Ga should be exactly as listed above without any prefix and suffix
and MAKE SURE this column name row is the FIRST row.
"""
)
data_file_path = r"C:\Users\12445\Desktop\magnetite\Makeng_test.xlsx"#Please enter the path to the data file
df = pd.read_excel(data_file_path)

index = ['BIF', 'IOA', 'IOCG', 'Magmatic', 'Orogenic', 'Porphyry', 'Skarn', 'Volcanic']
elements = ["Ti", "V", "Mg", "Mn", "Al", "Si", "Zn", "Ga"]
for element in elements:
    df[element] = pd.to_numeric(df[element], errors="coerce")

to_predict = df.loc[:, elements].dropna()
to_predict.reset_index(drop=True, inplace=True)
print(f"{to_predict.shape[0]} samples available")
print(to_predict.describe())
predict_res = grid.predict(to_predict)
predict_res = list(predict_res)
for i, ind in enumerate(predict_res):
    predict_res[i] = index[ind]

c: Counter[str] = Counter(predict_res)
if not c:
    input("no sample with the 8 features detected!")
    raise SystemExit()
    
proba = grid.predict_proba(to_predict)
predict_res = np.array(predict_res)
predict_res = predict_res.reshape((predict_res.shape[0], 1))
res = np.concatenate([predict_res, proba], axis=1)
res = pd.DataFrame(res, columns=['pred_magnetite_type', 'BIF_proba', 'IOA_proba', 'IOCG_proba', 'Magmatic_proba', 'Orogenic_proba', 'Porphyry_proba', 'Skarn_proba','Volcanic_proba'])
pd.set_option('display.max_columns', 10)
print('Detailed report preview:\n', res)

print("The samples are predicted respectively to be: ")
print(c.most_common(), "\n")
print(
    f"The most possible type of the group of samples is: {c.most_common(1)[0][0]}.\n"
)

if input('Save report? (y/n): ').lower() == 'y':
    base_filename = os.path.basename(data_file_path)
    prefix, _ = os.path.splitext(base_filename)
    save_name = prefix + '_result.xlsx'
    res2 = pd.concat([to_predict['V'], res], axis=1, )
    output = df.join(res2.set_index('V'), on='V')
    output.to_excel(save_name)
    print(f'{save_name} saved.')
input("Press any key to exit.")