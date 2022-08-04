import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import cross_val_score
import os
from typing import Counter
from imblearn.over_sampling import SMOTE

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
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
X_train=X_resampled
y_train=y_resampled
X_compare = np.log(X_train)
X_compare = StandardScaler().fit_transform(X_compare)
C = 1
models = (
          svm.SVC(kernel='linear', C=C, class_weight=None),
          svm.SVC(kernel='rbf', C=C), #, class_weight='balanced'),
         )
for clf in models:
    scores = cross_val_score(clf, X_compare, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
    print(f'{scores.mean():2.2f}' + '±' + f'{scores.std():2.2f}')

log_transformer = FunctionTransformer(np.log, validate=True)
pipe_clf = make_pipeline(log_transformer, StandardScaler(), SVC(cache_size=1000, class_weight=None, probability=True))
pipe_clf    
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
C_range = np.logspace(-2, 10, 13, base=10)
gamma_range = np.logspace(-7, 4, 12, base=10)
param_grid = {"svc__kernel": ["rbf"], "svc__gamma": gamma_range, "svc__C": C_range}
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=2)
print(C_range)
grid = GridSearchCV(
    pipe_clf, param_grid=param_grid, cv=10, scoring="f1_macro", n_jobs=-1, refit=True
)
grid.fit(X_train, y_train)
print(
    "The best parameters are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_)
)

# Visualization
scores = grid.cv_results_["mean_test_score"].reshape(len(C_range), len(gamma_range))
plt.figure(figsize=(3.75, 3.75))
plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(
    scores,
    interpolation="nearest",
    cmap=plt.cm.cividis,
    norm=MidpointNormalize(vmin=0.2, midpoint=0.8),
)
plt.xlabel("gamma")
plt.ylabel("C")
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45, ha="right")
plt.yticks(np.arange(len(C_range)), C_range)
plt.title("Validation Macro F1 Score")
plt.savefig(r'C:\Users\12445\Desktop\heatmap_SVM.pdf', dpi=600)
plt.savefig(r'C:\Users\12445\Desktop\heatmap_SVM.tiff', dpi=600)
plt.show()

y_test_pred = grid.predict(X_test)
t_train_pred = grid.predict(X_train)

# train set report
print(classification_report(y_train, t_train_pred))

# final evaluation
print(classification_report(y_test, y_test_pred, output_dict=False))
print(confusion_matrix(y_test, y_test_pred))

#Confusion matrix
label_order = [
    "Skarn",
    "Volcanic",
    "IOA",
    "IOCG",
    "Magmatic",
    "Porphry",
    "BIF",
    "Orogenic",
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
ax.set_title("Confusion matrix of test set(SVM)",fontsize=8)
ax.set_xlabel("Predictions")
ax.set_ylabel("True labels")
plt.xticks(rotation=45)
plt.tight_layout()
# f.suptitle('Fig. 5', x=0.05)
f.savefig(r'C:\Users\12445\Desktop\./confusion_matrix_SVM.tiff', dpi=600)
f.savefig(r'C:\Users\12445\Desktop\./confusion_matrix_SVM.pdf', dpi=600)

print(
    """
SVM classifier to predict the genetic classes of the magnetite source with "Ti", "V", "Mg", "Mn", "Al", "Si", "Zn", "Ga" values,
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
