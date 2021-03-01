import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
from scipy import interp
TreeAndDv = pd.read_csv(u'./label.txt', header=None, low_memory=False, sep='\t')
y = []
for i in TreeAndDv[0]:
    y.append(int(i))
print(len(y))
auc_all = []
acc_all = []
recall_all = []
F1score_all = []
pression_all = []


def xgb_modle(filename_1, dis_num):
    TreeAndDv_3 = pd.read_csv(filename_1, header=None, low_memory=False, sep=',')
    if dis_num == 128:
        j = 0
        train_vect = np.zeros((len(TreeAndDv_3[0]), 128))
        while j < len(TreeAndDv_3[0]):
            t_0 = 0
            while t_0 < 128:
                train_vect[j][t_0] = float(TreeAndDv_3[t_0][j])
                t_0 += 1
            j += 1
        print(len(train_vect[0]))
    elif dis_num == 64:
        j = 0
        train_vect = np.zeros((len(TreeAndDv_3[0]), 64))
        while j < len(TreeAndDv_3[0]):
            t_0 = 0
            while t_0 < 64:
                train_vect[j][t_0] = float(TreeAndDv_3[t_0][j])
                t_0 += 1
            j += 1
        print(len(train_vect[0]))
    else:
        j = 0
        train_vect = np.zeros((len(TreeAndDv_3[0]), 32))
        while j < len(TreeAndDv_3[0]):
            t_0 = 0
            while t_0 < 32:
                train_vect[j][t_0] = float(TreeAndDv_3[t_0][j])
                t_0 += 1
            j += 1
        print(len(train_vect[0]))
    X_train, X_test, y_train, y_test = train_test_split(train_vect, y, test_size=0.2, random_state=112)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    acc = []
    recall = []
    F1score = []
    pression = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    data = train_vect
    label = y
    i = 0
    for train_index, test_index in kf.split(data, label):
        X_train, X_test = data[train_index], data[test_index]
        Y_train, Y_test = np.array(label)[train_index], np.array(label)[test_index]
        model = xgb.XGBClassifier(learning_rate=0.08, n_estimators=200, max_depth=3, min_child_weight=8, seed=0, subsample=0.9, colsample_bytree=0.6, gamma=0.6, reg_alpha=0.1, reg_lambda=2)
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred_proba)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.roc_auc_score(Y_test, y_pred_proba)
        aucs.append(roc_auc)
        acc.append(metrics.accuracy_score(Y_test, y_pred))
        recall.append(metrics.recall_score(Y_test, y_pred))
        F1score.append(metrics.f1_score(Y_test, y_pred))
        pression.append(metrics.precision_score(Y_test, y_pred))
        i += 1
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    mean_acc = np.mean(acc)
    mean_recall = np.mean(recall)
    mean_f1 = np.mean(F1score)
    mean_precesion = np.mean(pression)
    auc_all.append(mean_auc)
    acc_all.append(mean_acc)
    recall_all.append(mean_recall)
    F1score_all.append(mean_f1)
    pression_all.append(mean_precesion)
    print("AUC", mean_auc)
    print('ACC: %.4f' % mean_acc)
    print('Recall: %.4f' % mean_recall)
    print('F1-score: %.4f' % mean_f1)
    print('Precesion: %.4f' % mean_precesion)


xgb_modle('E:\\1119\\3.lncRNA-disease_association_vector\\Minus\\5ep-64dis.csv', 64)
xgb_modle('E:\\1119\\3.lncRNA-disease_association_vector\\Minus\\5ep-128dis.csv', 128)
xgb_modle('E:\\1119\\3.lncRNA-disease_association_vector\\Minus\\10ep-64dis.csv', 64)
xgb_modle('E:\\1119\\3.lncRNA-disease_association_vector\\Minus\\10ep-128dis.csv', 128)

xgb_modle('E:\\1119\\3.lncRNA-disease_association_vector\\Minus\\15ep-64dis.csv', 64)
xgb_modle('E:\\1119\\3.lncRNA-disease_association_vector\\Minus\\\\15ep-128dis.csv', 128)
xgb_modle('E:\\1119\\3.lncRNA-disease_association_vector\\Minus\\20ep-64dis.csv', 64)
xgb_modle('E:\\1119\\3.lncRNA-disease_association_vector\\Minus\\20ep-128dis.csv', 128)

xgb_modle('E:\\1119\\3.lncRNA-disease_association_vector\\Minus\\25ep-64dis.csv', 64)
xgb_modle('E:\\1119\\3.lncRNA-disease_association_vector\\Minus\\25ep-128dis.csv', 128)
xgb_modle('E:\\1119\\3.lncRNA-disease_association_vector\\Minus\\30ep-64dis.csv', 64)
xgb_modle('E:\\1119\\3.lncRNA-disease_association_vector\\Minus\\30ep-128dis.csv', 128)

xgb_modle('E:\\1119\\3.lncRNA-disease_association_vector\\Minus\\35ep-64dis.csv', 64)
xgb_modle('E:\\1119\\3.lncRNA-disease_association_vector\\Minus\\35ep-128dis.csv', 128)
xgb_modle('E:\\1119\\3.lncRNA-disease_association_vector\\Minus\\40ep-64dis.csv', 64)
xgb_modle('E:\\1119\\3.lncRNA-disease_association_vector\\Minus\\40ep-128dis.csv', 128)

xgb_modle('E:\\1119\\3.lncRNA-disease_association_vector\\Minus\\45ep-64dis.csv', 64)
xgb_modle('E:\\1119\\3.lncRNA-disease_association_vector\\Minus\\45ep-128dis.csv', 128)
xgb_modle('E:\\1119\\3.lncRNA-disease_association_vector\\Minus\\50ep-64dis.csv', 64)
xgb_modle('E:\\1119\\3.lncRNA-disease_association_vector\\Minus\\50ep-128dis.csv', 128)

asso_1 = pd.DataFrame()
asso_1['auc'] = auc_all
asso_1['acc'] = acc_all
asso_1['Recall'] = recall_all
asso_1['F1-score'] = F1score_all
asso_1['precesion'] = pression_all
asso_1.to_csv('Xgscore_minus.csv', index=False)