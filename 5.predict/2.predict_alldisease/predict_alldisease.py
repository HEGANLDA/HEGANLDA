
import numpy as np
import pandas as pd
import xgboost as xgb
TreeAndDv = pd.read_csv(u'./label.txt', header=None, low_memory=False, sep='\t')
y = []
for i in TreeAndDv[0]:
    y.append(int(i))
print(len(y))
label_asso = pd.read_csv(u'./allnegative_sample.txt', header=None, low_memory=False, sep='\t')
source = []
targe = []
asso_predict = []
k = 0
while k < len(label_asso[0]):
    b = []
    b.append(label_asso[0][k])
    b.append(label_asso[1][k])
    source.append(label_asso[0][k])
    targe.append(label_asso[1][k])
    asso_predict.append(b)
    k += 1
TreeAndDv_3 = pd.read_csv('./35ep-128dis.csv', header=None, low_memory=False, sep=',')
TreeAndDv_4 = pd.read_csv(u'./predict_vector.csv', header=None, low_memory=False, sep=',')

j_0 = 0
predict_vect = np.zeros((len(TreeAndDv_4[0]), 128))
while j_0 < len(TreeAndDv_4[0]):
    t_0 = 0
    while t_0 < 128:
        predict_vect[j_0][t_0] = TreeAndDv_4[t_0][j_0]
        t_0 += 1
    j_0 += 1
print(len(predict_vect[0]))


j = 0
train_vect = np.zeros((len(TreeAndDv_3[0]), 128))
while j < len(TreeAndDv_3[0]):
    t_0 = 0
    while t_0 < 128:
        train_vect[j][t_0] = float(TreeAndDv_3[t_0][j])
        t_0 += 1
    j += 1
print(len(train_vect[0]))

model = xgb.XGBClassifier(learning_rate=0.08, n_estimators=200, max_depth=3, min_child_weight=8, seed=0, subsample=0.9, colsample_bytree=0.6, gamma=0.6, reg_alpha=0.1, reg_lambda=2)

model.fit(train_vect, y)
y_pred = model.predict(predict_vect)
y_pred_proba = model.predict_proba(predict_vect)
print(y_pred)
y_pred = list(y_pred)
y_pred_proba = list(y_pred_proba)
y_pred_proba = pd.DataFrame(y_pred_proba)
predict_save = pd.DataFrame()
predict_save['lncRNA '] = source
predict_save['disease'] = targe
predict_save['predict_label'] = y_pred
predict_save['0_probability'] = y_pred_proba[0]
predict_save['1_probability'] = y_pred_proba[1]
disease_num = []
predict_end = pd.DataFrame()
for i in targe:
    if i not in disease_num:
        disease_num.append(i)
for i in disease_num:
    p = predict_save.loc[predict_save["disease"] == i]
    p_1 = p.sort_values(by="1_probability", inplace=False, ascending=False) 
    p_2 = p_1.head(15)
    predict_end = pd.concat([predict_end, p_2])
print(predict_end)
predict_end.to_csv('./predict_alldisease.csv', index=False)