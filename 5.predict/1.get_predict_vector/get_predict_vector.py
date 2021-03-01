import numpy as np
import pandas as pd


def get_vect_all(filename_1, filename_2, dis_num):
    TreeAndDv = pd.read_csv(u'./allnegative_sample.txt', header=None, low_memory=False, sep='\t')
    TreeAndDv_3 = pd.read_csv(u'./asso_weight.txt', header=None, low_memory=False, sep='\t')
    zheng = []
    k = 0
    while k < len(TreeAndDv[0]):
        b = []
        b.append(TreeAndDv[0][k])
        b.append(TreeAndDv[1][k])
        zheng.append(b)
        k += 1
    name_1 = []
    name_2 = []
    score = []
    asso_all = list()
    for i in TreeAndDv_3[0]:
        name_1.append(i)
    for i in TreeAndDv_3[1]:
        name_2.append(i)
    for i in TreeAndDv_3[2]:
        score.append(i)
    print(len(name_1))
    a = 0
    while a < len(TreeAndDv_3[0]):
        b = []
        b.append(TreeAndDv_3[0][a])
        b.append(TreeAndDv_3[1][a])
        asso_all.append(b)
        a += 1
    print(len(asso_all))
    TreeAndDv_2 = pd.read_csv(filename_1, header=None, low_memory=False, sep=',')
    if dis_num == 64:
        j_vect = 0
        all_vect = np.zeros((len(TreeAndDv_2[0]), 64))
        while j_vect < len(TreeAndDv_2[0]):
            t_0 = 0
            while t_0 < 64:
                all_vect[j_vect][t_0] = TreeAndDv_2[t_0][j_vect]
                t_0 += 1
            j_vect += 1 
    else:
        j_vect = 0
        all_vect = np.zeros((len(TreeAndDv_2[0]), 128))
        while j_vect < len(TreeAndDv_2[0]):
            t_0 = 0
            while t_0 < 128:
                all_vect[j_vect][t_0] = TreeAndDv_2[t_0][j_vect]
                t_0 += 1
            j_vect += 1 

    def get_vect(n_1, n_2, dis_num):
        if dis_num == 64:
            end_vect = np.zeros((1, 64))
            lnc_vect = np.zeros((1, 64))
            dis_vect = np.zeros((1, 64))
            mi_vect = np.zeros((1, 64))
        else:
            end_vect = np.zeros((1, 128))
            lnc_vect = np.zeros((1, 128))
            dis_vect = np.zeros((1, 128))
            mi_vect = np.zeros((1, 128))
        n_1_asso = []
        n_2_asso = []
        n_1_local = []
        n_2_local = []
        for item in enumerate(name_1):
            if item[1] == n_1:
                n_1_local.append(item[0])
            elif item[1] == n_2:
                n_2_local.append(item[0])
            else:
                pass 
            
        for i in n_1_local:
            n_1_asso.append(name_2[i])
        for i in n_2_local:
            n_2_asso.append(name_2[i])
        tmp = [j for j in n_1_asso if j in n_2_asso] 
        for i in tmp:
            if (0 <= i < 861):
                asso_lnc = []
                asso_lnc.append(n_1)
                asso_lnc.append(i)
                local_lnc = asso_all.index(asso_lnc)
                lnc_vect = score[local_lnc]*1 * all_vect[i] + lnc_vect
            
            elif (1297 < i < 1730):
                asso_dis = []
                asso_dis.append(n_1)
                asso_dis.append(i)
                local_dis = asso_all.index(asso_dis)
                dis_vect = score[local_dis]*1 * all_vect[i] + dis_vect
                
            elif (860 < i < 1298):
                asso_mi_lnc_num = []
                for item in enumerate(name_1):
                    if item[1] == i:
                        asso_mi_lnc_num.append(item[0])
                mi_vect = (1/len(asso_mi_lnc_num)) * 1 * all_vect[i] + mi_vect
            else:
                print("Error: out of range")
        end_vect = (lnc_vect + dis_vect + mi_vect + all_vect[n_1] + all_vect[n_2]) * (1/(len(tmp) + 2))
        return end_vect

    asso_num = 0
    zheng_vect = []
    while asso_num < len(zheng):
        if dis_num == 64:
            zheng_num_1 = zheng[asso_num][0]
            zheng_num_2 = zheng[asso_num][1]
            vect_z = get_vect(zheng_num_1, zheng_num_2, 64)[0]
            zheng_vect.append(vect_z)
            print(asso_num)
            asso_num += 1
        else:
            zheng_num_1 = zheng[asso_num][0]
            zheng_num_2 = zheng[asso_num][1]
            vect_z = get_vect(zheng_num_1, zheng_num_2, 128)[0]
            zheng_vect.append(vect_z)
            print(asso_num)
            asso_num += 1

    zheng_vect = pd.DataFrame(zheng_vect)
    zheng_vect.to_csv(filename_2, sep=',', index=False, header=False)


get_vect_all('./35ep-128dis.csv', './predict_vector.csv', 128)
