import numpy as np
import pandas as pd


def get_vect_all(filename_1, filename_2, dis_num):
    TreeAndDv = pd.read_csv(u'./positive_sample.txt', header=None, low_memory=False, sep='\t')
    TreeAndDv_1 = pd.read_csv(u'./negative_sample.txt', header=None, low_memory=False, sep='\t')
    TreeAndDv_3 = pd.read_csv(u'./association_weight.txt', header=None, low_memory=False, sep='\t')
    positive = []
    k = 0
    while k < len(TreeAndDv[0]):
        b = []
        b.append(TreeAndDv[0][k])
        b.append(TreeAndDv[1][k])
        positive.append(b)
        k += 1
    #  负样本
    negative = []
    t = 0
    while t < len(TreeAndDv_1[0]):
        b = []
        b.append(TreeAndDv_1[0][t])
        b.append(TreeAndDv_1[1][t])
        negative.append(b)
        t += 1
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
    elif dis_num == 128:
        j_vect = 0
        all_vect = np.zeros((len(TreeAndDv_2[0]), 128))
        while j_vect < len(TreeAndDv_2[0]):
            t_0 = 0
            while t_0 < 128:
                all_vect[j_vect][t_0] = TreeAndDv_2[t_0][j_vect]
                t_0 += 1
            j_vect += 1 
    else:
        j_vect = 0
        all_vect = np.zeros((len(TreeAndDv_2[0]), 32))
        while j_vect < len(TreeAndDv_2[0]):
            t_0 = 0
            while t_0 < 32:
                all_vect[j_vect][t_0] = TreeAndDv_2[t_0][j_vect]
                t_0 += 1
            j_vect += 1 

    def get_vect(n_1, n_2, dis_num):
        if dis_num == 64:
            end_vect = np.zeros((1, 64))
            lnc_vect = np.zeros((1, 64))
            dis_vect = np.zeros((1, 64))
            mi_vect = np.zeros((1, 64))
        elif dis_num == 128:
            end_vect = np.zeros((1, 128))
            lnc_vect = np.zeros((1, 128))
            dis_vect = np.zeros((1, 128))
            mi_vect = np.zeros((1, 128))
        else:
            end_vect = np.zeros((1, 32))
            lnc_vect = np.zeros((1, 32))
            dis_vect = np.zeros((1, 32))
            mi_vect = np.zeros((1, 32))
        n_1_asso = []
        n_2_asso = []
        #n_score = []
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
            # lncRNA set
            if (0 <= i < 861):
                asso_lnc = []
                asso_lnc.append(n_1)
                asso_lnc.append(i)
                local_lnc = asso_all.index(asso_lnc)
                lnc_vect = score[local_lnc]*1 * all_vect[i] + lnc_vect
            #  disease set
            elif (1297 < i < 1730):
                asso_dis = []
                asso_dis.append(n_1)
                asso_dis.append(i)
                local_dis = asso_all.index(asso_dis)
                dis_vect = score[local_dis]*1 * all_vect[i] + dis_vect
            #  miRNA set
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
    positive_vect = []
    negative_vect = []
    while asso_num < len(positive):
        if dis_num == 64:
            positive_num_1 = positive[asso_num][0]
            positive_num_2 = positive[asso_num][1]
            vect_z = get_vect(positive_num_1, positive_num_2, 64)[0]
            positive_vect.append(vect_z)
            negative_num_1 = negative[asso_num][0]
            negative_num_2 = negative[asso_num][1]
            vect_f = get_vect(negative_num_1, negative_num_2, 64)[0]
            negative_vect.append(vect_f)
            print(asso_num)
            asso_num += 1
        elif dis_num == 128:
            positive_num_1 = positive[asso_num][0]
            positive_num_2 = positive[asso_num][1]
            vect_z = get_vect(positive_num_1, positive_num_2, 128)[0]
            positive_vect.append(vect_z)
            negative_num_1 = negative[asso_num][0]
            negative_num_2 = negative[asso_num][1]
            vect_f = get_vect(negative_num_1, negative_num_2, 128)[0]
            negative_vect.append(vect_f)
            print(asso_num)
            asso_num += 1
        else:
            positive_num_1 = positive[asso_num][0]
            positive_num_2 = positive[asso_num][1]
            vect_z = get_vect(positive_num_1, positive_num_2, 32)[0]
            positive_vect.append(vect_z)
            negative_num_1 = negative[asso_num][0]
            negative_num_2 = negative[asso_num][1]
            vect_f = get_vect(negative_num_1, negative_num_2, 32)[0]
            negative_vect.append(vect_f)
            print(asso_num)
            asso_num += 1
    positive_vect = pd.DataFrame(positive_vect)
    negative_vect = pd.DataFrame(negative_vect)
    all_vect = np.vstack((positive_vect, negative_vect))
    all_vect = pd.DataFrame(all_vect)
    all_vect.to_csv(filename_2, sep=',', index=False, header=False)


get_vect_all("./nodevector/5ep-64dis.csv", './LDVCHN/5ep-64dis.csv', 64)
get_vect_all("./nodevector/5ep-128dis.csv", './LDVCHN/5ep-128dis.csv', 128)
get_vect_all("./nodevector/10ep-64dis.csv", './LDVCHN/10ep-64dis.csv', 64)
get_vect_all("./nodevector/10ep-128dis.csv", './LDVCHN/10ep-128dis.csv', 128)
get_vect_all("./nodevector/15ep-64dis.csv", './LDVCHN/15ep-64dis.csv', 64)
get_vect_all("./nodevector/15ep-128dis.csv", './LDVCHN/15ep-128dis.csv', 128)
get_vect_all("./nodevector/20ep-64dis.csv", './LDVCHN/20ep-64dis.csv', 64)
get_vect_all("./nodevector/20ep-128dis.csv", './LDVCHN/.csv', 128)
get_vect_all("./nodevector/25ep-64dis.csv", './LDVCHN/25ep-64dis.csv', 64)
get_vect_all("./nodevector/25ep-128dis.csv", './LDVCHN/25ep-128dis.csv', 128)
get_vect_all("./nodevector/30ep-64dis.csv", './LDVCHN/30ep-64dis.csv', 64)
get_vect_all("./nodevector/30ep-128dis.csv", './LDVCHN/30ep-128dis.csv', 128)
get_vect_all("./nodevector/35ep-64dis.csv", './LDVCHN/35ep-64dis.csv', 64)
get_vect_all("./nodevector/35ep-128dis.csv", './LDVCHN/35ep-128dis.csv', 128)
get_vect_all("./nodevector/40ep-64dis.csv", './LDVCHN/40ep-64dis.csv', 64)
get_vect_all("./nodevector/40ep-128dis.csv", './LDVCHN/40ep-128dis.csv', 128)
get_vect_all("./nodevector/45ep-64dis.csv", './LDVCHN/45ep-64dis.csv', 64)
get_vect_all("./nodevector/45ep-128dis.csv", './LDVCHN/45ep-128dis.csv', 128)
get_vect_all("./nodevector/50ep-64dis.csv", './LDVCHN/50ep-64dis.csv', 64)
get_vect_all("./nodevector/50ep-128dis.csv", './LDVCHN/50ep-128dis.csv', 128)