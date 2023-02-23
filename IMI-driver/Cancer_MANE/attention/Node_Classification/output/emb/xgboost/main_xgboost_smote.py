import xlrd
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import XGBClassifier as XGBC
from sklearn.datasets import make_blobs #自创数据集
from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import confusion_matrix as cm, recall_score as recall, roc_auc_score as auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold

#下采样
def xgboost_train(X,Y,kf):
    pre_score =  np.array([])
    y_label = np.array([])
    num=1
    fc_mcc = np.array([])
    fc_threshold = np.array([])
    fc_significant = np.array([])
    test_gather=[]
    for tv_index, test_index in kf.split(X):
        print("epoch",num,"TRAIN:", tv_index, "TEST:", test_index)
        num=num+1
        X_tv, X_test = X.iloc[tv_index], X.iloc[test_index]
        y_tv, y_test = Y.iloc[tv_index], Y.iloc[test_index]
        X_train,X_validation,Y_train,Y_validation = TTS(X_tv,y_tv,test_size=0.3,random_state=420)
        res_score=np.zeros((len(y_test)))
        dtest = xgb.DMatrix(X_test,y_test)

        candiate_mcc = np.array([])
        candiate_threshold = np.array([])
        candiate_significant = np.array([])
        for i in range(10):
            tr_x_pos = X_train[Y_train==1]
            tr_x_neg = X_train[Y_train==0]
            tr_x_pos= pd.DataFrame(tr_x_pos.values)

            tr_y_pos = Y_train[Y_train==1]
            tr_y_neg = Y_train[Y_train==0]
            tr_y_pos= pd.DataFrame(tr_y_pos.values)

            #恢复索引
            tr_x_neg = tr_x_neg.reset_index(drop=True)
            tr_y_neg = tr_y_neg.reset_index(drop=True)

            #抽取负样本
            data = tr_x_neg .values
            label = tr_y_neg.values
            sample_num = int(len(tr_x_pos)) # 抽取与正样本相等数量的负样本
            sample_list = [i for i in range(len(data))] # [0, 1, 2, 3]
            sample_list = random.sample(sample_list, sample_num) # [1, 2]
            data = data[sample_list,:] # array([[ 4,  5,  6,  7], [ 8,  9, 10, 11]])
            label = label[sample_list] # array([2, 3]

            #合并正负样本组成训练集
            data = pd.DataFrame(data)
            label = pd.DataFrame(label)
            train_label  = pd.concat([tr_y_pos,label],axis=0).values
            train_data  = pd.concat([tr_x_pos,data],axis=0).values

            #打乱样本
            shuffle_ix = np.random.permutation(np.arange(len(train_data)))
            train_data = train_data[shuffle_ix]
            train_label = train_label[shuffle_ix].ravel()

            #建立xbg模型
            dtrain = xgb.DMatrix(train_data,train_label)
            param = {'objective':'binary:logistic'
              ,"subsample":1
              ,"eta":0.05
              ,"gamma":1
              ,"alpha":0
              ,"lambda":1
              ,"max_depth":6
              ,"colsample_bytree":1
              ,"colsample_bylevel":1
              ,"colsample_bynode":1
              ,"eval_metric":"auc"}
            num_round = 180
            xgbc = xgb.train(param, dtrain, num_round)

            #确定阈值
            temp_mcc = np.array([])
            temp_threshold = np.array([])
            temp_significant = np.array([])
            validation_label = Y_validation.copy().values
            dvalidation = xgb.DMatrix(X_validation,Y_validation)
            pre_v_score = xgbc.predict(dvalidation)
            for tt in np.arange(0,1,0.01):
                corr_identify = np.array([]);
                pre_validation_label = pre_v_score.copy();
                pre_validation_label[pre_validation_label >= tt] = 1;
                pre_validation_label[pre_validation_label < tt] = 0;
                #正确识别个数
                corr_identify = validation_label + pre_validation_label
                corr_identify = corr_identify[corr_identify == 2];#正确识别
                significant=len(pre_validation_label[pre_validation_label == 1])

                temp_mcc = np.append(temp_mcc,np.array(matthews_corrcoef(validation_label,pre_validation_label)))
                temp_threshold = np.append(temp_threshold , np.array(tt) )
                temp_significant = np.append(temp_significant , np.array(significant) )

            candiate_mcc = np.append(candiate_mcc , np.array(max(temp_mcc)) )
            candiate_threshold = np.append(candiate_threshold , np.array(temp_threshold[np.argmax(np.array(temp_mcc))]) )
            candiate_significant = np.append(candiate_significant , np.array(temp_significant[np.argmax(np.array(temp_mcc))]) )

            y_score = xgbc.predict(dtest)
            res_score=res_score+y_score

        fc_mcc = np.append(fc_mcc , np.array(max(candiate_mcc)) )
        fc_threshold = np.append(fc_threshold , np.array(candiate_threshold[np.argmax(np.array(candiate_mcc))]) )
        fc_significant = np.append(fc_significant , np.array(candiate_significant[np.argmax(np.array(candiate_mcc))]) )
        
        test_gather=np.append(test_gather,test_index)
        res_score = res_score/10
        pre_score = np.append(pre_score, res_score, axis=0)
        y_label = np.append(y_label, y_test, axis=0)
    return pre_score,y_label,fc_threshold,test_gather

#somete采样
def xgboost_smote_train(X,Y,kf):
    pre_score =  np.array([])
    y_label = np.array([])
    num=1
    fc_mcc = np.array([])
    fc_threshold = np.array([])
    fc_significant = np.array([])
    test_gather=[]
    for tv_index, test_index in kf.split(X):
        print("epoch",num,"TRAIN:", tv_index, "TEST:", test_index)
        num=num+1
        X_tv, X_test = X.iloc[tv_index], X.iloc[test_index]
        y_tv, y_test = Y.iloc[tv_index], Y.iloc[test_index]
        X_train,X_validation,Y_train,Y_validation = TTS(X_tv,y_tv,test_size=0.3,random_state=420)
        res_score=np.zeros((len(y_test)))
        dtest = xgb.DMatrix(X_test,y_test)

        candiate_mcc = np.array([])
        candiate_threshold = np.array([])
        candiate_significant = np.array([])

        # 建模
        smote_model = SMOTE(k_neighbors=3, random_state=42)
        #smote_model = ADASYN()
        # fit
        train_data, train_label = smote_model.fit_resample(X_train, Y_train)


        #建立xbg模型
        dtrain = xgb.DMatrix(train_data,train_label)
        param = {'objective':'binary:logistic'
          ,"subsample":1
          ,"eta":0.05
          ,"gamma":1
          ,"alpha":0
          ,"lambda":1
          ,"max_depth":6
          ,"colsample_bytree":1
          ,"colsample_bylevel":1
          ,"colsample_bynode":1
          ,"eval_metric":"auc"
          ,"n_jobs":90}
        num_round = 180
        xgbc = xgb.train(param, dtrain, num_round)

        #确定阈值
        temp_mcc = np.array([])
        temp_threshold = np.array([])
        temp_significant = np.array([])
        validation_label = Y_validation.copy().values
        dvalidation = xgb.DMatrix(X_validation,Y_validation)
        pre_v_score = xgbc.predict(dvalidation)
        for tt in np.arange(0,1,0.01):
            corr_identify = np.array([]);
            pre_validation_label = pre_v_score.copy();
            pre_validation_label[pre_validation_label >= tt] = 1;
            pre_validation_label[pre_validation_label < tt] = 0;
            #正确识别个数
            corr_identify = validation_label + pre_validation_label
            corr_identify = corr_identify[corr_identify == 2];#正确识别
            significant=len(pre_validation_label[pre_validation_label == 1])

            temp_mcc = np.append(temp_mcc,np.array(matthews_corrcoef(validation_label,pre_validation_label)))
            temp_threshold = np.append(temp_threshold , np.array(tt) )
            temp_significant = np.append(temp_significant , np.array(significant) )

        candiate_mcc = np.append(candiate_mcc , np.array(max(temp_mcc)) )
        candiate_threshold = np.append(candiate_threshold , np.array(temp_threshold[np.argmax(np.array(temp_mcc))]) )
        candiate_significant = np.append(candiate_significant , np.array(temp_significant[np.argmax(np.array(temp_mcc))]) )

        y_score = xgbc.predict(dtest)
        res_score=res_score+y_score

        fc_mcc = np.append(fc_mcc , np.array(max(candiate_mcc)) )
        fc_threshold = np.append(fc_threshold , np.array(candiate_threshold[np.argmax(np.array(candiate_mcc))]) )
        fc_significant = np.append(fc_significant , np.array(candiate_significant[np.argmax(np.array(candiate_mcc))]) )

        test_gather=np.append(test_gather,test_index)
        res_score = res_score
        pre_score = np.append(pre_score, res_score, axis=0)
        y_label = np.append(y_label, y_test, axis=0)
    return pre_score,y_label,fc_threshold,test_gather

def calculate_metric(gt, pred): 
    pred[pred>0.5]=1
    pred[pred<1]=0
    confusion = confusion_matrix(gt,pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    Sensitivity= TP / float(TP+FN)
    Specificity = TN / float(TN+FP)
    return Sensitivity,Specificity
def main():

    #读取数据

    #读取共同网络
    raw_consistent = pd.read_csv('Embedding_3_64_50_consistent.csv',header=0)
    print(raw_consistent)

    raw_feature_45 = pd.read_csv('feature_45.csv',header=0)

    raw_label = pd.read_csv('benchmark标签.csv',header=0)

    #处理数据

    #共同信息
    X_raw_consistent = raw_consistent.iloc[:,1:raw_consistent.shape[1]].copy()#共同网络
    X_raw_feature_45 = raw_feature_45.copy()

    Y_raw_label = raw_label.copy()

    X_consistent =  pd.concat([pd.DataFrame(X_raw_consistent), pd.DataFrame(X_raw_feature_45)], axis=1)


    cancer_t = pd.read_excel('diff_cancer_29.xlsx', header=0)

    #共同信息
    X_raw_consistent = raw_consistent.iloc[:,1:raw_consistent.shape[1]].copy()#共同网络
    X_raw_feature_45 = raw_feature_45.copy()

    Y_raw_label = raw_label.copy()
    #选择基准数据
    Y=Y_raw_label.iloc[:,3].copy()
    Y[Y != 1]=0 #xbg要把负类设置为0
    threshold =  np.array([])
    num =  np.array([])
    Auc =  np.array([])
    Acc = np.array([])
    F1 = np.array([])
    Mcc = np.array([])
    Precision = np.array([])
    Sensitivity = np.array([])
    Specificity = np.array([])
    all_pre_sorce = np.arange(0,20530,1)
    all_pre_label = np.arange(0,20530,1)
    for i in np.arange(0,len(cancer_t),1):
        cancer = cancer_t.iloc[i,0]
        specific_file = 'Embedding_5_64_50_specific_'+cancer+'.csv'
        feature_specific_file='Dif_'+cancer+'.csv'

        #读取特性网络
        raw_specific = pd.read_csv(specific_file ,header=0)
        #print(raw_specific)
        #读取特性性特征（两种差异）
        feature_specific= pd.read_csv(feature_specific_file)


        #特异性信息
        X_raw_specific = raw_specific.iloc[:,1:raw_specific.shape[1]].copy()#特异性网络
        X_feature_specific = feature_specific.copy()#差异特征

        #两种特征
        X_specific =  pd.concat([pd.DataFrame(X_raw_specific), pd.DataFrame(X_feature_specific)], axis=1)
        X_specific = X_raw_specific.copy()
        X_consistent =  pd.concat([pd.DataFrame(X_raw_consistent), pd.DataFrame(X_raw_feature_45)], axis=1)
        #全部特征

        #####可替换####
        X_all = pd.concat([pd.DataFrame(X_specific), pd.DataFrame(X_consistent)], axis=1)
        X=X_all.copy()
        X.columns = range(0,X.shape[1])

        #分组
        kf = KFold(10,shuffle=True)
        #print( kf.get_n_splits(X))

        print('Calculating '+cancer+':')
        pre_score,y_label,fc_threshold,test_gather=xgboost_smote_train(X,Y,kf)

        #pre_score,y_label,fc_threshold,test_gather=xgboost_train(X,Y,kf)

        pre = pre_score.copy()
        pre[pre >= np.average(fc_threshold) ] = 1
        pre[pre < np.average(fc_threshold) ] = 0

        threshold = np.append(threshold,np.average(fc_threshold)).reshape(-1,1)
        num = np.append(num, pre[pre==1].shape[0]).reshape(-1,1)

        fpr,tpr,thresholds=roc_curve(y_label,pre_score)
        auc(fpr,tpr)
        Auc = np.append(Auc,auc(fpr,tpr)).reshape(-1,1)

        Mcc = np.append(Mcc,matthews_corrcoef(y_label, pre)).reshape(-1,1)
        F1 = np.append(F1,f1_score(y_label,pre,average='macro')).reshape(-1,1)
        Acc = np.append(Acc,accuracy_score(y_label, pre)).reshape(-1,1)

        sensitivity,specificity =calculate_metric(y_label,pre)
        Sensitivity = np.append(Sensitivity,sensitivity).reshape(-1,1)
        Specificity = np.append(Specificity,specificity).reshape(-1,1)
        Precision = np.append(Precision,precision_score(y_label,pre),).reshape(-1,1)
        Gold_standard = np.hstack((threshold,num,Auc,Mcc,F1,Acc,Sensitivity,Specificity,Precision,))
        Gold_standard=pd.DataFrame(Gold_standard,columns=['Threshold','Num','Auc','Mcc','F1','Acc','Sensitivity','Specificity','Precision'])
        Gold_standard.to_excel('Gold_standard_smote.xlsx',header=1,index=0)

        test_gather=test_gather.reshape(len(test_gather),1)
        pre_score=pre_score.reshape(len(pre_score),1)
        pre=pre.reshape(len(pre),1)
        y_label=y_label.reshape(len(y_label),1)


        #输出为测试集索引，测试集预测分数，测试集预测值，测试集标签
        end=np.hstack((test_gather,pre_score,pre,y_label))
        end=pd.DataFrame(end,columns=['index','pre_score','pre_label','true_abel'])
        #排序
        end = end.sort_values(by='index', ascending=True)

        all_pre_sorce = pd.concat([pd.DataFrame(all_pre_sorce),pd.DataFrame(end.iloc[:,1].copy().reset_index(drop=True))], axis=1)
        all_pre_sorce.to_excel('pre_sorce_smote.xlsx',header=1,index=0)
        all_pre_label = pd.concat([pd.DataFrame(all_pre_label),pd.DataFrame(end.iloc[:,2].copy().reset_index(drop=True))], axis=1)
        all_pre_label.to_excel('pre_label_smote.xlsx',header=1,index=0)

if __name__ == '__main__':
    main()
