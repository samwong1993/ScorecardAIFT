#Designed by Sam
#huangsen1993@gmail.com
#导入包
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import os
import config_file as cfg_file
import sys
import datetime
import warnings
warnings.filterwarnings('ignore')
def make_print_to_file(path='./'):
    class Logger(object):
        def __init__(self, filename="Default.txt", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            pass
    fileName = datetime.datetime.now().strftime('day' + '%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)
    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
make_print_to_file(path='./')




# #1）数据读取：训练集数据、测试集数据
# train_data = pd.read_csv('./cs-training.csv')
# test_data = pd.read_csv('./cs-test.csv')
# train_data.head()
# test_data.head()
#
# #更改数据集索引
# train_data.set_index('Unnamed: 0',inplace=True)
# test_data.set_index('Unnamed: 0',inplace=True)
#
# def strange_delete(data):
#     data = data[data['RevolvingUtilizationOfUnsecuredLines'] < 1]
#     data = data[data['age'] > 18]
#     data = data[data['NumberOfTime30-59DaysPastDueNotWorse'] < 80]
#     data = data[data['NumberOfTime60-89DaysPastDueNotWorse'] < 80]
#     data = data[data['NumberOfTimes90DaysLate'] < 80]
#     data = data[data['NumberRealEstateLoansOrLines'] < 50]
#     return data
#
#
# train_data = strange_delete(train_data)
# test_data = strange_delete(test_data)
#
# # 查看经过异常值处理后是否还存在异常值
# train_data.loc[(train_data['RevolvingUtilizationOfUnsecuredLines'] > 1) | (train_data['age'] < 18) | (
#             train_data['NumberOfTime30-59DaysPastDueNotWorse'] > 80) | (
#                            train_data['NumberOfTime60-89DaysPastDueNotWorse'] > 80) | (
#                            train_data['NumberOfTimes90DaysLate'] > 80) | (
#                            train_data['NumberRealEstateLoansOrLines'] > 50)]
# test_data.loc[(test_data['RevolvingUtilizationOfUnsecuredLines'] > 1) | (test_data['age'] < 18) | (
#             test_data['NumberOfTime30-59DaysPastDueNotWorse'] > 80) | (
#                           test_data['NumberOfTime60-89DaysPastDueNotWorse'] > 80) | (
#                           test_data['NumberOfTimes90DaysLate'] > 80) | (test_data['NumberRealEstateLoansOrLines'] > 50)]
# print(train_data.shape)
# print('----------------------')
# print(test_data.shape)
#
# # 3.2缺失值处理
# # 3.2.1对家属数量的缺失值进行删除
# train_data = train_data[train_data['NumberOfDependents'].notnull()]
# print(train_data.shape)
# print('----------------------')
# test_data = test_data[test_data['NumberOfDependents'].notnull()]
# print(test_data.shape)
#
#
#
# # 3.2缺失值处理
# # 3.2.2对月收入缺失值用随机森林的方法进行填充--训练集
# # 创建随机森林函数
# def fillmonthlyincome(data):
#     known = data[data['MonthlyIncome'].notnull()]
#     unknown = data[data['MonthlyIncome'].isnull()]
#     x_train = known.iloc[:, [1, 2, 3, 4, 6, 7, 8, 9, 10]]
#     y_train = known.iloc[:, 5]
#     x_test = unknown.iloc[:, [1, 2, 3, 4, 6, 7, 8, 9, 10]]
#     rfr = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
#     pred_y = rfr.fit(x_train, y_train).predict(x_test)
#     return pred_y
#
#
# # 用随机森林填充训练集缺失值
# predict_data = fillmonthlyincome(train_data)
# train_data.loc[train_data['MonthlyIncome'].isnull(), 'MonthlyIncome'] = predict_data
# print(train_data.info())
#
#
#
# # 3.2.2对月收入缺失值用随机森林的方法进行填充--测试集
# # 创建随机森林函数
# def fillmonthlyincome(data):
#     known = data[data['MonthlyIncome'].notnull()]
#     unknown = data[data['MonthlyIncome'].isnull()]
#     x_train = known.iloc[:, [2, 3, 4, 6, 7, 8, 9, 10]]
#     y_train = known.iloc[:, 5]
#     x_test = unknown.iloc[:, [2, 3, 4, 6, 7, 8, 9, 10]]
#     rfr = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
#     pred_y = rfr.fit(x_train, y_train).predict(x_test)
#     return pred_y
#
#
# # 用随机森林填充测试集缺失值
# predict_data = fillmonthlyincome(test_data)
# test_data.loc[test_data['MonthlyIncome'].isnull(), 'MonthlyIncome'] = predict_data
# print(test_data.info())
#
# # 缺失值和异常值处理完后进行检查
# print(train_data.isnull().sum())
# print('----------------------')
# print(test_data.isnull().sum())
#
# label = 'SeriousDlqin2yrs'
# data = pd.concat([train_data,test_data])

# continuous_var = ['RevolvingUtilizationOfUnsecuredLines','age','DebtRatio','MonthlyIncome']
# discrete_var = ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
#




#
#
# data = pd.read_csv("./default_credit_card.csv")
# data.drop(columns="ID",inplace=True)
# data.set_index('label',inplace=True)
# data.reset_index(inplace=True)
# # train_data = data.iloc[:20000,:]
# # test_data = data.iloc[20000:,:]
# # test_data.reset_index(drop=True,inplace=True)
# # continuous_var = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2',
# #        'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
# #        'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
# # discrete_var = ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
# #        'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
# label = "label"






# data = pd.read_csv("./data.csv")
# data['Daily payment activity']/data['Daily payment activity'].max()
# data["Daily average YueBao balance"]/data["Daily average YueBao balance"].max()
# data['Security funds in Taobao']/data['Security funds in Taobao'].max()
# data["Daily average Alipay Wallet balance"]/data["Daily average Alipay Wallet balance"].max()
# data['label'] = data["Daily average Alipay Wallet balance"]/data["Daily average Alipay Wallet balance"].max() + data['Daily payment activity']/data['Daily payment activity'].max() + data["Daily average YueBao balance"]/data["Daily average YueBao balance"].max() + data['Security funds in Taobao']/data['Security funds in Taobao'].max()
# data.drop(columns="Total amount of ecommerce purchases",inplace=True)
# data.sort_values(by="label", ascending=False,inplace=True)
# data["label"] = 0
# data.loc[:600,["label"]] = 1
# data = data.sample(frac=1)
# label = 'label'


data = pd.read_csv("./seller.csv")
data = data.loc[:,['asin','rank', 'review', 'review_increase', 'price_1', 'price_2',
       'price_3', 'price_4', 'price_5', 'price_6', 'price_7', 'price_8',
       'price_9', 'price_10', 'price_11', 'price_12', 'sales_1', 'sales_2',
       'sales_3', 'sales_4', 'sales_5', 'sales_6', 'sales_7', 'sales_8',
       'sales_9', 'sales_10', 'sales_11', 'sales_12','label']]
data = data[data['label'] != -1]
data = data[data.loc[:,['price_1', 'price_2',
       'price_3', 'price_4', 'price_5', 'price_6', 'price_7', 'price_8',
       'price_9', 'price_10', 'price_11', 'price_12', 'sales_1', 'sales_2',
       'sales_3', 'sales_4', 'sales_5', 'sales_6', 'sales_7', 'sales_8',
       'sales_9', 'sales_10', 'sales_11', 'sales_12']].astype(bool).astype(int).sum(axis=1)>14]
label = 'label'

data.loc[data.loc[:,['sales_1', 'sales_2',
       'sales_3', 'sales_4', 'sales_5', 'sales_6', 'sales_7', 'sales_8',
       'sales_9', 'sales_10', 'sales_11', 'sales_12']].sum(axis=1)>8000,'label'] = 0
data.loc[data.loc[:,['sales_1', 'sales_2',
       'sales_3', 'sales_4', 'sales_5', 'sales_6', 'sales_7', 'sales_8',
       'sales_9', 'sales_10', 'sales_11', 'sales_12']].sum(axis=1)<50,'label'] = 1

# continuous_var = ['Total inflow of funds in Alipay','Transaction volume','Network effect score','Daily average Alipay Wallet balance']
# discrete_var = ['House property', 'Car property', 'Number of credit cards',
#        "Owners age", "Gender", "Firms age", 'City tiers','Shop rating','Log-ins','Security funds in Taobao', 'VIP class',
#        'Number of good feedtbacks from clients','Link to credit card', 'Daily payment activity',"Daily average YueBao balance"]
# data["label"] = np.random.randint(0, 2,(len(data), 1))
# data.set_index('label',inplace=True)
# data.reset_index(inplace=True)
# train_data = data.iloc[:40000,:]
# test_data = data.iloc[40000:,:]
# test_data.reset_index(drop=True,inplace=True)

#重要参数score_range：评分卡模型参数
score_range = [45,165]
training_size = 0.8
#务必将label放在第一列，用以下两行代码可以实现
data.set_index(label,inplace=True)
data.reset_index(inplace=True)

train_data = data.iloc[:int(len(data)*training_size),:]
test_data = data.iloc[int(len(data)*training_size):,:]
test_data.reset_index(drop=True,inplace=True)
train_data.to_csv('./train.csv')
test_data.to_csv('./test.csv')
train_data = train_data.drop('asin',axis=1)
test_data = test_data.drop('asin',axis=1)
#重要参数IV_threshold：IV最小为0, IV越小分析的变量越多.0默认所有feature
IV_threshold = 0.00
print("==================================评分卡决策树混合风控模型==============================================")
print("=============================Designed by AIFT,all reserved!========================================")
print("===================================================================================================")
#重要参数discrete_var与continuous_var：自动识别离散特征和连续特征,如果识别有误差,请手动设置
discrete_var = []
continuous_var = ['review_increase', 'sales_3', 'sales_4', 'sales_5', 'sales_6', 'sales_7', 'sales_8', 'sales_9', 'sales_10', 'sales_11', 'sales_12','rank', 'review', 'price_1', 'price_2', 'price_3', 'price_4', 'price_5', 'price_6', 'price_7', 'price_8', 'price_9', 'price_10', 'price_11', 'price_12', 'sales_1', 'sales_2']
if len(discrete_var) == 0 and len(continuous_var) == 0:
    for i in train_data.columns:
        if i == label:
            continue
        if len(set(train_data[i]))/len(train_data) < 0.10 and train_data[i].sum() == np.round(train_data[i].sum()):
            discrete_var.append(i)
        else:
            continuous_var.append(i)
        print("自动识别离散特征和连续特征,如果识别有误差,请手动设置discrete_var和continuous_var")
else:
    print("手动设置离散特征和连续特征")
print(f'Continuous Var:{continuous_var}')
print(f'Discrete Var:{discrete_var}')
print("===================================================================================================")
badnum = train_data[label].sum()
goodnum = train_data[label].count()-train_data[label].sum()
print('训练集数据中，好客户数量为：%i,坏客户数量为：%i,坏客户所占比例为：%.2f%%' %(goodnum,badnum,(badnum/train_data[label].count())*100))
print("======================================开始训练评分卡模型==============================================")
ninf = float('-inf')  # 负无穷大
pinf = float('inf')  # 正无穷大
def auto_bin(discrete_var):
    cutx_list = {}
    for feature in discrete_var:
        lis = list(set(train_data[feature]))
        lis.sort()
        N = len(lis)
        if N <= 2:
            cutx_list[feature] = cutx_list.get(feature,[]) + [ninf, np.mean(list(lis)), pinf]
        elif 3 <= N <= 10:
            cutx_list[feature] = cutx_list.get(feature,[]) + [ninf] + [(lis[i] + lis[i+1])/2 for i in range(N-1)] + [pinf]
        elif N < 100:
            cutx_list[feature] = cutx_list.get(feature,[]) + [ninf] + list(np.arange(min(lis), max(lis), round((max(lis) - min(lis)) / 10))) + [pinf]
        else:
            cutx_list[feature] = cutx_list.get(feature,[]) + [ninf] + list(np.arange(min(lis), max(lis), round((max(lis) - min(lis)) / 20))) + [pinf]
    return cutx_list

cutx_list = auto_bin(discrete_var)
# 自动分箱函数
def mono_bin(Y, X, n=10):
    r = 0
    good=Y.sum()
    bad=Y.count()-good
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n,duplicates='drop')})
        d2 = d1.groupby('Bucket', as_index = True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    d3 = pd.DataFrame(d2.X.min(), columns = ['min'])
    d3['min']=d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe']=np.clip(np.log((d3['rate']/(1-d3['rate']))/(good/bad)),-3,+3)
    d3['goodattribute']=d3['sum']/good
    d3['badattribute']=(d3['total']-d3['sum'])/bad
    iv=((d3['goodattribute']-d3['badattribute'])*d3['woe']).sum()
    d4 = (d3.sort_values(by = 'min')).reset_index(drop=True)
    woe=list(d4['woe'].round(3))
    cut=[]
    cut.append(float('-inf'))
    for i in range(1,n+1):
         qua=X.quantile(i/(n+1))
         cut.append(round(qua,4))
    cut.append(float('inf'))
    return d4,iv,cut,woe


def self_bin(Y,X,cut):
    badnum = np.array(Y.sum())
    goodnum = np.array(Y.count()-badnum)
    d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.cut(X, cut)})
    d2 = d1.groupby('Bucket', as_index = True)
    d3 = pd.DataFrame(d2.X.min(), columns = ['min'])
    d3['min']=d2.min().X
    d3['max'] = d2.max().X
    d3['bad'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    #woe过高时，重置为2到4之间的数字
    d3['woe'] = np.clip(np.log((d3['bad']/badnum)/((d3['total'] - d3['bad'])/goodnum)),-3,+3)
    d3['badattr'] = d3['bad']/badnum
    d3['goodattr'] = (d3['total'] - d3['bad'])/goodnum
    iv = ((d3['badattr']-d3['goodattr'])*d3['woe']).sum()
    d4 = (d3.sort_values(by = 'min')).reset_index(drop=True)
    woe=list(d4['woe'].round(3))
    return d4,iv,woe

ivlist = []
index = []
feature_map = {}
for i in range(len(train_data.columns)):
    feature_map[i] = feature_map.get(i, "") + train_data.columns[i]
    if train_data.columns[i] == label:
        continue
    elif train_data.columns[i] in continuous_var:
        exec("x"+str(i)+"_d, x"+str(i)+"_iv, x"+str(i)+"_cut, x"+str(i)+"_woe = mono_bin(train_data[label],train_data[train_data.columns[i]])")
        ivlist.append(eval("x" + str(i) + "_iv"))
        index.append("x" + str(i))
    elif train_data.columns[i] in discrete_var:
        cutx = cutx_list[train_data.columns[i]]
        exec("dfx"+str(i)+", ivx"+str(i)+", woex"+str(i)+" = self_bin(train_data[label], train_data[train_data.columns[i]],cutx)")
        ivlist.append(eval("ivx" + str(i)))
        index.append("x" + str(i))
    else:
        index.append("x" + str(i))
        ivlist.append(0)

fig1 = plt.figure(1,figsize=(8,5))
ax1 = fig1.add_subplot(1, 1, 1)
x = np.arange(len(index))+1
ax1.bar(x,ivlist,width=.4)
ax1.set_xticks(x)
ax1.set_xticklabels(index, rotation=0, fontsize=15)
ax1.set_ylabel('IV', fontsize=16)   #IV(Information Value),
for a, b in zip(x, ivlist):
    plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=12)


#替换成woe函数
def trans_woe(var,var_name,woe,cut):
    woe_name=var_name+'_woe'
    for i in range(len(woe)):
        if i==0:
            var.loc[(var[var_name]<=cut[i+1]),woe_name]=woe[i]
        elif (i>0) and  (i<=len(woe)-2):
            var.loc[((var[var_name]>cut[i])&(var[var_name]<=cut[i+1])),woe_name]=woe[i]
        else:
            var.loc[(var[var_name]>cut[len(woe)-1]),woe_name]=woe[len(woe)-1]
    return var


valid_feature = [i + 1 for i in range(len(ivlist)) if ivlist[i] > IV_threshold]
invalid_feature = [i + 1 for i in range(len(ivlist)) if ivlist[i] < IV_threshold]
valid_name = [feature_map[i] for i in valid_feature]
invalid_name = [feature_map[i] for i in invalid_feature]

for i in valid_feature:
    exec("x" + str(i) + "_name=feature_map[i]")
    if feature_map[i] == label:
        continue
    elif feature_map[i] in continuous_var:
        exec("train_data=trans_woe(train_data,x"+str(i)+"_name,x"+str(i)+"_woe,x"+str(i)+"_cut)")
    elif feature_map[i] in discrete_var:
        cutx = cutx_list[feature_map[i]]
        exec("train_data=trans_woe(train_data,x"+str(i)+"_name,woex"+str(i)+",cutx)")

def interpolation(data):
    data.isna().sum()
    data.fillna(0,inplace=True)
    data = data.replace([np.inf, -np.inf], 0)
    return data
train_data = interpolation(train_data)

y = train_data[label]
X = train_data.drop(invalid_name +[label],axis=1)[[i + "_woe" for i in valid_name]]
X.head()
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

import statsmodels.api as sm
train_X = sm.add_constant(train_X)
logit = sm.Logit(train_y,train_X)
result = logit.fit_regularized()
print(result.summary())

from sklearn import metrics
test_X = sm.add_constant(test_X)
resu = result.predict(test_X)
fpr,tpr,threshold=metrics.roc_curve(test_y,resu)
rocauc=metrics.auc(fpr,tpr)

plt.figure(figsize=(8,5))
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'% rocauc)
plt.legend(loc='lower right',fontsize=14)
plt.plot([0.0, 1.0], [0.0, 1.0], 'r--')
plt.xlim=([0.0, 1.0])
plt.ylim=([0.0, 1.0])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('TPR-True Positive',fontsize=16)
plt.xlabel('FPR-False Positive',fontsize=16)

p = 10/np.log(2)
q = 100 + 10*np.log(0.2)/np.log(2)

def coe_transfer(x_coe,valid_feature,feature_map,label,continuous_var,discrete_var):
    tmp = (max(valid_feature) + 1)*[0]
    j = 1
    for i in valid_feature:
        if feature_map[i] == label:
            continue
        elif feature_map[i] in continuous_var or discrete_var:
            tmp[i] = x_coe[j]
            j += 1
    tmp[0] = x_coe[0]
    x_coe = tmp
    return x_coe

x_coe = list(result.params)
x_coe = coe_transfer(x_coe,valid_feature,feature_map,label,continuous_var,discrete_var)

baseScore = round(q-p*x_coe[0],0)

def get_score(coe,woe,factor):
    # scores=[]
    coe = abs(coe)
    woe = [x for x in woe if np.isnan(x) == False]
    if ninf in woe:
        woe.remove(ninf)
    elif pinf in woe:
        woe.remove(pinf)
    scores = [round(coe*i*factor) for i in woe]
    return scores


for i in valid_feature:
    if feature_map[i] == label:
        continue
    elif feature_map[i] in continuous_var:
        exec("x"+str(i)+"_score=get_score(x_coe["+str(i)+"],x"+str(i)+"_woe,p)")
    elif feature_map[i] in discrete_var:
        exec("x"+str(i)+"_score=get_score(x_coe["+str(i)+"],woex"+str(i)+",p)")

def compute_score(series,cut,score):
    list = []
    i = 0
    while i < len(series):
        value = series.iloc[i]
        j = len(cut) - 2
        m = len(cut) - 2
        while j >= 0:
            if value >= cut[j]:
                j = -1
            else:
                j -= 1
                m -= 1
        list.append(score[min(m,len(score)-1)])
        i += 1
    return list

train_data['BaseScore']=np.zeros(len(train_data))+baseScore
for i in valid_feature:
    if feature_map[i] == label:
        continue
    elif feature_map[i] in continuous_var:
        exec("train_data['x" + str(i) + "'] = compute_score(train_data['" + feature_map[i] + "'], x" + str(i) + "_cut, x" + str(i) + "_score)")
    elif feature_map[i] in discrete_var:
        cutx = cutx_list[feature_map[i]]
        exec("train_data['x" + str(i) + "'] = compute_score(train_data['" + feature_map[i] + "'], cutx, x" + str(i) + "_score)")

# train_data = interpolation(train_data)

equation = "train_data['Score'] = baseScore"
for i in valid_feature:
    equation = equation + "-train_data['x"+str(i)+"']"
exec(equation)

# train_data = interpolation(train_data)

choose_list = ["x"+str(i) for i in valid_feature] + ["Score",label]

scoretable1 = train_data[[label,'BaseScore'] + ["x"+str(i) for i in valid_feature] + ["Score"]]
scoretable1.head()

colNameDict = {}
for i in valid_feature:
    colNameDict["x"+str(i)] = feature_map[i]
scoretable1 = scoretable1.rename(columns=colNameDict,inplace=False)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
train_data = train_data.drop(invalid_name,axis=1)

training,testing = train_test_split(train_data,test_size=0.25,random_state=1)

x_train=training[["x"+str(i) for i in valid_feature] + ["Score"]]
y_train=training[label]
x_test=testing[["x"+str(i) for i in valid_feature] + ["Score"]]
y_test=testing[label]
clf = LogisticRegression()
clf.fit(x_train,y_train)
#对测试集做预测
score_proba = clf.predict_proba(x_test)
y_predproba = score_proba[:,1]
coe = clf.coef_

#对模型做评估
from sklearn.metrics import roc_curve,auc
fpr,tpr,threshold = roc_curve(y_test,y_predproba)
auc_score = auc(fpr,tpr)
plt.figure(figsize=(8,5))  #只能在这里面设置
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'% auc_score)
plt.legend(loc='lower right',fontsize=14)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim=([0, 1])
plt.ylim=([0, 1])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('TPR-True Positive',fontsize=16)
plt.xlabel('FPR-False Positive',fontsize=16)


fig,ax = plt.subplots()
ax.plot(1-threshold,tpr,label='tpr')
ax.plot(1-threshold,fpr,label='fpr')
ax.plot(1-threshold,tpr-fpr,label='KS')
plt.xlabel('score')
plt.title('KS curve')
plt.xlim = ([0.0,1.0])
plt.ylim = ([0.0,1.0])
legend = ax.legend(loc='upper left')



max(tpr-fpr)

for i in valid_feature:
    exec("x" + str(i) + "_name=feature_map[i]")
    if feature_map[i] == label:
        continue
    elif feature_map[i] in continuous_var:
        exec("test_data=trans_woe(test_data,x" + str(i) + "_name,x" + str(i) + "_woe,x" + str(i) + "_cut)")
    elif feature_map[i] in discrete_var:
        cutx = cutx_list[feature_map[i]]
        exec("test_data=trans_woe(test_data,x" + str(i) + "_name,woex" + str(i) + ",cutx)")

test_data = interpolation(test_data)

test_data = test_data.drop(invalid_name,axis=1)
X_test = test_data[[i+"_woe" for i in valid_name]]
Y_test = test_data[label]   #因变量
X_train = training.copy()[["x"+str(i) for i in valid_feature]]
Y_train = training.copy()[label]
clf = LogisticRegression()
clf.fit(X_train,Y_train)
#对测试集做预测
score_proba = clf.predict_proba(X_test)

# test_data['y_predproba']=score_proba[:,1]

coe = clf.coef_

x_coe = list(result.params)
x_coe = coe_transfer(x_coe,valid_feature,feature_map,label,continuous_var,discrete_var)

for i in valid_feature:
    if feature_map[i] == label:
        continue
    elif feature_map[i] in continuous_var:
        exec("x"+str(i)+"_score=get_score(x_coe["+str(i)+"],x"+str(i)+"_woe,p)")
    elif feature_map[i] in discrete_var:
        exec("x"+str(i)+"_score=get_score(x_coe["+str(i)+"],woex"+str(i)+",p)")

test_data['BaseScore']=np.zeros(len(test_data))+baseScore

for i in valid_feature:
    if feature_map[i] == label:
        continue
    elif feature_map[i] in continuous_var:
        exec("test_data['x" + str(i) + "'] = compute_score(test_data['" + feature_map[i] + "'], x" + str(i) + "_cut, x" + str(i) + "_score)")
    elif feature_map[i] in discrete_var:
        cutx = cutx_list[feature_map[i]]
        exec("test_data['x" + str(i) + "'] = compute_score(test_data['" + feature_map[i] + "'], cutx, x" + str(i) + "_score)")

# test_data = interpolation(test_data)

equation1 = "test_data['Score'] = baseScore"
for i in valid_feature:
    equation1 = equation1 + "-test_data['x" + str(i) + "']"
exec(equation1)

scoretable2 = test_data[[label,'BaseScore'] + ["x"+str(i) for i in valid_feature] + ["Score"]]  #选取需要的列，就是评分列
# scoretable2 = test_data[[label,'y_predproba','BaseScore'] + ["x"+str(i) for i in valid_feature] + ["Score"]]  #选取需要的列，就是评分列
print("============================================Score Table============================================")
print(scoretable2.head())
print("===================================================================================================")
print("开始训练决策树模型,多重交叉验证")
scoretable2 = scoretable2.rename(columns=colNameDict,inplace=False)


plt.figure()
cut = [ninf] + list(np.arange(score_range[0], score_range[1], 5)) + [pinf]
d1 = pd.DataFrame({"Score": scoretable1["Score"], "Y": scoretable1[label], "Bucket": pd.cut(scoretable1["Score"], cut)})
d2 = d1.groupby('Bucket', as_index=True)

plt.bar([score_range[0] - 5] + list(np.arange(score_range[0], score_range[1], 5)), list(d2.count().Y/max(d2.count().Y)),width=3,label='Number of Users')
plt.plot([score_range[0] - 5] + list(np.arange(score_range[0], score_range[1], 5)), list(d2.sum().Y/d2.count().Y),color = 'red',label='Bad Users Rate')
# plt.xticks([0,1], ['Accuracy', 'AUC'])
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
# plt.xlim(-0.5,1.5)
# plt.ylim(0,1)
plt.title("Score Scard")
plt.legend()
# print(d2.sum().Y)





from sklearn.metrics import mean_absolute_error
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
train_data = data.iloc[:int(len(data)*training_size),:]
test_data = data.iloc[int(len(data)*training_size):,:]
test_data.reset_index(drop=True,inplace=True)
X_data = np.array(train_data[continuous_var+discrete_var])
Y_data = np.array(train_data[label])
X_test = np.array(test_data[continuous_var+discrete_var])
Y_test = np.array(test_data[label])
kfolder = KFold(n_splits=2, shuffle=True, random_state=2018)
oof_cb = np.zeros(len(X_data))
predictions_cb = np.zeros(len(X_test))
predictions_train_cb = np.zeros(len(X_data))
kfold = kfolder.split(X_data, Y_data)
fold_ = 0
for train_index, vali_index in kfold:
    fold_ = fold_ + 1
    print("fold n°{}".format(fold_))
    k_x_train = X_data[train_index]
    k_y_train = Y_data[train_index]
    k_x_vali = X_data[vali_index]
    k_y_vali = Y_data[vali_index]
    model_cb = CatBoostClassifier()
    # train the model
    model_cb.fit(k_x_train, k_y_train, eval_set=[(k_x_vali, k_y_vali)], verbose=300, early_stopping_rounds=600)
    oof_cb[vali_index] = model_cb.predict(k_x_vali, ntree_end=model_cb.best_iteration_)
    predictions_cb += model_cb.predict(X_test, ntree_end=model_cb.best_iteration_) / kfolder.n_splits
    predictions_train_cb += model_cb.predict(X_data, ntree_end=model_cb.best_iteration_) / kfolder.n_splits


Y_test = np.array(Y_test)
Y_pred = np.array(model_cb.predict(X_test, ntree_end=model_cb.best_iteration_))
print(f'测试集准确率: {sum([1 if i == j else 0 for i,j in zip(Y_test,Y_pred)])/len(Y_pred)*100:.2f}%')
scoretable2["pred"] = Y_pred
scoretable2.to_csv('.\测试集评分卡.csv')
Y_pred = np.array(model_cb.predict(X_data, ntree_end=model_cb.best_iteration_))
scoretable1["pred"] = Y_pred
scoretable1.to_csv('.\训练集评分卡.csv')
# print('The mae of prediction is:', mean_absolute_error(Y_test, Y_pred))
print("评分卡与决策树训练完成！请查看csv文件")
plt.show()


