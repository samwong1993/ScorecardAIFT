import numpy as np
import pandas as pd
import random
from random import sample

def generator_discrete(num,mean,std,pctl25,median,pctl75):
    def sub(mid,low,up,size):
        res = []
        while len(res) < size:
            tmp = np.round(np.random.normal(loc=mid, scale=(up - mid),
                                   size=size - len([i for i in res if low <= i <= up])))
            tmp1 = [i for i in tmp if low <= i <= up]
            res = res + tmp1
        return res
    factor = 1
    mean_lis = 4*[0]
    mean_lis[0] = 0.5 * pctl25
    mean_lis[1] = pctl25 + 0.5 * (median - pctl25)
    mean_lis[2] = median + 0.5 * (pctl75 - median)
    mean_lis[3] = 4 * mean - (mean_lis[0] + mean_lis[1] + mean_lis[2])
    idx = [i for i in range(len(mean_lis)) if mean_lis[i] > mean][0]
    N = [int(num/4)]*4
    res_0_25 = sub(mean_lis[0],0,pctl25,N[0])
    # print(mean_lis[0],np.mean(res_0_25))
    res_25_50 = sub(mean_lis[1],pctl25,median,N[1])
    # print(mean_lis[1], np.mean(res_25_50))
    res_50_75 = sub(mean_lis[2],median,pctl75,N[2])
    # print(mean_lis[2], np.mean(res_50_75))
    res_75_100 = sub(mean_lis[3],pctl75,factor*max(2 * mean_lis[3] - pctl75,2 *pctl75 - mean_lis[3]),N[3])
    # print(mean_lis[3], np.mean(res_75_100))
    res = res_0_25 + res_25_50 + res_50_75 + res_75_100
    # print("mean: ",np.mean(res),"std: ",np.std(res))
    tune = int(num/100)
    while True:
        if np.std(res) < std and np.mean(res) < mean:
            factor = 1.01 * factor
            for i in range(len(N)):
                if i == 0 or i == 3:
                    N[i] = N[i] + tune
                else:
                    N[i] = N[i] - tune
        elif np.std(res) < std and np.mean(res) > mean:
            for i in range(len(N)):
                if i < idx:
                    N[i] = N[i] + tune
                else:
                    N[i] = N[i] - tune
        elif np.std(res) > std and np.mean(res) < mean:
            for i in range(len(N)):
                if i < idx:
                    N[i] = N[i] - tune
                else:
                    N[i] = N[i] + tune
        elif np.std(res) > std and np.mean(res) > mean:
            factor = 0.99 * factor
            for i in range(len(N)):
                if i == 0 or i == 3:
                    N[i] = N[i] - tune
                else:
                    N[i] = N[i] + tune
        res_0_25 = sub(mean_lis[0], 0, pctl25, N[0])
        # print(mean_lis[0], np.mean(res_0_25))
        res_25_50 = sub(mean_lis[1], pctl25, median, N[1])
        # print(mean_lis[1], np.mean(res_25_50))
        res_50_75 = sub(mean_lis[2], median, pctl75, N[2])
        # print(mean_lis[2], np.mean(res_50_75))
        res_75_100 = sub(mean_lis[3], pctl75, factor*max(2 * mean_lis[3] - pctl75,2 *pctl75 - mean_lis[3]), N[3])
        # print(mean_lis[3], np.mean(res_75_100))
        res = res_0_25 + res_25_50 + res_50_75 + res_75_100
        print("样本分布", N)
        print("True parameters: mean: %.2f" % mean, "std: %.2f" % std)
        print("Simulated para:  mean: %.2f" % np.mean(res), "std: %.2f" % np.std(res))
        if abs(np.std(res) - std)/std < 0.02 and abs(np.mean(res) - mean)/mean < 0.02:
            break
    print("样本分布",N)
    print("True parameters: mean: %.2f" % mean, "std: %.2f" % std)
    print("Simulated para:  mean: %.2f" % np.mean(res), "std: %.2f" % np.std(res))
    res = np.array(res)
    index = [i for i in range(len(res))]
    random.shuffle(index)
    res = list(res[index])
    if len(res) < num:
        res = res + res
    res = sample(res, num)
    return res





def generator_binary(num,mean):
    sum = int(num * mean)
    res = pd.DataFrame([0 for _ in range(num)])
    res.iloc[0:sum] = 1
    print("mean: ",float(res.mean())," std: ",float(res.std())," pctl25",float(np.percentile(res, 25))," median",float(np.percentile(res, 50))," pctl75",float(np.percentile(res, 75)))
    res = np.array(res[0])
    index = [i for i in range(len(res))]
    random.shuffle(index)
    res = res[index]
    return res





def generator_continuous(num,mean,std,pctl25,median,pctl75):
    def sub(mid,low,up,size):
        res = []
        while len(res) < size:
            tmp = np.random.normal(loc=mid, scale=(up - mid),
                                   size=size - len([i for i in res if low <= i <= up]))
            tmp1 = [i for i in tmp if low <= i <= up]
            res = res + tmp1
        return res
    factor = 1
    mean_lis = 4 * [0]
    mean_lis[0] = 0.5 * pctl25
    mean_lis[1] = pctl25 + 0.5 * (median - pctl25)
    mean_lis[2] = median + 0.5 * (pctl75 - median)
    mean_lis[3] = 4 * mean - (mean_lis[0] + mean_lis[1] + mean_lis[2])
    idx = [i for i in range(len(mean_lis)) if mean_lis[i] > mean][0]
    N = [int(num/4)]*4
    res_0_25 = sub(mean_lis[0],0,pctl25,N[0])
    # print(mean_lis[0],np.mean(res_0_25))
    res_25_50 = sub(mean_lis[1],pctl25,median,N[1])
    # print(mean_lis[1], np.mean(res_25_50))
    res_50_75 = sub(mean_lis[2],median,pctl75,N[2])
    # print(mean_lis[2], np.mean(res_50_75))
    res_75_100 = sub(mean_lis[3],pctl75,factor*max(2 * mean_lis[3] - pctl75,2 *pctl75 - mean_lis[3]),N[3])
    # print(mean_lis[3], np.mean(res_75_100))
    res = res_0_25 + res_25_50 + res_50_75 + res_75_100
    # print("mean: ",np.mean(res),"std: ",np.std(res))
    tune = int(num/100)
    while True:
        if np.std(res) < std and np.mean(res) < mean:
            factor = 1.01 * factor
            for i in range(len(N)):
                if i == 0 or i == 3:
                    N[i] = N[i] + tune
                else:
                    N[i] = N[i] - tune
        elif np.std(res) < std and np.mean(res) > mean:
            for i in range(len(N)):
                if i < idx:
                    N[i] = N[i] + tune
                else:
                    N[i] = N[i] - tune
        elif np.std(res) > std and np.mean(res) < mean:
            for i in range(len(N)):
                if i < idx:
                    N[i] = N[i] - tune
                else:
                    N[i] = N[i] + tune
        elif np.std(res) > std and np.mean(res) > mean:
            factor = 0.99 * factor
            for i in range(len(N)):
                if i == 0 or i == 3:
                    N[i] = N[i] - tune
                else:
                    N[i] = N[i] + tune
        res_0_25 = sub(mean_lis[0], 0, pctl25, N[0])
        # print(mean_lis[0], np.mean(res_0_25))
        res_25_50 = sub(mean_lis[1], pctl25, median, N[1])
        # print(mean_lis[1], np.mean(res_25_50))
        res_50_75 = sub(mean_lis[2], median, pctl75, N[2])
        # print(mean_lis[2], np.mean(res_50_75))
        res_75_100 = sub(mean_lis[3], pctl75, factor*max(2 * mean_lis[3] - pctl75,2 *pctl75 - mean_lis[3]), N[3])
        # print(mean_lis[3], np.mean(res_75_100))
        res = res_0_25 + res_25_50 + res_50_75 + res_75_100
        print("样本分布", N)
        print("True parameters: mean: %.2f" % mean, "std: %.2f" % std)
        print("Simulated para:  mean: %.2f" % np.mean(res), "std: %.2f" % np.std(res))
        if abs(np.std(res) - std)/std < 0.02 and abs(np.mean(res) - mean)/mean < 0.02:
            break
    print("样本分布",N)
    print("True parameters: mean: %.2f" % mean, "std: %.2f" % std)
    print("Simulated para:  mean: %.2f" % np.mean(res), "std: %.2f" % np.std(res))
    res = np.array(res)
    index = [i for i in range(len(res))]
    random.shuffle(index)
    res = list(res[index])
    res = sample(res, num)
    return res




# from numpy import random
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.distplot(np.random.normal(loc=mean_1, scale=(pctl25 - mean_1), size=1000), hist=False)




num = 50000
data = pd.DataFrame(columns = ["House property","Car property","Number of credit cards","Owner's age","Gender","Firm's age","City tiers","Total inflow of funds in Alipay","Shop rating","Transaction volume","Log-ins","Security funds in Taobao","VIP class","Number of good feedtbacks from clients","Network effect score","Link to credit card","Daily average Yu'eBao balance","Daily average Alipay Wallet balarce","Daily payment activity","Payment activity","Tatal amount of ecommerce purchases","Quarterly consumption of goods","Number of fulfilled contracts in daily life","Stability of contact information","Duration in one location"])

mean = 0.621
std = 0.485
pctl25 = 0
median = 1
pctl75 = 1
data['House property'] = generator_binary(num,mean)

mean = 0.662
std = 0.473
pctl25 = 0
median = 1
pctl75 = 1
data['Car property'] = generator_binary(num,mean)

mean = 2.968
std = 3.838
pctl25 = 0
median = 2
pctl75 = 4
data['Number of credit cards'] = generator_discrete(num,mean,std,pctl25,median,pctl75)


mean = 29
std = 6
pctl25 = 25
median = 28
pctl75 = 31
data["Owner's age"] = generator_discrete(num,mean,std,pctl25,median,pctl75)

mean = 0.65
std = 0.48
pctl25 = 0
median = 1
pctl75 = 1
data["Gender"] = generator_binary(num,mean)


mean = 4.48
std = 2.24
pctl25 = 2.7
median = 4.07
pctl75 = 5.99
data["Firm's age"] = generator_discrete(num,mean,std,pctl25,median,pctl75)

mean = 2.468
std = 1.027
pctl25 = 2
median = 2
pctl75 = 3
data['City tiers'] = generator_discrete(num,mean,std,pctl25,median,pctl75)



mean = 516708.00
std = 1149718.00
pctl25 = 138534.10
median = 264210.00
pctl75 = 534783.60
data['Total inflow of funds in Alipay'] = generator_continuous(num,mean,std,pctl25,median,pctl75)



mean = 4.19
std = 2.96
pctl25 = 2
median = 4
pctl75 = 6
data['Shop rating'] = generator_discrete(num,mean,std,pctl25,median,pctl75)


mean = 9196.80
std = 58973.99
pctl25 = 1964.35
median = 4754.43
pctl75 = 9954.34
data['Transaction volume'] = generator_continuous(num,mean,std,pctl25,median,pctl75)


mean = 68.06
std = 42.54
pctl25 = 33
median = 59
pctl75 = 99
data['Log-ins'] = generator_discrete(num,mean,std,pctl25,median,pctl75)


mean = 312.78
std = 827.78
pctl25 = 0
median = 0
pctl75 = 1000
data['Security funds in Taobao'] = generator_discrete(num,mean,std,pctl25,median,pctl75)



mean = 2.94
std = 1.28
pctl25 = 2
median = 3
pctl75 = 4
data['VIP class'] = generator_discrete(num,mean,std,pctl25,median,pctl75)


mean = 950.94
std = 8002.71
pctl25 = 23
median = 118
pctl75 = 460
data['Number of good feedtbacks from clients'] = generator_discrete(num,mean,std,pctl25,median,pctl75)


mean = 64.48
std = 27.66
pctl25 = 48.85
median = 59.95
pctl75 = 74.09
data['Network effect score'] = generator_continuous(num,mean,std,pctl25,median,pctl75)


mean = 0.56
std = 0.5
pctl25 = 0
median = 1
pctl75 = 1
data['Link to credit card'] = generator_binary(num,mean)




mean = 974.84
std = 6656.89
pctl25 = 0
median = 106
pctl75 = 532.6
data["Daily average Yu'eBao balance"] = generator_continuous(num,mean,std,pctl25,median,pctl75)


mean = 871.30
std = 2487.46
pctl25 = 95.92
median = 362.18
pctl75 = 1117.84
data["Daily average Alipay Wallet balarce"] = generator_continuous(num,mean,std,pctl25,median,pctl75)

mean = 1511.85
std = 3682.70
pctl25 = 647
median = 1007.00
pctl75 = 1587.00
data["Daily payment activity"] = generator_discrete(num,mean,std,pctl25,median,pctl75)



mean = 204.72
std = 623.59
pctl25 = 81
median = 140
pctl75 = 236
data["Payment activity"] = generator_discrete(num,mean,std,pctl25,median,pctl75)




mean = 33514.44
std = 128465.10
pctl25 = 9132.47
median = 18080.32
pctl75 = 34788.37
data["Total amount of ecommerce purchases"] = generator_continuous(num,mean,std,pctl25,median,pctl75)



mean = 838.73
std = 7754.11
pctl25 = 0
median = 57.70
pctl75 = 323
data["Quarterly consumption of goods"] = generator_continuous(num,mean,std,pctl25,median,pctl75)


mean = 1.07
std = 0.63
pctl25 = 1
median = 1
pctl75 = 1
data['Number of fulfilled contracts in daily life'] = generator_discrete(num,mean,std,pctl25,median,pctl75)


mean = 1.87
std = 1.35
pctl25 = 1
median = 2
pctl75 = 2
data['Stability of contact information'] = generator_discrete(num,mean,std,pctl25,median,pctl75)


mean = 1815.63
std = 762.28
pctl25 = 1261.00
median = 1714.00
pctl75 = 2318.00
data["Duration in one location"] = generator_continuous(num,mean,std,pctl25,median,pctl75)

data.to_csv("./data.csv",index=None)
# ['House property', 'Car property', 'Number of credit cards',
#        "Owner's age", 'Gender', "Firm's age", 'City tiers',
#        'Total inflow of funds in Alipay', 'Shop rating', 'Transaction volume',
#        'Log-ins', 'Security funds in Taobao', 'VIP class',
#        'Number of good feedbacks from clients', 'Network effect score',
#        'Link to credit card', "Daily average Yu'eBao balance",
#        'Daily average Alipay Wallet balance', 'Daily payment activity',
#        'Payment activity', 'Total amount of ecommerce purchases',
#        'Quarterly consumption of goods',
#        'Number of fulfilled contracts in daily life',
#        'Stability of contact information', 'Duration in one location']
