import pandas as pd;
import numpy  as np;
import math;
from scipy import signal;
import matplotlib.pyplot as plt;
from sklearn.linear_model import LinearRegression 
import statsmodels.api as sm
import warnings
import pywt
pd.options.mode.chained_assignment = None;
    
#读取数据文件
#用户数据:id 性别 城市 星座 
user_profile_table=pd.read_csv(r"./user_profile_table.csv",sep=',',engine='python',encoding='utf-8');
#用户申购赎回数数据
user_balance_table=pd.read_csv(r"./user_balance_table.csv",sep=',',engine='python',encoding='utf-8',parse_dates = ['report_date']);
#收益率：日期 万分收益 七日年化收益
mfd_day_share_interest=pd.read_csv(r"./mfd_day_share_interest.csv",sep=',',engine='python',encoding='utf-8',parse_dates = ['mfd_date']);
#银行拆放利率:日期 隔夜利率（%）1周利率（%）2周利率（%）1个月利率（%）3个月利率（%）6个月利率（%）9个月利率（%）
mfd_bank_shibor=pd.read_csv(r"./mfd_bank_shibor.csv",sep=',',engine='python',encoding='utf-8',parse_dates = ['mfd_date']);



#user_balance_table 数据缺失值处理 填充0(众数)
user_balance_table=user_balance_table.fillna(0);



#购买赎回总量计算
user_balance = user_balance_table.groupby(['report_date']);
purchase_redeem_total = user_balance['total_purchase_amt', 'total_redeem_amt'].sum();


#计算星期列
date=pd.DataFrame(purchase_redeem_total.index);
#date['day_of_week']=date['report_date'].dt.dayofweek;
date['day_of_week']=date['report_date'].dt.weekday_name;
tt_date = date.groupby(['report_date']);
tt_date = tt_date['day_of_week'].sum();
#转化为0 1 哑变量 并进行DF拼接
purchase_redeem_total_with_week_day=pd.concat([pd.get_dummies(tt_date,columns='day_of_week'),purchase_redeem_total],axis=1);

#收益
time_mfd_day_share_interest=mfd_day_share_interest.groupby(['mfd_date']);
share_interest=time_mfd_day_share_interest['mfd_daily_yield','mfd_7daily_yield'].sum();
#拆放利率
t_mfd_bank_shibor = (mfd_bank_shibor.groupby(['mfd_date']));
time_mfd_bank_shibor=t_mfd_bank_shibor['Interest_O_N'].sum();
time_mfd_bank_shibor=time_mfd_bank_shibor.fillna(method='pad');

prtwd=pd.concat([purchase_redeem_total_with_week_day,share_interest,time_mfd_bank_shibor,tt_date],axis=1);
prtwd=prtwd.fillna(method='pad');

#选出2014年 3月份到7月的数据 的数据
prtwd1403to1407=prtwd['2014-03':'2014-07']




prtwd1403to1406=prtwd1403to1407['2014-03':'2014-06'];
prtwd1404to1406=prtwd1403to1407['2014-04':'2014-06'];
prtwd1407=prtwd1403to1407['2014-07'];


index_list=np.array(prtwd1403to1406['total_purchase_amt'].astype('float'));
A2,D2,D1 = pywt.wavedec(index_list,'db4',mode='sym',level=2);

# 对每层小波系数求解模型系数
order_A2 = sm.tsa.arma_order_select_ic(A2,ic='aic')['aic_min_order']   # AIC准则求解模型阶数p,q
order_D2 = sm.tsa.arma_order_select_ic(D2,ic='aic')['aic_min_order']   # AIC准则求解模型阶数p,q
order_D1 = sm.tsa.arma_order_select_ic(D1,ic='aic')['aic_min_order']   # AIC准则求解模型阶数p,q

print(order_A2,order_D2,order_D1)   #各个模型阶次

# 对每层小波系数构建ARMA模型
model_A2 =  sm.tsa.ARMA(A2,order=order_A2)   # 建立模型
model_D2 =  sm.tsa.ARMA(D2,order=order_D2)
model_D1 =  sm.tsa.ARMA(D1,order=order_D1)

results_A2 = model_A2.fit()
results_D2 = model_D2.fit()
results_D1 = model_D1.fit()


# 画出每层的拟合曲线
plt.figure(figsize=(10,15))
plt.subplot(3, 1, 1)
plt.plot(A2, 'blue')
plt.plot(results_A2.fittedvalues,'red')
plt.title('purchase_model_A2')

plt.subplot(3, 1, 2)
plt.plot(D2, 'blue')
plt.plot(results_D2.fittedvalues,'red')
plt.title('purchase_model_D2')

plt.subplot(3, 1, 3)
plt.plot(D1, 'blue')
plt.plot(results_D1.fittedvalues,'red')
plt.title('purchase_model_D1')




A2_all,D2_all,D1_all = pywt.wavedec(np.array(prtwd1403to1407['total_purchase_amt'].astype('float')),'db4',mode='sym',level=2) # 对所有序列分解
delta = [len(A2_all)-len(A2),len(D2_all)-len(D2),len(D1_all)-len(D1)];
#delta = [8,8,16,16,32] # 求出差值，则delta序列对应的为每层小波系数ARMA模型需要预测的步数
# 预测小波系数 包括in-sample的和 out-sample的需要预测的小波系数
pA2 = model_A2.predict(params=results_A2.params,start=1,end=len(A2)+delta[0])
pD2 = model_D2.predict(params=results_D2.params,start=1,end=len(D2)+delta[1])
pD1 = model_D1.predict(params=results_D1.params,start=1,end=len(D1)+delta[2])
# 重构
coeff_new = [pA2,pD2,pD1]
denoised_index = pywt.waverec(coeff_new,'db4')

purchase_denoised_index=denoised_index;







index_list=np.array(prtwd1403to1406['total_redeem_amt'].astype('float'));
A2,D2,D1 = pywt.wavedec(index_list,'db4',mode='sym',level=2);

# 对每层小波系数求解模型系数
order_A2 = sm.tsa.arma_order_select_ic(A2,ic='aic')['aic_min_order']   # AIC准则求解模型阶数p,q
order_D2 = sm.tsa.arma_order_select_ic(D2,ic='aic')['aic_min_order']   # AIC准则求解模型阶数p,q
order_D1 = sm.tsa.arma_order_select_ic(D1,ic='aic')['aic_min_order']   # AIC准则求解模型阶数p,q

print(order_A2,order_D2,order_D1)   #各个模型阶次

# 对每层小波系数构建ARMA模型
model_A2 =  sm.tsa.ARMA(A2,order=order_A2)   # 建立模型
model_D2 =  sm.tsa.ARMA(D2,order=order_D2)
model_D1 =  sm.tsa.ARMA(D1,order=order_D1)

results_A2 = model_A2.fit()
results_D2 = model_D2.fit()
results_D1 = model_D1.fit()


# 画出每层的拟合曲线
plt.figure(figsize=(10,15))
plt.subplot(3, 1, 1)
plt.plot(A2, 'blue')
plt.plot(results_A2.fittedvalues,'red')
plt.title('redeem_model_A2')

plt.subplot(3, 1, 2)
plt.plot(D2, 'blue')
plt.plot(results_D2.fittedvalues,'red')
plt.title('redeem_model_D2')

plt.subplot(3, 1, 3)
plt.plot(D1, 'blue')
plt.plot(results_D1.fittedvalues,'red')
plt.title('redeem_model_D1')




A2_all,D2_all,D1_all = pywt.wavedec(np.array(prtwd1403to1407['total_redeem_amt'].astype('float')),'db4',mode='sym',level=2) # 对所有序列分解
delta = [len(A2_all)-len(A2),len(D2_all)-len(D2),len(D1_all)-len(D1)];
#delta = [8,8,16,16,32] # 求出差值，则delta序列对应的为每层小波系数ARMA模型需要预测的步数
# 预测小波系数 包括in-sample的和 out-sample的需要预测的小波系数
pA2 = model_A2.predict(params=results_A2.params,start=1,end=len(A2)+delta[0])
pD2 = model_D2.predict(params=results_D2.params,start=1,end=len(D2)+delta[1])
pD1 = model_D1.predict(params=results_D1.params,start=1,end=len(D1)+delta[2])
# 重构
coeff_new = [pA2,pD2,pD1]
denoised_index = pywt.waverec(coeff_new,'db4')


redeem_denoised_index=denoised_index;

# 画出重构后的原序列预测图
plt.figure()
plt.plot(np.array(prtwd1403to1407.ix['2014-07','total_purchase_amt'].astype('float64')), 'blue',label='test')
plt.plot(purchase_denoised_index[len(denoised_index)-31:len(denoised_index)],'red',label='predict')
plt.title('july\'s total_purchase_amt predict')
plt.legend();
plt.draw();

plt.figure()
plt.plot(np.array(prtwd1403to1407.ix['2014-07','total_redeem_amt'].astype('float64')), 'blue',label='test')
plt.plot(redeem_denoised_index[len(denoised_index)-31:len(denoised_index)],'red',label='predict')
plt.title('july\'s total_redeem_amt predict')
plt.legend();
plt.draw();


plt.show();

