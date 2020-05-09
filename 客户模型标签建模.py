import pandas as pd
import numpy as np
import cx_Oracle
import time
from sqlalchemy import create_engine
import os
import datetime
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import DBSCAN
os.environ['NLS_LANG']='SIMPLIFIED CHINESE_CHINA.UTF8'
engine = create_engine("oracle://HRICMS:123456@127.0.0.1:1521/ORCL",encoding="utf-8")
def custom():
    db = cx_Oracle.connect("HRICMS/123456@127.0.0.1:1521/ORCL") #连接数据库
    cursors = db.cursor()                                       #创建一个游标
    try:
        cursors.execute("drop table custom")
        cursors.execute("create table custom(cons_id int,"
                        "cons_name varchar(200),"
                        "客户级别 varchar(20),"
                        "客户类型 varchar(20),"
                        "行业 varchar(20),"
                        "用户安全等级 int,"
                        "保安负荷 int,"
                        "是否有合同 int,"
                        "合同签订日期 date,"
                        "是否有附件 int,"
                        "风险确认书是否合格 int,"
                        "合同类型 varchar(20),"
                        "用电分类 varchar(20),"
                        "负荷性质 varchar(20),"
                        "供电电压 varchar(20),"
                        "电源数目 varchar(20),供电方式 varchar(300),报装容量 varchar(20),是否制定应急预案 varchar(20),"
                        "安全运行管理制度 varchar(20),是否定期检查预试 varchar(20),自动化装置配合 int,非电性质保安措施 int,供电企业性质 varchar(20),运行容量 int,"
                        "自备电源数目 int,自备电源容量 int,自备电源闭锁情况 int,电工数量 int,电工过期 int,供电单位 varchar(200),档案完整率 int,"
                        "经度坐标 float,纬度坐标 float,证件数目 int,证件在期数目 int,证件类型 varchar(400),隐患数目 int,安全隐患类型 varchar(200),隐患完成时间 float)")
        cursors.execute("insert into custom select * from v_c_cons_infos")
        db.commit()
        db.close()
    except Exception as e:
        print(e)
        db.rollback()
def Data_cleaning():
    list=[]
    start=time.time()
    datas = pd.read_sql("select * from custom", engine)
    datas.fillna(value={"行业":datas["行业"].mode()[0]}, inplace=True)
    # datas.fillna(value={"客户状态": datas["客户状态"].mode()[0]}, inplace=True)
    datas.fillna(value={"供电方式": datas["供电方式"].mode()[0]}, inplace=True)
    datas.fillna(value={"合同类型": datas["合同类型"].mode()[0]}, inplace=True)
    datas.隐患完成时间.fillna(value=0, inplace=True)
    for x in datas.自备电源闭锁情况:
        pass
    datas.自备电源容量.fillna(value=0, inplace=True)
    fillna_Titanic=[]
    #按照行业类别进行填充的
    for i in datas.行业.unique():
        update = datas.loc[datas.行业 == i,].fillna(
            value={'客户类型': datas.客户类型[datas.行业 == i].mode()[0], "用户安全等级": datas.用户安全等级[datas.行业 == i].mode()[0],
                   "保安负荷": datas.保安负荷[datas.行业 == i].mode()[0],
                   "报装容量": datas.报装容量[datas.行业 == i].mode()[0],
                   "用电分类":datas.用电分类[datas.行业 == i].mode()[0],
                   "经度坐标": datas.经度坐标[datas.行业 == i].mean(),
                   "合同签订日期": datas.合同签订日期[datas.行业 == i].mode()[0],
                   "证件类型": datas.证件类型[datas.行业 == i].mode()[0],
                   "纬度坐标": datas.纬度坐标[datas.行业 == i].mean(),
                   "电源数目": datas.电源数目[datas.行业 == i].mode()[0],
                   "自备电源数目": datas.自备电源数目[datas.行业 == i].mode()[0],
                   "负荷性质": datas.负荷性质[datas.行业 == i].mode()[0],
                   "供电企业性质": datas.供电企业性质[datas.行业 == i].mode()[0],
                   "是否制定应急预案": datas.是否制定应急预案[datas.行业 == i].mode()[0],
                   "安全运行管理制度": datas.安全运行管理制度[datas.行业 == i].mode()[0],
                   "是否定期检查预试": datas.是否定期检查预试[datas.行业 == i].mode()[0],
                   "自动化装置配合": datas.自动化装置配合[datas.行业 == i].mode()[0],
                   "档案完整率": datas.档案完整率[datas.行业 == i].mode()[0],
                   "非电性质保安措施": datas.非电性质保安措施[datas.行业 == i].mode()[0], "运行容量": round(datas.运行容量[datas.行业 == i].mean()),
                   "电工数量": round(datas.电工数量[datas.行业 == i].mean()), "电工过期": round(datas.电工过期[datas.行业 == i].mean()),
                   "证件数目": round(datas.证件数目[datas.行业 == i].mean()), "证件在期数目": round(datas.证件在期数目[datas.行业 == i].mean()),
                    "供电电压": datas.供电电压[datas.行业 == i].mode()[0]},
            inplace=False)
        fillna_Titanic.append(update)
    data = pd.concat(fillna_Titanic)
    fillna_Titanic_mm = []
    # 按照电源数目分类进行填充供电方式
    for x in data.电源数目.unique():
        updates = data.loc[data.电源数目 == x,].fillna(value={"供电方式": data.供电方式[data.电源数目 == x].mode()[0]})
        fillna_Titanic_mm.append(updates)
    data_mm = pd.concat(fillna_Titanic_mm)
    # data_mm.drop_duplicates(inplace=True)          #删除重复数据
    data_mm = data_mm.reset_index()         #建立新的索引
    del data_mm["index"]
    train_data = np.array(data_mm)
    train_newdata_list = train_data.tolist()     # list
    return data_mm
# for x in Data_cleaning().cons_id:
#     print(x)
def Model_label_archives():        #档案完整率
    list=[]
    data=Data_cleaning()
    for x in data.档案完整率:
        if x==100.0:
            x='齐全'
        elif x>=90 and x<100:
            x='较高'
        else:
            x='较低'
        list.append(x)
    mn=pd.DataFrame(np.array(list),columns=['档案完整程度'])
    return mn
def Running_state():           #运行状态
    list = []
    data = Data_cleaning()
    # dty=data.dtypes
    data.报装容量=data.报装容量.astype('float')
    data.eval('new1=报装容量-运行容量', inplace=True)
    for y,x in enumerate(data.new1):
        if x==0 and data.报装容量[y]!=0:
            x='正常'
        elif x<0:
            x='超负荷运行'
        elif x>0 and (x/data.运行容量[y])>0.5:
            x='低负荷运行'
        elif (x/data.运行容量[y])<0.5:
            x='超低负荷'
        list.append(x)
    mn = pd.DataFrame(np.array(list), columns=['运行状态'])
    return mn
def Customer_photo():   #客户证照
    list = []
    data = Data_cleaning()
    data.eval('new2=证件数目-证件在期数目', inplace=True)
    for y,x in enumerate(data.new2):
        if x==0 and data.证件数目[y]!=0:
            x='齐全'
        elif data.证件数目[y]==0 and data.证件在期数目[y]==0:
            x='缺失'
        elif x>0:
            x='过期'
        list.append(x)
    mn = pd.DataFrame(np.array(list), columns=['客户证照'])
    return mn
def power_qualified():#电源是否合格
    list=[]
    data = Data_cleaning()
    data.eval('new11=证件数目-证件在期数目', inplace=True)
    for x,y in enumerate(data.new11):
        if (data.客户级别[x]=='特级重要用户' and data.电源数目[x]=='双电源') or (data.客户级别[x]=='特级重要用户' and data.电源数目[x]=='多电源') or (data.客户级别[x]=='一级重要用户' and data.电源数目[x]=='双电源') or (data.客户级别[x]=='二级重要用户' and data.电源数目[x]=='双电源') or (data.客户级别[x]=='特级重要用户' and data.电源数目[x]=='双双回路'):
            y='合格'
        else:
            y='不合格'
        list.append(y)
    mn = pd.DataFrame(np.array(list), columns=['电源是否合格'])
    return mn
def Public_contracts():    #公用合同
    list = []
    data = Data_cleaning()
    data.eval('new12=证件数目-证件在期数目', inplace=True)
    for x,y in enumerate(data.new12):
        if data.是否有合同[x]==1 and data.是否有附件[x]==1 and data.风险确认书是否合格[x]==1:
            y='合格'
        elif data.是否有合同[x]==1 and data.是否有附件[x]==1 and data.风险确认书是否合格[x]==0:
            y='缺少风险确认书'
        elif data.是否有合同[x]==1 and data.是否有附件[x]==0 and data.风险确认书是否合格[x]==1:
            y='缺少附件'
        elif data.是否有合同[x]==1 and data.是否有附件[x]==0 and data.风险确认书是否合格[x]==0:
            y='缺失风险确认及附件'
        elif data.是否有合同[x]==0 and data.是否有附件[x]==1 and data.风险确认书是否合格[x]==1:
            y='缺少合同档案'
        elif data.是否有合同[x] ==0 and data.是否有附件[x] == 0 and data.风险确认书是否合格[x] ==0:
            y='合同档案、附件、风险确认书全部缺失'
        list.append(y)
    mn = pd.DataFrame(np.array(list), columns=['公用合同'])
    return mn
def emergency_power_supply():      #自备应急电源
    list = []
    data = Data_cleaning()
    data.eval('new3=自备电源数目-自备电源闭锁情况', inplace=True)
    for y,x in enumerate(data.new3):
        if x==0:
            x='合规'
        elif data.自备电源数目[y]==0 and data.自备电源闭锁情况[y] == 0:
            x='无自备'
        elif data.自备电源数目[y]!=0 and data.自备电源闭锁情况[y]==0:
            x='无闭锁'
def protect_apply():   #保护措施
    list = []
    data = Data_cleaning()
    data.eval('new4=自动化装置配合-非电性质保安措施', inplace=True)
    for y,x in enumerate(data.new4):
        if x==0 and data.自动化装置配合[y]!=0:
            x='合规'
        elif x!=0:
            x='按需补缺'
        elif data.自动化装置配合[y]==0 and data.非电性质保安措施[y]==0:
            x='全部缺失'
        list.append(x)
    mn = pd.DataFrame(np.array(list), columns=['保护措施'])
    return mn
def Operation_management(): #运行管理
    list = []
    data = Data_cleaning()
    data.安全运行管理制度 = data.安全运行管理制度.astype('float')
    data.是否定期检查预试 = data.是否定期检查预试.astype('float')
    data.eval('new5=安全运行管理制度-是否定期检查预试', inplace=True)
    for y,x in enumerate(data.new5):
        if data.安全运行管理制度[y]!=0 and data.是否定期检查预试[y]!=0 and data.电工数量[y]!=0 and data.电工过期[y]==0:
            x='合规'
        elif data.安全运行管理制度[y]==0 and data.是否定期检查预试[y]==0 and data.电工数量[y]==0:
            x='严重缺失'
        else:
            x='急需修正'
        list.append(x)
    mn = pd.DataFrame(np.array(list), columns=['运行管理'])
    return mn
def emergency_plan():   #是否制定应急预案
    list = []
    data = Data_cleaning()
    data.是否制定应急预案 = data.是否制定应急预案.astype('float')
    for y,x in enumerate(data.是否制定应急预案):
        if x==1:
            x='是'
        elif x==0:
            x='否'
        list.append(x)
    mn = pd.DataFrame(np.array(list), columns=['是否制定应急预案'])
    return mn

def Quality_enterprise():  #企业管理质量   （通过聚类实现，一共聚为3类）
    list = []
    data1 = Data_cleaning().用户安全等级
    data2=Data_cleaning().隐患数目
    data3=pd.concat([data1,data2],axis=1)
    kmean=KMeans(n_clusters=3)
    kmean.fit(data3)
    data3['企业管理质量']=kmean.labels_
    data=data3['企业管理质量']
    for x in data:
        if x==0:
            x='中'
        elif x==1:
            x='高'
        elif x==2:
            x='低'
        list.append(x)
    mn = pd.DataFrame(np.array(list), columns=['企业管理质量'])
    return mn
def k_SSE(X, clusters):   #通过拐点法设置聚类的最佳参数
    # 选择连续的K种不同的
    K = range(1,clusters+1)
    # 构建空列表用于存储总的簇内离差平方和
    TSSE = []
    for k in K:
        # 用于存储各个簇内离差平方和
        SSE = []
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        # 返回簇标签
        labels = kmeans.labels_
        # 返回簇中心
        centers = kmeans.cluster_centers_
        # 计算各簇样本的离差平方和，并保存到列表中
        for label in set(labels):
            SSE.append(np.sum((X.loc[labels == label,]-centers[label,:])**2))
        # 计算总的簇内离差平方和
        TSSE.append(np.sum(SSE))
    # 中文和负号的正常显示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置绘图风格
    plt.style.use('ggplot')
    # 绘制K的个数与GSSE的关系
    plt.plot(K, TSSE, 'b*-')
    plt.xlabel('簇的个数')
    plt.ylabel('簇内离差平方和之和')
    # 显示图形
    plt.show()
# 自定义函数，计算簇内任意两样本之间的欧氏距离
def short_pair_wise_D(each_cluster):
    mu = each_cluster.mean(axis = 0)
    Dk = sum(sum((each_cluster - mu)**2)) * 2.0 * each_cluster.shape[0]
    return Dk
# 计算簇内的Wk值
def compute_Wk(data, classfication_result):
    Wk = 0
    label_set = set(classfication_result)
    for label in label_set:
        each_cluster = data[classfication_result == label, :]
        Wk = Wk + short_pair_wise_D(each_cluster)/(2.0*each_cluster.shape[0])
    return Wk
def gap_statistic(X, B=10, K=range(1, 11), N_init=10):
    # 将输入数据集转换为数组
    X = np.array(X)
    # 生成B组参照数据
    shape = X.shape
    tops = X.max(axis=0)
    bots = X.min(axis=0)
    dists = np.matrix(np.diag(tops - bots))
    rands = np.random.random_sample(size=(B, shape[0], shape[1]))
    for i in range(B):
        rands[i, :, :] = rands[i, :, :] * dists + bots

    # 自定义0元素的数组，用于存储gaps、Wks和Wkbs
    gaps = np.zeros(len(K))
    Wks = np.zeros(len(K))
    Wkbs = np.zeros((len(K), B))
    # 循环不同的k值，
    for idxk, k in enumerate(K):
        k_means = KMeans(n_clusters=k)
        k_means.fit(X)
        classfication_result = k_means.labels_
        # 将所有簇内的Wk存储起来
        Wks[idxk] = compute_Wk(X, classfication_result)

        # 通过循环，计算每一个参照数据集下的各簇Wk值
        for i in range(B):
            Xb = rands[i, :, :]
            k_means.fit(Xb)
            classfication_result_b = k_means.labels_
            Wkbs[idxk, i] = compute_Wk(Xb, classfication_result_b)

    # 计算gaps、sd_ks、sk和gapDiff
    gaps = (np.log(Wkbs)).mean(axis=1) - np.log(Wks)
    sd_ks = np.std(np.log(Wkbs), axis=1)
    sk = sd_ks * np.sqrt(1 + 1.0 / B)
    # 用于判别最佳k的标准，当gapDiff首次为正时，对应的k即为目标值
    gapDiff = gaps[:-1] - gaps[1:] + sk[1:]

    # 中文和负号的正常显示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置绘图风格
    plt.style.use('ggplot')
    # 绘制gapDiff的条形图
    plt.bar(np.arange(len(gapDiff)) + 1, gapDiff, color='steelblue')
    plt.xlabel('簇的个数')
    plt.ylabel('k的选择标准')
    plt.show()
# 构造自定义函数，用于绘制不同k值和对应轮廓系数的折线图
def k_silhouette(X, clusters):
    K = range(2,clusters+1)
    # 构建空列表，用于存储个中簇数下的轮廓系数
    S = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        labels = kmeans.labels_
        # 调用字模块metrics中的silhouette_score函数，计算轮廓系数
        S.append(metrics.silhouette_score(X, labels, metric='euclidean'))
    # 中文和负号的正常显示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置绘图风格
    plt.style.use('ggplot')
    # 绘制K的个数与轮廓系数的关系
    plt.plot(K, S, 'b*-')
    plt.xlabel('簇的个数')
    plt.ylabel('轮廓系数')
    # 显示图形
    plt.show()
def Customer_security_type():     #客户安全类型  (聚类实现)
    list1 = []
    list2=[]
    list=[]
    data1 = Data_cleaning()
    for x in data1.负荷性质:
        if x=='一类':
            x=3
        elif x=='二类':
            x=2
        elif x=='三类':
            x=1
        list1.append(x)
    for i in data1.客户级别:
        if i=='一级重要用户':
            i=3
        elif i=='二级重要用户':
            i=2
        elif i=='三级重要用户':
            i=1
        list2.append(i)
    m1= pd.DataFrame(np.array(list1), columns=['负荷性质'])
    m2=pd.DataFrame(np.array(list2), columns=['客户级别'])
    m3=pd.concat([m1,m2,Data_cleaning().隐患数目],axis=1)
    m3.负荷性质=m3.负荷性质.astype('category')
    m3.客户级别 = m3.客户级别.astype('category')
    kmean=KMeans(n_clusters=4)
    kmean.fit(m3)
    m3['客户安全类型'] = kmean.labels_
    data=m3['客户安全类型']
    for x in m3.客户安全类型:
        if x==0:
            x='中危险型'
        elif x==2:
            x='高危险型'
        elif x==1:
            x='适当关注型'
        elif x==3:
            x='低危险型'
        list.append(x)
    mn=pd.DataFrame(np.array(list), columns=['客户安全类型'])
    return mn
    # print(m3)
def protect_data():    #安全隐患时间处理
    list = []
    list1=[]
    data = Data_cleaning()
    data.eval('new12=证件数目-证件在期数目', inplace=True)
    now=datetime.datetime.now()
    datetime_hetong=data.合同签订日期
    for x in datetime_hetong:
        mm=((now-x).days)/356
        if mm>=1:
            n=1
        else:
            n=0
        list.append(n)
    mn1 = pd.DataFrame(np.array(list), columns=['新老客户'])
    mm=pd.concat([mn1,data.隐患完成时间],axis=1)
    kmean = KMeans(n_clusters=3)
    kmean.fit(mm)
    mm['隐患整改积极性'] = kmean.labels_
    for x in mm.隐患整改积极性:
        if x==0:
            x='积极配合型'
        elif x==1:
            x='懒惰怠慢型'
        elif x==2:
            x='适当配合型'
        list1.append(x)
    mn = pd.DataFrame(np.array(list1), columns=['隐患整改积极性'])
    return mn
def Enterprise_standard():
    list=[]
    data=Data_cleaning()
    m1=data.证件在期数目 /data.证件数目
    m2=pd.DataFrame(m1,columns=['证件在期率'])
    m3=pd.concat([m2,data.是否有合同],axis=1)
    # k_SSE(m3, 10)     聚为4类较为合适
    kmean = KMeans(n_clusters=4)
    kmean.fit(m3)
    m3['企业规范'] = kmean.labels_
    # for x,y in m3.iterrows():
    #     print(y)
    for x in m3.企业规范:
        if x==0:
            x="极高"
        elif x==1:
            x='一般'
        elif x==2:
            x='高'
        elif x==3:
            x=='较差'
        list.append(x)
    mn = pd.DataFrame(np.array(list), columns=['企业规范'])
    return mn

def datalist():   #将数据框写入数据库里面去
    train_data = np.array(Data_cleaning())
    train_newdata_list = train_data.tolist()  # list
    return train_newdata_list
def insert_sql():
    db = cx_Oracle.connect("HRICMS/123456@127.0.0.1:1521/ORCL")# 连接数据库
    cursors = db.cursor()  # 创建一个游标
    cursors.execute("DROP TABLE customs")
    cursors.execute("create table customs(mon varchar(20),cons_id int,"
                    "cons_name varchar(500),"
                    "客户级别 varchar(20),"
                    "客户类型 varchar(20),"
                    "客户状态 varchar(20),"
                    "行业 varchar(20),"
                    "用户安全等级 int,"
                    "保安负荷 int,"
                    "合同状态 varchar(20),"
                    "合同类型 varchar(20),"
                    "用电分类 varchar(20),"
                    "负荷性质 varchar(20),"
                    "供电电压 varchar(20),"
                    "电源数目 varchar(20),供电方式 varchar(300),报装容量 varchar(20),是否制定应急预案 varchar(20),"
                    "安全运行管理制度 varchar(20),是否定期检查预试 varchar(20),自动化装置配合 int,非电性质保安措施 int,供电企业性质 varchar(20),运行容量 int,"
                    "自备电源数目 int,自备电源闭锁情况 int,电工数量 int,电工过期 int,供电单位 varchar(200),档案完整率 int,"
                   "经度坐标 float,纬度坐标 float,证件数目 int,证件在期数目 int,证件类型 varchar(400),隐患数目 int,安全隐患类型 varchar(200))")
    for x in Data_cleaning():
        list = []
        for i in x:
            list.append(i)
        cursors.execute("insert into customs(mon,cons_id,cons_name,客户级别,客户类型,客户状态,行业,用户安全等级,保安负荷,合同状态，"
                        "合同类型,用电分类,负荷性质,供电电压,电源数目,供电方式,报装容量,是否制定应急预案,安全运行管理制度,是否定期检查预试,自动化装置配合,非电性质保安措施,"
                        "供电企业性质,运行容量,自备电源数目,自备电源闭锁情况,电工数量,电工过期,供电单位,档案完整率,经度坐标,纬度坐标,证件数目,证件在期数目,证件类型,隐患数目,安全隐患类型) "
                        "values('%s',%s,'%s','%s','%s','%s','%s',%s,%s,'%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s',%s,%s,'%s',%s,%s,%s,%s,%s,'%s',%s,%s,%s,%s,%s,'%s',%s,'%s')"
                        %(list[0],list[1],list[2],list[3],list[4],list[5],list[6],list[7],list[8],list[9],list[10],list[11],list[12],
                          list[13],list[14],list[15],list[16],list[17],list[18],list[19],list[20],list[21],list[22],list[23],list[24],list[25],list[26],
                          list[27],list[28],list[29],list[30],list[31],list[32],list[33],list[34],list[35],list[36]))
        print(""+list[3])
    db.commit()
    db.close()
def insert_sql():
    db = cx_Oracle.connect("HRICMS/123456@127.0.0.1:1521/ORCL")# 连接数据库
    cursors = db.cursor()  # 创建一个游标
    cursors.execute("DROP TABLE customss")
    cursors.execute("DROP TABLE customs")
    cursors.execute("create table customs(mon varchar(20),cons_id int)")

# print(Enterprise_standard())
# if __name__=='__mian__':
#     insert_sql()
def hp(y, lamb=10):
    def D_matrix(N):
        D = np.zeros((N-1,N))
        D[:,1:] = np.eye(N-1)
        D[:,:-1] -= np.eye(N-1)
        """D1
        [[-1.  1.  0. ...  0.  0.  0.]
         [ 0. -1.  1. ...  0.  0.  0.]
         [ 0.  0. -1. ...  0.  0.  0.]
         ...
         [ 0.  0.  0. ...  1.  0.  0.]
         [ 0.  0.  0. ... -1.  1.  0.]
         [ 0.  0.  0. ...  0. -1.  1.]]
        """
        return D
    N = len(ts)
    D1 = D_matrix(N)
    D2 = D_matrix(N-1)
    D = D2 @ D1
    g = np.linalg.inv((np.eye(N)+lamb*D.T@D))@ ts
    return g
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

N = 100
t = np.linspace(1,10,N)
ts = np.sin(t) + np.cos(20*t) + np.random.randn(N)*0.1
plt.figure(figsize=(10,12))
for i,l in enumerate([0.1,1,10,100,1000, 10000]):
    plt.subplot(3,2,i+1)
    g = hp(ts,l)
    plt.plot(ts, label='original')
    plt.plot(g, label='filtered')
    plt.legend()
    plt.title('$\lambda$='+str(l))
plt.show()
