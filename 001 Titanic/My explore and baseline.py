#From Jupyter notebook
#C1_Titanic T5.txt

#1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
f=open("E:/Tinky/大学课件及作业/6 自学课/6-3.Kaggle竞赛/C1_泰坦尼克号生还预测/泰坦尼克号数据/train.csv")
data=pd.read_csv(f)

#2 数据可视化
fig=plt.figure(figsize=(18,6))
alpha=alpha_scatterplot=0.2
alpha_bar_chart=0.55
ax1=plt.subplot2grid((2,3),(0,0))
data.Survived.value_counts().plot(kind='bar',alpha=alpha_bar_chart)
ax1.set_xlim(-1,2)
plt.title("Distribution of Survival,(1=Survived)")

#绘制年龄的散点图
plt.subplot2grid((2,3),(0,1))
plt.scatter(data.Survived,data.Age,alpha=alpha_scatterplot)
plt.ylabel('Age')
plt.grid(b=True,which='major',axis='y')
plt.title("Survival by Age,(1=Survived)")

#Class的直方图
ax3=plt.subplot2grid((2,3),(0,2))
data.Pclass.value_counts().plot(kind="barh",alpha=alpha_bar_chart)
ax3.set_ylim(-1,len(data.Pclass.value_counts()))
plt.title("Class Distribution")

#Class点的密度
plt.subplot2grid((2,3),(1,0),colspan=2)#横向占了两格，如果是纵向就是rowspan=x
data.Age[data.Pclass==1].plot(kind='kde')
data.Age[data.Pclass==2].plot(kind='kde')
data.Age[data.Pclass==3].plot(kind='kde')
plt.xlabel('Age')
plt.title('Age Distribution within classer')
plt.legend(('1st Class','2nd Class','3rd Class'),loc='best')

#查看不同Boarding Location的直方图
ax5=plt.subplot2grid((2,3),(1,2))
data.Embarked.value_counts().plot(kind='bar',alpha=alpha_bar_chart)
ax5.set_xlim(-1,len(data.Embarked.value_counts()))
plt.title("Passengers per boarding location")
plt.show()

#3 生还情况：查看是否生还的直方图
plt.figure(figsize=(6,4))
ax=plt.subplot()
data.Survived.value_counts().plot(kind='barh',color='blue',alpha=0.65)
ax.set_ylim(-1,len(data.Survived.value_counts()))
plt.title("Survival Breakdown (1=Survived,0=Died)")
plt.show()

#4 生还与性别的关系
fig2=plt.figure(figsize=(18,6))
data_male=data.Survived[data.Sex=='male'].value_counts().sort_index()#sort_index:对行列进行索引排序
data_female=data.Survived[data.Sex=='female'].value_counts().sort_index()
ax1=fig2.add_subplot(121)#一行两列第一个位置，add_subplot：画子图，参数含义与subplot相同
data_male.plot(kind='barh',label='Male',alpha=0.55)
data_female.plot(kind='barh',color='#FA2379',label='Female',alpha=0.55)
plt.title("Who Survived? With respect to Gender, (raw value counts)")
plt.legend(loc='best')
ax1.set_ylim(-1,2)

#生还比例的直方图
ax2=fig2.add_subplot(122)
(data_male/float(data_male.sum())).plot(kind='barh',label='Male',alpha=0.55)
(data_female/float(data_female.sum())).plot(kind='barh',color='#FA2379',label='Female',alpha=0.55)
plt.title("Who Survived proportionally? with respect to Gender")
plt.legend(loc='best')
ax2.set_ylim(-1,2)
plt.show()

#5
fig3=plt.figure(figsize=(18,12))
a=0.65
w=0.35#设置宽度
index = np.arange(2)
#A 生还人数对比
ax1=fig3.add_subplot(341)
data.Survived.value_counts().plot(width=w,kind='bar',color='blue',alpha=a)
ax1.set_xlim(-1,len(data.Survived.value_counts()))
plt.title("Step.1")
#B 性别是否有关
ax2=fig3.add_subplot(345)
#data.Survived[data.Sex=='male'].value_counts().plot(width=w,kind='bar',label='Male')#我改成改为下面两行，画出来更好看
plt.bar(index,data.Survived[data.Sex=='male'].value_counts() , w,  color='blue', label='Male')
plt.xticks(index + w, ('Died', 'Survived'))
#data.Survived[data.Sex=='female'].value_counts().plot(width=w,kind='bar',color='#FA2379',label='Female')
plt.bar(index+w,data.Survived[data.Sex=='female'].value_counts() , w,color='#FA2379',label='Female')
plt.xticks(index + w,('Died', 'Survived'))
ax2.set_xlim(-1,2)
plt.title("Step.2 \nWho survived?with respect to Gender.")
plt.legend(loc='best')
ax3=fig3.add_subplot(346)
(data.Survived[data.Sex=='male'].value_counts()/float(data.Sex[data.Sex=='male'].size)).plot(width=w,kind='bar',label='Male')
(data.Survived[data.Sex=='female'].value_counts()/float(data.Sex[data.Sex=='female'].size)).plot(width=w,kind='bar',color='#FA2379',label='Female')
ax3.set_xlim(-1,2)
plt.title("Who survived proportionally?")
plt.legend(loc='best')
#C 是否与社会地位有关？
#female high class
ax4=fig3.add_subplot(349)
female_highclass=data.Survived[data.Sex=='female'][data.Pclass!=3].value_counts()
female_highclass.plot(kind='bar',label='female,highclass',color='#FA2479',alpha=a)
ax4.set_xticklabels(['Survived','Died'],rotation=0)
ax4.set_xlim(-1,len(female_highclass))
plt.title("Who Survived? with respect to Gender and Class")
plt.legend(loc='best')
#female low class
ax5=fig3.add_subplot(3,4,10,sharey=ax4)#指定具有相同的y轴（或x轴 sharex）
female_lowclass=data.Survived[data.Sex=='female'][data.Pclass==3].value_counts()
female_lowclass.plot(kind='bar',label='female,lowclass',color='pink',alpha=a)
ax5.set_xticklabels(['Died','Survived'],rotation=0)
ax5.set_xlim(-1,len(female_lowclass))
plt.legend(loc='best')
#male low class
ax6=fig3.add_subplot(3,4,11,sharey=ax4)
male_lowclass=data.Survived[data.Sex=='male'][data.Pclass==3].value_counts()
male_lowclass.plot(kind='bar',label='male,lowclass',color='lightblue',alpha=a)
ax6.set_xticklabels(['Died','Survived'],rotation=0)
ax6.set_xlim(-1,len(male_lowclass))
plt.legend(loc='best')
#male high class
ax7=fig3.add_subplot(3,4,12,sharey=ax4)
male_highclass=data.Survived[data.Sex=='male'][data.Pclass!=3].value_counts()
male_highclass.plot(kind='bar',label='male,highclass',color='steelblue',alpha=a)
ax7.set_xticklabels(['Died','Survived'],rotation=0)
ax7.set_xlim(-1,len(male_highclass))
plt.legend(loc='best')
plt.show()

#6 兄弟姐妹是否有关
g = data.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print(df)
data.Cabin.value_counts()#和Cabin的关系

#7
from sklearn.ensemble import RandomForestRegressor
#拟合缺失的年龄数据，此处用 RandomForestClassifier
def Fix_the_missing_ages(df):
    # 把已有的数值型特征取出来放入Random Forest Regressor
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    # 分类：已知年龄、未知年龄
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()#as_matrix: 将dataframe变为numpy的ndarrey
    y = known_age[:, 0]# 预测的目标年龄
    X = known_age[:, 1:]# 特征属性值
    #用RamdomForest拟合
    RFR_ = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    RFR_.fit(X, y)
    predictedAges = RFR_.predict(unknown_age[:, 1::])# 用得到的模型进行未知年龄预测
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges# 用得到的预测结果填补原缺失数据
    return df, RFR_
def Setting_Cabin_types(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df
data, RFR_ = Fix_the_missing_ages(data)
data = Setting_Cabin_types(data)#将Cabin那一列根据数据的有无换为Yes和No

#8
#逻辑回归建模时，需要输入的特征都是数值型特征，这里我先对类目型的特征因子化。
#Cabin原本取值是[‘yes’,’no’]，这里我将其变为’Cabin_yes’,’Cabin_no’两个属性

#原本Cabin为yes，”Cabin_yes”=1，”Cabin_no”=0
#原本Cabin为no，”Cabin_yes”=0，”Cabin_no”=1

dummies_Cabin = pd.get_dummies(data['Cabin'], prefix= 'Cabin')#使用pandas的”get_dummies”，并拼接在原来的”data_train”之上
#data : 列数据或表格，prefix:新建的列名
dummies_Embarked = pd.get_dummies(data['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data['Pclass'], prefix= 'Pclass')
data = pd.concat([data,dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
data.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)#丢弃原先的那些列

#9
#这里我用了scaling，防止fare、age列相差比较大的数值影响回归
#用preprocessing模块，将一些变化幅度较大的特征化到[-1,1]之内并保证均值为零，方差为一
import sklearn.preprocessing as preprocessing
scaler=preprocessing.StandardScaler()
Age_scalefixed=scaler.fit(data['Age'])
data['Age_scaled']=scaler.fit_transform(data['Age'],Age_scalefixed)
Fare_scalefixed=scaler.fit(data['Fare'])
data['Fare_scaled']=scaler.fit_transform(data['Fare'],Fare_scalefixed)

#10
from sklearn import  linear_model
#建模；抽出属性特征，转成LogisticRegression可以处理的格式
#把需要feature字段取出，转成numpy格式，使用scikit-learn中的LogisticRegression建模
#用正则取出需要的属性值，用filter构建器的Regex方法构建正则过滤，其中正则化的特征用_.*表示
train_df=data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np=train_df.as_matrix()
y=train_np[:,0]#Survive的结果
X=train_np[:,1:]#特征属性值
#拟合
clf=linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
clf.fit(X,y)

#11
#将测试集做相同的处理（特征变换也相同）
f0=open("E:/Tinky/大学课件及作业/6 自学课/6-3.Kaggle竞赛/C1_泰坦尼克号生还预测/泰坦尼克号数据/test.csv")
data1=pd.read_csv(f0)
data1.loc[ (data1.Fare.isnull()), 'Fare' ] = 0
tmp_df = data1[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data1.Age.isnull()].as_matrix()
X = null_age[:, 1:]
predictedAges = RFR_.predict(X)
data1.loc[ (data1.Age.isnull()), 'Age' ] = predictedAges

data1 = Setting_Cabin_types(data1)
dummies_Cabin = pd.get_dummies(data1['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data1['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data1['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data1['Pclass'], prefix= 'Pclass')


data1 = pd.concat([data1, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
data1.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
data1['Age_scaled'] = scaler.fit_transform(data1['Age'], Age_scalefixed)
data1['Fare_scaled'] = scaler.fit_transform(data1['Fare'], Fare_scalefixed)

#12
#预测结果
test = data1.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions=clf.predict(test)
result=pd.DataFrame({'PassengerId':data1['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32)})
result.to_csv("E:/Tinky/大学课件及作业/6 自学课/6-3.Kaggle竞赛/C1_泰坦尼克号生还预测/泰坦尼克号数据/test_X0.csv",index=False)
#最基本的模型，准确率0,76555

#13
# 参考了一些方法进行优化
#关联分析，model系数和feature
pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})#根据正负号判断相关性

#14 交叉验证
from sklearn import cross_validation
clf=linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
all_data=data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
X=all_data.as_matrix()[:,1:]
y=all_data.as_matrix()[:,0]#用X[:,0]选取第一行, X[:,1] 取其余行
p=cross_validation.cross_val_score(clf,X,y,cv=5)
print(p)
'''
PS:cross_validation被废弃后，可以改为：
#from sklearn.model_selection import train_test_split
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=33)
'''

#15 查看误判数据，手动分析
#分割数据，3：7的比例
split_train,split_cv=cross_validation.train_test_split(data,test_size=0.3,random_state=0)
train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
#模型拟合
clf=linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
clf.fit(train_df.as_matrix()[:,1:],train_df.as_matrix()[:,0])
#对交叉验证的数据预测
cv_df=split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(cv_df.as_matrix()[:,1:])
f1=open("E:/Tinky/大学课件及作业/6 自学课/6-3.Kaggle竞赛/C1_泰坦尼克号生还预测/泰坦尼克号数据/train.csv")
origin_data_train=pd.read_csv(f1)
bad_cases=origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions!=cv_df.as_matrix()[:,0]]['PassengerId'].values)]
print(bad_cases)
#有了”train_df” 和 “vc_df” 两个数据部分，前者用于训练model，后者用于评定和选择模型，可以反复进行

#16 判断是否过拟合
# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出拟合曲线
from sklearn.learning_curve import learning_curve
def Learning_curve_drawing(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(0.05, 1.0, 30), verbose=0, plot=True):
    """
    参数解释
    estimator : 用的分类器。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training
    n_jobs : 并行的的任务数
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)#计算矩阵平均值
    train_scores_std = np.std(train_scores, axis=1)#计算矩阵标准差
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Size of Sample")
        plt.ylabel("Score")
        plt.gca().invert_yaxis()#获得当前Axes对象ax
        plt.grid()#显示网格

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")#将两条曲线之间填充上颜色（很直观的表示）
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="Training Score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="Score on CV")
        plt.legend(loc="best")
        plt.draw()#draw可以进行交互模型绘制，改变数据或表格时图表自身也会变化
        plt.gca().invert_yaxis()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

Learning_curve_drawing(clf, "Learning Curcve", X, y)

#17 模型融合（原理：不同模型建模判断，投票式决定最终结果）
#每次取训练集的一个subset做训练，能用同一个算法得到不一样的模型
#用scikit-learn的Bagging完成
from sklearn.ensemble import BaggingRegressor
train_df = data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*')
train_np = train_df.as_matrix()
y = train_np[:, 0]# ySurvival结果
X = train_np[:, 1:]# 特征属性值
# 用BaggingRegressor拟合
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)
test = data1.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*')
predictions = bagging_clf.predict(test)
result = pd.DataFrame({'PassengerId':data1['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("E:/Tinky/大学课件及作业/6 自学课/6-3.Kaggle竞赛/C1_泰坦尼克号生还预测/泰坦尼克号数据/test_Xf.csv", index=False)
#这次的准确率0.77511
