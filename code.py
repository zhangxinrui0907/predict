import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold,cross_val_score
import matplotlib.pyplot as plt
from  sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
import graphviz
from gplearn.genetic import SymbolicRegressor
from bayes_opt import BayesianOptimization
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
dataset = pd.read_excel('datasets.xlsx')
height,width = dataset.shape

column = dataset.columns
# print(bb)
x_data_o = dataset.iloc[:,3:14]#Original feature set
x_data_a = dataset.iloc[:,[14,15,16,17,18,8,9,12,13]]#feature set a
x_data_b = dataset.iloc[:,[19,20,21,22,23,7,8,11]]#feature set b
x_data = x_data_o
y_data_k = dataset.iloc[:,1]
y_data_g = dataset.iloc[:,2]
# print(x_data.columns)
# print(x_data_a.columns)
# print(x_data_b.columns)
#5-Fold
a=[]
b = []
cv=KFold(n_splits=5,shuffle=True ,random_state=42
         )

#-----------------------------------------------------------------------
# Original feature set pre_K
# lasso
modle = Lasso(
                alpha=0.036,
                )
rmse = cross_val_score(modle,
                       x_data,y_data_k,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_k,
                     cv=cv,scoring='r2')
# print(np.mean(r2),np.mean(rmse))
a.append(np.mean(r2))
b.append(np.mean(rmse))
# ridge
modle =Ridge(
            alpha=0.0016
             )
rmse = cross_val_score(modle,
                       x_data,y_data_k,
                       cv=cv,scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_k,
                     cv=cv,scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
# KNN
modle = KNeighborsRegressor(
                            n_neighbors=5,
                            leaf_size=30,
                            n_jobs=1,
                    )
rmse = cross_val_score(modle,
                       x_data,y_data_k,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_k,cv=cv,
                     scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
# DT
modle = DecisionTreeRegressor(
                                max_depth=None,
                                min_samples_split=2, min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,
                              )
rmse = cross_val_score(modle,
                       x_data,y_data_k,
                       cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_k,cv=cv,
                     scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
# ----------------------------
# RF,BayesianOptimization
cv=KFold(n_splits=5,shuffle=True,random_state=42)
def black_box_function(n_estimators, min_samples_split,min_sample_leaf, max_features,max_depth):
    modle = RandomForestRegressor(n_estimators=int(n_estimators),
                                min_samples_split=int(min_samples_split),
                                min_samples_leaf=int(min_sample_leaf),
                                max_features=min(max_features, 0.999),  # float
                                max_depth=int(max_depth),
                                random_state=42
                                )
    # modle.fit(x_train_k,y_train_k)
    # pre_k = modle.predict(x_test_k)
    cv = KFold(n_splits=5, shuffle=True, random_state=42
               )
    res = cross_val_score(modle,x_data,y_data_k,cv = cv,scoring='r2')
    return np.mean(res)
pbounds= {'n_estimators': (5, 1000),
         'min_samples_split': (2, 100),
         'min_sample_leaf':(2,100),
         'max_features': (0.1, 1),
         'max_depth': (2, 30)}
optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=42,
    )
# BayesianOptimization
optimizer.maximize(
        init_points=25,
        n_iter=50,)
modle = RandomForestRegressor(
                                n_estimators=120,
                                min_samples_split=37,
                                min_samples_leaf=9,
                                max_features=0.7941, # float
                                max_depth=22,
                                random_state=42
                                )
rmse = cross_val_score(modle,
                       x_data,y_data_k,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_k,cv=cv,
                     scoring='r2')
# print(np.mean(r2),np.mean(rmse))
a.append(np.mean(r2))
b.append(np.mean(rmse))
#xgboost
def black_box_function(max_depth,learning_rate,n_estimators):
    modle = xgb.XGBRegressor(max_depth=int(max_depth),
                # gama = gama,
                learning_rate=learning_rate,
                n_estimators=int(n_estimators)
                                )
    res = cross_val_score(modle,x_data,y_data_k,cv = cv,scoring='r2')
    return np.mean(res)
pbounds= {
    'max_depth': (2, 30),
    # 'gama':(0,1),
    'learning_rate':(0,0.1),
    'n_estimators': (5, 1000),

         }
optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=42,
    )
optimizer.maximize(
        init_points=50,
        n_iter=50,
    )
print(optimizer.max)

modle = xgb.XGBRegressor(
                            max_depth=2,
                            learning_rate=0.7081,
                            n_estimators=970,
                            n_jobs=1,
                            gamma=0,

                         )

rmse = cross_val_score(modle,
                       x_data,y_data_k,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_k,cv=cv,
                     scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
# LGBM
def black_box_function(learning_rate,n_estimators, min_child_samples,max_depth):
    modle = lgb.LGBMRegressor(learning_rate=learning_rate,
                                n_estimators=int(n_estimators),
                                min_child_samples=int(min_child_samples),
                                max_depth=int(max_depth),
                                random_state=42
                                )
    res = cross_val_score(modle,x_data,y_data_k,cv = cv,scoring='r2')
    return np.mean(res)
pbounds= {'learning_rate':(0.0001,0.1),
        'n_estimators': (5, 1000),
          'min_child_samples':(18,40),
         'max_depth': (2, 30)}
optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=0,
    )
optimizer.maximize(
        init_points=25,  #执行随机搜索的步数
        n_iter=50,   #执行贝叶斯优化的步数
    )
print(optimizer.max)
modle = lgb.LGBMRegressor(learning_rate=0.05223,
                                n_estimators=775,
                                min_child_samples=23,
                                max_depth=13,
                                random_state=42)
rmse = cross_val_score(modle,
                       x_data,y_data_k,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_k,cv=cv,
                     scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
# print('R2:',a,'RMSE:',b)
#------------------------------------------------------------------------------------
# Original feature set pre_G
# lasso
modle = Lasso(
                alpha=0.0025,
                )
rmse = cross_val_score(modle,
                       x_data,y_data_g,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_g,
                     cv=cv,scoring='r2')
# print(np.mean(r2),np.mean(rmse))
a.append(np.mean(r2))
b.append(np.mean(rmse))
# ridge
modle =Ridge(
            alpha=0.002
             )
rmse = cross_val_score(modle,
                       x_data,y_data_g,
                       cv=cv,scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_g,
                     cv=cv,scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
#KNN
modle = KNeighborsRegressor(
                            n_neighbors=5,
                            leaf_size=30,
                            n_jobs=1,
                    )
rmse = cross_val_score(modle,
                       x_data,y_data_g,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_g,cv=cv,
                     scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
# DT
modle = DecisionTreeRegressor(
                                max_depth=None,
                                min_samples_split=2, min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,
                              )
rmse = cross_val_score(modle,
                       x_data,y_data_g,
                       cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_g,cv=cv,
                     scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
# RF
modle = RandomForestRegressor(
                                    n_estimators=10,
                                    max_depth=None,
                                    min_samples_split=2,
                                    min_samples_leaf=1,

                                      )
rmse = cross_val_score(modle,
                       x_data,y_data_g,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_g,cv=cv,
                     scoring='r2')
print(np.mean(r2),np.mean(rmse))
a.append(np.mean(r2))
b.append(np.mean(rmse))
#xgboost
def black_box_function(max_depth,learning_rate,n_estimators):
    modle = xgb.XGBRegressor(max_depth=int(max_depth),
                # gama = gama,
                learning_rate=learning_rate,
                n_estimators=int(n_estimators)
                                )
    res = cross_val_score(modle,x_data,y_data_g,cv = cv,scoring='r2')
    return np.mean(res)
pbounds= {
    'max_depth': (2, 30),
    # 'gama':(0,1),
    'learning_rate':(0,0.1),
    'n_estimators': (5, 1000),

         }
optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=42,
    )
optimizer.maximize(
        init_points=25,
        n_iter=50,
)
modle = xgb.XGBRegressor(
                            max_depth=3,
                            learning_rate=0.0196,
                            n_estimators=328,
                            n_jobs=1,
                            gamma=0,

                         )

rmse = cross_val_score(modle,
                       x_data,y_data_g,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_g,cv=cv,
                     scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
# LGBM
def black_box_function(learning_rate,n_estimators, min_child_samples,max_depth):
    modle = lgb.LGBMRegressor(learning_rate=learning_rate,
                                n_estimators=int(n_estimators),
                                min_child_samples=int(min_child_samples),
                                max_depth=int(max_depth),
                                random_state=42
                                )
    res = cross_val_score(modle,x_data,y_data_g,cv = cv,scoring='r2')
    return np.mean(res)
pbounds= {'learning_rate':(0.0001,0.1),
        'n_estimators': (5, 1000),
          'min_child_samples':(18,40),
         'max_depth': (2, 30)}
optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=0,
    )
optimizer.maximize(
        init_points=25,
        n_iter=50,
    )
print(optimizer.max)
modle = lgb.LGBMRegressor(
    learning_rate=0.04242,
    max_depth=20,
    min_child_samples=27,
    n_estimators=892)
rmse = cross_val_score(modle,
                       x_data,y_data_g,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_g,cv=cv,
                     scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
# print('R2:',a,'RMSE:',b)
#----------------Original feature SymbolicRegressor
# a1-a5
modle = SymbolicRegressor(population_size=1000,
                              generations=20,
                              tournament_size=20,
                              stopping_criteria=0.0,
                              const_range=(- 1.0, 1.0),
                              init_depth=(2, 6),
                              init_method='half and half',
                              function_set=('add', 'sub', 'mul', 'div'
                                            ,'sqrt','log','abs','neg',
                                            'inv','sin','cos','tan'

                                            ),
                              metric='pearson',
                              parsimony_coefficient='auto',
                              p_crossover=0.9,
                              p_subtree_mutation=0.01,
                              p_hoist_mutation=0.01,
                              p_point_mutation=0.01,
                              p_point_replace=0.05,
                              max_samples=1.0,
                              feature_names=['Smix', 'δ', 'Vm', 'χ', '∆χ',
                                                'VEC', '∆VEC', 'Tm', '∆Tm', 'Ea',
                                                '∆E'],
                              warm_start=False,
                              low_memory=False,
                              n_jobs=1,
                              verbose=0,
                              random_state=0



    )
modle.fit(x_data,y_data_k)
# print(modle._program)
pre_k = modle.predict(x_data)
A = pd.Series(pre_k)
B = pd.Series(y_data_k)
pr = A.corr(B,method='pearson')
# print(pr)
# dot_data = modle._program.export_graphviz()
# graph = graphviz.Source(dot_data)
# # print(graph)
# graph.view()
# b1-b5
modle = SymbolicRegressor(population_size=1000,
                              generations=20,
                              tournament_size=20,
                              stopping_criteria=0.0,
                              const_range=(- 1.0, 1.0),
                              init_depth=(2, 6),
                              init_method='half and half',
                              function_set=('add', 'sub', 'mul', 'div'
                                            ,'sqrt','log','abs','neg',
                                            'inv','sin','cos','tan'

                                            ),
                              metric='pearson',
                              parsimony_coefficient='auto',
                              p_crossover=0.9,
                              p_subtree_mutation=0.01,
                              p_hoist_mutation=0.01,
                              p_point_mutation=0.01,
                              p_point_replace=0.05,
                              max_samples=1.0,
                              feature_names=['Smix', 'δ', 'Vm', 'χ', '∆χ',
                                                'VEC', '∆VEC', 'Tm', '∆Tm', 'Ea',
                                                '∆E'],
                              warm_start=False,
                              low_memory=False,
                              n_jobs=1,
                              verbose=0,
                              random_state=0



    )
modle.fit(x_data,y_data_g)
# print(modle._program)
pre_g = modle.predict(x_data)
A = pd.Series(pre_g)
B = pd.Series(y_data_g)
pr = A.corr(B,method='pearson')
# print(pr)
# dot_data = modle._program.export_graphviz()
# graph = graphviz.Source(dot_data)
# # print(graph)
# graph.view()
#------------------------------------------------------------------------------
# feature set a pre_k
x_data = x_data_a
# lasso
modle = Lasso(
                alpha=0.0016,
                )
rmse = cross_val_score(modle,
                       x_data,y_data_k,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_k,
                     cv=cv,scoring='r2')
print(np.mean(r2),np.mean(rmse))
a.append(np.mean(r2))
b.append(np.mean(rmse))
# ridge
modle =Ridge(
            alpha=0.00013
             )
rmse = cross_val_score(modle,
                       x_data,y_data_k,
                       cv=cv,scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_k,
                     cv=cv,scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
# KNN
modle = KNeighborsRegressor(
                            n_neighbors=5,
                            leaf_size=30,
                            n_jobs=1,
                    )
rmse = cross_val_score(modle,
                       x_data,y_data_k,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_k,cv=cv,
                     scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
# DT
modle = DecisionTreeRegressor(
                                max_depth=None,
                                min_samples_split=2, min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,
                              )
rmse = cross_val_score(modle,
                       x_data,y_data_k,
                       cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_k,cv=cv,
                     scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
# RF
def black_box_function(n_estimators, min_samples_split,min_sample_leaf, max_features,max_depth):
    modle = RandomForestRegressor(n_estimators=int(n_estimators),
                                min_samples_split=int(min_samples_split),
                                min_samples_leaf=int(min_sample_leaf),
                                max_features=min(max_features, 0.999),  # float
                                max_depth=int(max_depth),
                                random_state=42
                                )
    # modle.fit(x_train_k,y_train_k)
    # pre_k = modle.predict(x_test_k)
    cv = KFold(n_splits=5, shuffle=True, random_state=42
               )
    res = cross_val_score(modle,x_data,y_data_k,cv = cv,scoring='r2')
    return np.mean(res)
pbounds= {'n_estimators': (5, 1000),
         'min_samples_split': (2, 100),
         'min_sample_leaf':(2,100),
         'max_features': (0.1, 1),
         'max_depth': (2, 30)}
optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=42,
    )
# optimizer.maximize(
#         init_points=100,  #执行随机搜索的步数
#         n_iter=50,   #执行贝叶斯优化的步数
#     )
modle = RandomForestRegressor(n_estimators=50,
                                min_samples_split=13,
                                min_samples_leaf=3,
                                max_features=0.7821,  # float
                                max_depth=11,
                                random_state=42

                                      )
rmse = cross_val_score(modle,
                       x_data,y_data_k,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_k,cv=cv,
                     scoring='r2')
# print(np.mean(r2),np.mean(rmse))
a.append(np.mean(r2))
b.append(np.mean(rmse))
#xgboost
def black_box_function(max_depth,learning_rate,n_estimators):
    modle = xgb.XGBRegressor(max_depth=int(max_depth),
                # gama = gama,
                learning_rate=learning_rate,
                n_estimators=int(n_estimators)
                                )
    res = cross_val_score(modle,x_data,y_data_k,cv = cv,scoring='r2')
    return np.mean(res)
pbounds= {
    'max_depth': (2, 30),
    # 'gama':(0,1),
    'learning_rate':(0,0.1),
    'n_estimators': (5, 1000),

         }
optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=42,
        )
optimizer.maximize(
        init_points=25,  #执行随机搜索的步数
        n_iter=25,
        )
modle = xgb.XGBRegressor(
                            max_depth=3,
                            learning_rate=0.05027,
                            n_estimators=282,
                            n_jobs=1,
                            gamma=0
                        )

rmse = cross_val_score(modle,
                       x_data,y_data_k,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_k,cv=cv,
                     scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
# LGBM
def black_box_function(learning_rate,n_estimators, min_child_samples,max_depth):
    modle = lgb.LGBMRegressor(learning_rate=learning_rate,
                                n_estimators=int(n_estimators),
                                min_child_samples=int(min_child_samples),
                                max_depth=int(max_depth),
                                random_state=42
                                )
    res = cross_val_score(modle,x_data,y_data_k,cv = cv,scoring='neg_mean_absolute_error')
    return np.mean(res)
pbounds= {'learning_rate':(0.0001,0.1),
        'n_estimators': (5, 1000),
          'min_child_samples':(18,40),
         'max_depth': (2, 30)}
optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=0,
    )
optimizer.maximize(
        init_points=25,
        n_iter=50,
    )
print(optimizer.max)
modle = lgb.LGBMRegressor(learning_rate=0.03187,
                          max_depth=13,
                          min_child_samples=19,
                          n_estimators=694 )
rmse = cross_val_score(modle,
                       x_data,y_data_k,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_k,cv=cv,
                     scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
# print('R2:',a,'RMSE:',b)

#-------------------------------------------------------------------------------
# feature set b pre_G
x_data = x_data_b
# lasso
modle = Lasso(
                alpha=0.0015,
                )
rmse = cross_val_score(modle,
                       x_data,y_data_g,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_g,
                     cv=cv,scoring='r2')
print(np.mean(r2),np.mean(rmse))
a.append(np.mean(r2))
b.append(np.mean(rmse))
# ridge
modle =Ridge(
            alpha=0.004
             )
rmse = cross_val_score(modle,
                       x_data,y_data_g,
                       cv=cv,scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_g,
                     cv=cv,scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
# KNN
modle = KNeighborsRegressor(
                            n_neighbors=5,
                            leaf_size=30,
                            n_jobs=1,
                    )
rmse = cross_val_score(modle,
                       x_data,y_data_g,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_g,cv=cv,
                     scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
# DT
modle = DecisionTreeRegressor(
                                max_depth=None,
                                min_samples_split=2, min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,
                              )
rmse = cross_val_score(modle,
                       x_data,y_data_g,
                       cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_g,cv=cv,
                     scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
# RF
modle = RandomForestRegressor(
                                n_estimators=120,
                                min_samples_split=37,
                                min_samples_leaf=9,
                                max_features=0.7941,  # float
                                max_depth=22,
                                random_state=42
                              )
rmse = cross_val_score(modle,
                       x_data,y_data_g,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_g,cv=cv,
                     scoring='r2')
# print(np.mean(r2),np.mean(rmse))
a.append(np.mean(r2))
b.append(np.mean(rmse))
#xgboost
def black_box_function(max_depth,learning_rate,n_estimators):
    modle = xgb.XGBRegressor(max_depth=int(max_depth),
                # gama = gama,
                learning_rate=learning_rate,
                n_estimators=int(n_estimators)
                                )
    res = cross_val_score(modle,x_data,y_data_g,cv = cv,scoring='r2')
    return np.mean(res)
pbounds= {
    'max_depth': (2, 30),
    # 'gama':(0,1),
    'learning_rate':(0,0.1),
    'n_estimators': (5, 1000),

         }
optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=42,
    )
optimizer.maximize(
        init_points=50,  #执行随机搜索的步数
        n_iter=50,   #执行贝叶斯优化的步数
    )
modle = xgb.XGBRegressor(
                            max_depth=2,
                            learning_rate=0.01079,
                            n_estimators=638,
                            n_jobs=1,
                            gamma=0,

                         )

rmse = cross_val_score(modle,
                       x_data,y_data_g,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_g,cv=cv,
                     scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
# LGBM
def black_box_function(learning_rate,n_estimators, min_child_samples,max_depth):
    modle = lgb.LGBMRegressor(learning_rate=learning_rate,
                                n_estimators=int(n_estimators),
                                min_child_samples=int(min_child_samples),
                                max_depth=int(max_depth),
                                random_state=42
                                )
    res = cross_val_score(modle,x_data,y_data_g,cv = cv,scoring='r2')
    return np.mean(res)
pbounds= {'learning_rate':(0.0001,0.1),
        'n_estimators': (5, 1000),
          'min_child_samples':(18,40),
         'max_depth': (2, 30)}
optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=0,
    )
optimizer.maximize(
        init_points=25,  #执行随机搜索的步数
        n_iter=50,   #执行贝叶斯优化的步数
    )
print(optimizer.max)
modle = lgb.LGBMRegressor(
    learning_rate=0.05764,
    max_depth=28,
    min_child_samples=25,
    n_estimators=669)
rmse = cross_val_score(modle,
                       x_data,y_data_g,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_g,cv=cv,
                     scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))



