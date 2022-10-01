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
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
dataset = pd.read_excel('datasets.xlsx',index_col=0)
height,width = dataset.shape

column = dataset.columns
# print(bb)
x_data_o = dataset.iloc[:,2:13]#Original feature set
x_data_a = dataset.iloc[:,[13,14,15,16,17,7,8,11,12]]#feature set a
x_data_b = dataset.iloc[:,[18,19,20,21,22,6,7,10]]#feature set b
x_data = x_data_o
y_data_k = dataset.iloc[:,1]
y_data_g = dataset.iloc[:,2]
print(x_data.columns)
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
                alpha=1,
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
            alpha=1.0
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
modle = RandomForestRegressor(
                                    n_estimators=10,
                                    max_depth=None,
                                    min_samples_split=2,
                                    min_samples_leaf=1,

                                      )
rmse = cross_val_score(modle,
                       x_data,y_data_k,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_k,cv=cv,
                     scoring='r2')
print(np.mean(r2),np.mean(rmse))
a.append(np.mean(r2))
b.append(np.mean(rmse))
#xgboost
modle = xgb.XGBRegressor(
                            max_depth=3,
                            learning_rate=0.1,
                            n_estimators=100,
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
modle = lgb.LGBMRegressor()
rmse = cross_val_score(modle,
                       x_data,y_data_k,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_k,cv=cv,
                     scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
print('R2:',a,'RMSE:',b)
#------------------------------------------------------------------------------------
# Original feature set pre_G
# lasso
modle = Lasso(
                alpha=1,
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
            alpha=1.0
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
modle = xgb.XGBRegressor(
                            max_depth=3,
                            learning_rate=0.1,
                            n_estimators=100,
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
modle = lgb.LGBMRegressor()
rmse = cross_val_score(modle,
                       x_data,y_data_g,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_g,cv=cv,
                     scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
print('R2:',a,'RMSE:',b)
#----------------Original feature SymbolicRegressor
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
print(pr)
dot_data = modle._program.export_graphviz()
graph = graphviz.Source(dot_data)
# print(graph)
graph.view()
#------------------------------------------------------------------------------
# feature set a pre_k
x_data = x_data_a
# lasso
modle = Lasso(
                alpha=1,
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
            alpha=1.0
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
modle = RandomForestRegressor(
                                    n_estimators=10,
                                    max_depth=None,
                                    min_samples_split=2,
                                    min_samples_leaf=1,

                                      )
rmse = cross_val_score(modle,
                       x_data,y_data_k,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_k,cv=cv,
                     scoring='r2')
print(np.mean(r2),np.mean(rmse))
a.append(np.mean(r2))
b.append(np.mean(rmse))
#xgboost
modle = xgb.XGBRegressor(
                            max_depth=3,
                            learning_rate=0.1,
                            n_estimators=100,
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
modle = lgb.LGBMRegressor()
rmse = cross_val_score(modle,
                       x_data,y_data_k,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_k,cv=cv,
                     scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))
print('R2:',a,'RMSE:',b)

#-------------------------------------------------------------------------------
# feature set b pre_G
x_data = x_data_b
# lasso
modle = Lasso(
                alpha=1,
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
            alpha=1.0
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
modle = xgb.XGBRegressor(
                            max_depth=3,
                            learning_rate=0.1,
                            n_estimators=100,
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
modle = lgb.LGBMRegressor()
rmse = cross_val_score(modle,
                       x_data,y_data_g,cv=cv,
                       scoring='neg_mean_absolute_error')
r2 = cross_val_score(modle,
                     x_data,y_data_g,cv=cv,
                     scoring='r2')
a.append(np.mean(r2))
b.append(np.mean(rmse))


