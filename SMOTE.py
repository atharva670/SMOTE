#SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('train.csv')
le_y = LabelEncoder()
le_x = LabelEncoder()
y= le_y.fit_transform(df['Outlet_Type'])
X= le_x.fit_transform(df['Outlet_Size'])
X=pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
Xv=le_x.inverse_transform(X_train_res)
Yv=le_y.inverse_transform(y_train_res)
X1=pd.DataFrame(Xv,columns=['Outlet_Size'])
Y1=pd.DataFrame(Yv,columns=['Outlet_Type'])
df2=pd.concat([X1,Y1],axis=1)
df3=pd.concat([df,df2],axis=1)
print(df3)


