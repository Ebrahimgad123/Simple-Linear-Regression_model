import pandas as pd        #data#
import numpy as np       #معادلات#
import seaborn as sns    #رسم#
import matplotlib.pyplot as plt     #رسم#
from sklearn.model_selection import train_test_split     #train_test_split#
import sklearn.linear_model as LinerRegression     #LinerRegression#
from sklearn.metrics import r2_score         #accurcy#
from sklearn.metrics import mean_squared_error   #Error#

import os;
os.chdir(r"C:\Users\DELL\OneDrive\Desktop\simple linear regression model")

data=pd.read_csv("./Salary Data.csv")

print(data.info())

print(data.describe())

print(data.tail())

print(data.head())


# استكشاف وتحليل البيانات
sns.pairplot(data)
plt.show()

# تحضير البيانات
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
# print(Y)

# تقسيم البيانات
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# تدريب النموذج
model = LinerRegression.LinearRegression()
model.fit(X_train, Y_train)
relation =model.score(X_train,Y_train)
relationTest =model.score(X_test,Y_test)
# الرسمه الاولى بعد التوضيح
print("Relation between train x and train y ",relation)
print("Relation between test x and test y",relationTest)

plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,model.predict(X_train),color="blue")
plt.title('salary vs experience Year (training set)')
plt.xlabel('Year of experience ')
plt.ylabel('Salary')
plt.show()
 
Y_pred= model.predict(X_test)
plt.scatter(X_test,Y_test,color='yellow')
plt.plot(X_test,Y_pred,color="green")
plt.title('salary vs experience Year (testing set)')
plt.xlabel('Year of experience ')
plt.ylabel('Salary')
plt.show()




c = [i for i in range(len(Y_test))]
plt.plot(c, Y_test, color="r", linestyle="-", label="Actual")
plt.plot(c, Y_pred, color="b", linestyle="-", label="Predicted")
plt.xlabel('Index')
plt.ylabel('Salary')
plt.title('Salary Prediction')
plt.legend()
plt.show()


c = [i for i in range(len(Y_test))]
plt.plot(c, Y_test-Y_pred, color="b", linestyle="-", label="Error")
plt.xlabel('Index')
plt.ylabel('Error')
plt.title('Error value')
plt.legend()
plt.show()
# تقييم النموذج
accuracy = r2_score(Y_test, Y_pred)
print(accuracy)    #accuracy#
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)  #Error#
print(model.intercept_)   
print(model.coef_)   
YearOfExperience=4
Y_hat=9423 * YearOfExperience + 25321
print("Experience Year",YearOfExperience)
print("Salary",Y_hat)

















