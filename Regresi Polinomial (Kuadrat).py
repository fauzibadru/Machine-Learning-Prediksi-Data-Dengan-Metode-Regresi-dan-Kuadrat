from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np

#Database
# x = Data, y = Target
x = [[3], [6], [9], [12], [15], [18], [21], [24], [27], [30]]
y = [9, 36, 81, 144, 225, 324, 441, 576, 729, 900]

#Data uji
predict = np.array([[12]])
poly = PolynomialFeatures (degree=2)
x_ = poly.fit_transform(x)
predict_ = poly.fit_transform(predict)
regr = linear_model.LinearRegression()
regr.fit(x_,y)

#Menampilkan data prediksi
print ("Prediksi")
print ("Input = ", predict)
print ("Output = ", regr.predict (predict_))
