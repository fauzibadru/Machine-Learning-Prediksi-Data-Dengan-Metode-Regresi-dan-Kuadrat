import numpy as np
from sklearn.linear_model import LinearRegression

#Database
# x = Data, y = Target
x = [[1],[3],[6],[9],[12],[15],[18],[21],[24],[27]]
y = [1, 6, 12, 18, 24, 30, 36, 42, 48, 54]

regr = LinearRegression().fit(x,y)
regr.score(x, y)

#Data uji
predict = np.array([[15]])

#Menampilkan data prediksi
print ("M Fauzi Badru Z")
print ("Prediksi")
print ("Input = ", predict)
print ("Output = ", regr.predict(predict))
