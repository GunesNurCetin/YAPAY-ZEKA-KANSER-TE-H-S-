# -*- coding: utf-8 -*-
#Güneş Nur ÇETİN

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score


# VERI SETININ OKUNMASI
veriSeti = pd.read_csv("dataR2.csv")



# VERIYI ANLAMA VE HAZIRLAMA
veriSeti = veriSeti.rename(columns={"Classification": "Karar"})
veriSeti["Karar"].value_counts()
veriSeti["Karar"] = np.where(veriSeti["Karar"]==1, "Sağlıklı", "Kanser")
veriSeti["Karar"].value_counts()

veriSeti.Karar = veriSeti.Karar.astype("category")

pd.set_option("display.max_columns", 20)
veriSeti.describe(include="all")

veriSeti.dtypes

# Eksik veri kontrolu
print(veriSeti.isnull().sum())

# Nitelikler arasinda korelasyonun incelenmesi
my_cors = pd.DataFrame(np.corrcoef(veriSeti.iloc[:,0:9], rowvar=False).round(2), columns=veriSeti.columns[0:9])
my_cors.index=veriSeti.columns[0:9]

# Korelasyon Isı haritası
sns.heatmap(
    my_cors, 
    annot = True, 
    square=True,
    cmap=sns.color_palette("flare", as_cmap=True))

# Egitim ve test veri setinin olusturulmasi
X_train, X_test, y_train, y_test = train_test_split(veriSeti.iloc[:,0:9], veriSeti.iloc[:,9], test_size=0.3, random_state=1)

# Veri normalizasyonu
scaler = MinMaxScaler()
X_train_n = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_n = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

X_train.describe()
X_train_n.describe()

# MODELLEME
# knn Siniflandiricisinin Olusturulmasi
knn_modeli = KNeighborsClassifier(n_neighbors = 5, metric="euclidean")
knn_modeli.fit(X_train_n, y_train)

# knn Siniflandirici Tahminlerinin Elde Edilmesi
y_tahmin = knn_modeli.predict(X_test_n)

print("k-NN Modeli Tahminleri:", y_tahmin[0:5])
print("Gerçek Değerler:", np.array(y_test[0:5]))


# MODEL PERFORMANS DEGERLENDIRMESI

# I. YOL
y_test.value_counts()
my_cm = confusion_matrix(y_true = y_test, y_pred = y_tahmin, labels=["Sağlıklı","Kanser"])
my_cm


my_cm_p = ConfusionMatrixDisplay(my_cm,  display_labels=["Sağlıklı","Kanser"])
my_cm_p.plot()



tn, fp, fn, tp = my_cm.ravel()

print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

# Dogruluk (accuracy)
dogruluk = (tp+tn)/(tp+fp+fn+tn)
# Hata orani (error rate)
hata = 1 - dogruluk
# Duyarlilik (sensitivity)
duyarlilik = tp/(tp+fn)
# Belirleyicilik (specificity)
belirleyicilik = tn/(tn+fp)
# False Negative Rate
FNR = 1 - duyarlilik
# False Positive Rate
FPR = 1 - belirleyicilik
# Pozitif Ongoru Degeri / Kesinlik (Positive Predictive Value / Precision)
kesinlik = tp/(tp+fp)
# Negatif Ongoru Degeri (Negative Predictive Value)
NPV = tn/(tn+fn)
# F-olcusu (F-measure)
FOlcusu = (2*duyarlilik*kesinlik)/(duyarlilik+kesinlik)

print("Doğruluk (Accuracy) = ", dogruluk)
print("Hata (Error Rate) = ", hata)
print("Duyarlılık (Sensitivity) = ", duyarlilik)
print("Belirleyicilik (Specificity) = ", belirleyicilik)
print("False Negative Rate = ", FNR)
print("False Positive Rate = ", FPR)
print("Kesinlik (Positive Predicted Value / Precision) = ", kesinlik)
print("Negatif Öngörü Değeri (Negative Predicted Value) = ", NPV)
print("F-Ölçüsü (F-measure) = ", FOlcusu)


# II. YOL
rapor = classification_report(y_true = y_test, y_pred = y_tahmin, labels=["Sağlıklı","Kanser"])
print(rapor)


# Sağlıklı sınıfı pozitif sınıf olarak kabul edilirse
my_cm = confusion_matrix(y_true = y_test, y_pred = y_tahmin, labels=["Kanser", "Sağlıklı"])
my_cm

tn, fp, fn, tp = my_cm.ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)


# En İyi k Komşu Sayısının Belirlenmesi
dogruluk = []
fOlcusu = []
k = range(2,21)
for i in k:
    knn_modeli = KNeighborsClassifier(n_neighbors = i, metric="euclidean")
    knn_modeli.fit(X_train_n, y_train)
    y_tahmin = knn_modeli.predict(X_test_n)
    dgrlk = accuracy_score(y_test, y_tahmin).round(4)
    fOlc = f1_score(y_test, y_tahmin, average='binary', pos_label="Kanser").round(4)
    dogruluk.append(dgrlk)
    fOlcusu.append(fOlc)

plt.plot(k, dogruluk, 'bx-')
plt.xticks(k)
plt.title("k-NN Model Performansı")
plt.xlabel("k Komşu Sayısı")
plt.ylabel("Doğruluk")
plt.show()

plt.plot(k, fOlcusu, 'bx-')
plt.xticks(k)
plt.title("k-NN Model Performansı")
plt.xlabel("k Komşu Sayısı")
plt.ylabel("F-Ölçüsü")
plt.show()

np.round(dogruluk,3)
np.round(fOlcusu,3)