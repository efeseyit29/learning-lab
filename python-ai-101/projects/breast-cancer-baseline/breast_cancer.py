from sklearn.model_selection import cross_val_score 
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Veriyi yükle: data, dict gibi davranan bir obje (data.data, data.target, data.feature_names vs. içerir)
data = load_breast_cancer()      

X = data.data   # Özellikler: 569 hasta x 30 ölçüm (radius, texture, worst radius vs.)
y = data.target # Etiketler: 0 = malignant, 1 = benign (569 elemanlı dizi)

# İlk denemem (dengesiz sınıf ağırlıklı model) - artık kullanmıyorum, kıyaslama için saklıyorum
# classifier = DecisionTreeClassifier(max_depth=2, min_samples_split=5, min_samples_leaf=2)
# classifier.fit(X, y)
# y_pred = cross_val_predict(classifier, X, y, cv=5)

# Dengeli model: class_weight='balanced' malignant'ı kaçırmayı daha "pahalı" hale getiriyor
classifier_balanced = DecisionTreeClassifier(max_depth=2, min_samples_split=5, min_samples_leaf=2, class_weight='balanced')

# 5-fold cross-validation: veriyi 5 parçaya böl, 5 kez "4 parça eğit, 1 parça test" yap, 5 accuracy skoru döndür
cross_val_skorlar = cross_val_score(classifier_balanced, X, y, cv=5)
print(f"5-Katlamalı Çapraz Doğrulama Skorları {cross_val_skorlar}")
print(f"Ortalama Doğruluk: {cross_val_skorlar.mean():.2f}")

# cross_val_predict: aynı 5-fold mantığı ama skor yerine her hasta için tahmin edilen etiketi topluyor
y_pred_balanced = cross_val_predict(classifier_balanced, X, y , cv=5)    

# Gerçek (y) ile tahmin (y_pred_balanced) karşılaştırılıp 2x2 confusion matrix üretiliyor
cm = confusion_matrix(y, y_pred_balanced) 

# Matrisi görselleştir: satır=gerçek sınıf, sütun=tahmin edilen sınıf
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot(cmap='Blues')
plt.show()


# Final model: tüm veriyle eğitilmiş tek bir ağaç (artık cross-validation yok, hangi feature'lara baktığını görmek istiyoruz)
classifier_balanced.fit(X, y)

# Hangi feature'lar ağacın kararlarında kullanılmış, ne kadar etkili olmuş (0 = hiç kullanılmamış)
for name, importance in zip(data.feature_names, classifier_balanced.feature_importances_):
    if importance > 0:
        print(f"{name}: {importance:.3f}")
