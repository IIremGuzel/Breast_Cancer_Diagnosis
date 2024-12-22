#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio


# In[3]:


data =pd.read_csv('/content/drive/MyDrive/data.txt')
data.head(5)


# In[4]:


col = data.columns
print(col)


# In[5]:


y = data.diagnosis                          
list = ['id','diagnosis']
x = data.drop(list,axis = 1 )
x.head()


# In[6]:


fig = px.histogram(y, x="diagnosis", color="diagnosis",width=700,height=500)
fig.show()


# In[7]:


import time
sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,0:15]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
tic = time.time()
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=90)


# In[8]:


data = pd.concat([y,data_n_2.iloc[:,15:31]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=90)


# In[9]:


corr_matrix = x.corr() # Korelasyon matrisi

plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Feature Correlation Matrix with Encoded Diagnosis')
plt.show()


# In[10]:


print(x.dtypes)


# In[11]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)


print("Classes: ", label_encoder.classes_) # Label Encoder
print("Encoded Labels: ", encoded_y)


# In[12]:



# Hedef değişken ile özellikler arasındaki korelasyonları hesaplama
corr_with_target = x.apply(lambda feature: feature.corr(pd.Series(encoded_y)))

# Korelasyonları görselleştirme
plt.figure(figsize=(10, 8))
corr_with_target.sort_values(ascending=False).plot(kind='bar')
plt.title('Correlation of Features with Target Variable (Diagnosis)')
plt.xlabel('Features')
plt.ylabel('Correlation with Target')
plt.xticks(rotation=90)
plt.show()


# In[14]:


# RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=10000)

# RFE ile özellik seçimi
rfe = RFE(model, n_features_to_select=10)  # Seçmek istediğim öellik sayısı
fit = rfe.fit(X_train, y_train)

# Seçilen özellikleri gösterme
selected_features = X_train.columns[fit.support_]
print("Selected Features (RFE): ", selected_features)

# Özelliklerin sıralanmış şekilde önem dereceleri
ranking = pd.Series(fit.ranking_, index=X_train.columns).sort_values()
print("Feature Rankings (RFE): \n", ranking)


# In[39]:


#  LASSO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Lasso, LassoCV
# Teşhis etiketlerini sayısal değerlere dönüştürme
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# LassoCV 
lasso_cv = LassoCV(alphas=[0.01, 0.1, 1, 10], cv=5) #  en iyi alpha değerini bulma
lasso_cv.fit(X_scaled, encoded_y)

best_alpha = lasso_cv.alpha_
print(f"Best alpha value: {best_alpha}")

# fit etme
lasso_best = Lasso(alpha=best_alpha)
lasso_best.fit(X_scaled, encoded_y)

# Özelliklerin önem dereceleri
importance_best = np.abs(lasso_best.coef_)
selected_features_lasso_best = x.columns[importance_best > 0]

print("Selected Features (Lasso with best alpha): ", selected_features_lasso_best)

# Özelliklerin sıfır olmayan katsayıları
importance_series_best = pd.Series(importance_best, index=x.columns).sort_values(ascending=False)
print("Feature Importance (Lasso with best alpha): \n", importance_series_best)


# In[40]:


print("x Shape: ", x.shape)
print("y Shape: ", y.shape)


# In[41]:


del list


# In[42]:


# Seçilen özellikler
rfe_features = ['radius_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
                'perimeter_se', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
                'concave points_worst', 'symmetry_worst']

lasso_features = ['texture_mean', 'concave points_mean', 'fractal_dimension_mean', 'radius_se',
                  'smoothness_se', 'concavity_se', 'radius_worst', 'texture_worst',
                  'smoothness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst']

# Ortak özellikler
common_features = list(set(rfe_features) & set(lasso_features))

# Özelliklerin veri kümesine verme
X_rfe = x[rfe_features]
X_lasso = x[lasso_features]
X_common = x[common_features]

from sklearn.model_selection import train_test_split

X_rfe_train, X_rfe_test, y_train, y_test = train_test_split(X_rfe, y, test_size=0.3, random_state=42)
X_lasso_train, X_lasso_test, _, _ = train_test_split(X_lasso, y, test_size=0.3, random_state=42)
X_common_train, X_common_test, _, _ = train_test_split(X_common, y, test_size=0.3, random_state=42)


# In[43]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Model oluşturma + değerlendirme fonksiyonu
def evaluate_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# RFE
accuracy_rfe = evaluate_model(X_rfe_train, X_rfe_test, y_train, y_test)
print(f"RFE Features Accuracy: {accuracy_rfe:.4f}")

# Lasso 
accuracy_lasso = evaluate_model(X_lasso_train, X_lasso_test, y_train, y_test)
print(f"Lasso Features Accuracy: {accuracy_lasso:.4f}")

# Ortak özellikler 
accuracy_common = evaluate_model(X_common_train, X_common_test, y_train, y_test)
print(f"Common Features Accuracy: {accuracy_common:.4f}")


# In[44]:



# Özelliklerin veri kümesine uygulanması
X_rfe = x[rfe_features]
X_lasso = x[lasso_features]

# Korelasyon matrislerini hesaplama
corr_matrix_rfe = X_rfe.corr()
corr_matrix_lasso = X_lasso.corr()

# RFE özellikleri için korelasyon matrisini görselleştirme
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix_rfe, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Korelasyon Matrisi - RFE Seçilen Özellikler')
plt.show()


# In[45]:


# Lasso için korelasyon matrisini görselleştirme
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix_lasso, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Korelasyon Matrisi - Lasso Seçilen Özellikler')
plt.show()


# In[48]:


from sklearn.preprocessing import LabelEncoder

#le = LabelEncoder()
#y_lasso_encoded = le.fit_transform(y_encoded)

#print("Dönüştürülmüş etiketlerin ilk birkaç değeri: ", y_lasso_encoded[:10])
#print("Benzersiz etiketler: ", np.unique(y_lasso_encoded))


# In[49]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X_lasso, encoded_y, test_size=0.3, random_state=42)

# Normalize etme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Veriyi 3D tensöre dönüştürme (samples, steps, features) - CNN için
X_train_scaled = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
X_test_scaled = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)


# In[50]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Sonuç görselleştirme
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.title('Model Doğruluğu')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




