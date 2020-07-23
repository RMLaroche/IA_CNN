#Test de Gitpod
from tensorflow import keras
from keras.datasets import mnist
#Imports for CNN
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical


#Telecharge et stock les données MNIST dans deux jeux de données :
#jeu de données d'apprentissage (train) qui contient les vecteurs (data) et leurs correspondance en int (res)
#jeu de données de test (test) qui contient les vecteurs (data) et leurs correspondance en int (res)
(train_data, train_res), (test_data, test_res) = mnist.load_data()

#Decoupe des données en entrée
train_data = train_data.reshape(60000,28,28,1)
test_data = test_data.reshape(10000,28,28,1)
#conversion des resultats attendus d'entiers vers tableau de correspondance
train_res = to_categorical(train_res)
test_res = to_categorical(test_res)

#Création du reseau
model = Sequential()

#Definition des couches
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#compilation du réseau
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Entrainement du réseau
model.fit(train_data, train_res,validation_data=(test_data, test_res), epochs=1, batch_size=200)

#Export du modèle du réseau
model_json = model.to_json()
with open("output"+".json", "w") as json_file:
    json_file.write(model_json)

#Export des poids
model.save_weights("output"+".h5")

#Test final et etablissement du score
scores = model.evaluate(test_data, test_res, verbose=0)
print("Score : ", (scores[1]*100))
print("Taux d'erreur :", (100-scores[1]*100))