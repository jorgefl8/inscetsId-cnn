import tensorflow as tf # utilizado para el aprendizaje automático
import matplotlib.pyplot as plt # utilizado para la visualización de datos
import os # utilizado para la manipulación de archivos
import random # utilizado para la generación de números aleatorios
import argparse
import cv2 # utilizado para el procesamiento de imágenes

parser = argparse.ArgumentParser(description='Train model with specified epochs.')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
args = parser.parse_args()

if not os.path.exists('./charts'):
        os.makedirs('./charts')

if not os.path.exists('./models'):
        os.makedirs('./models')
             
#Creamos cojunto de entrenamiento 
training_set = tf.keras.utils.image_dataset_from_directory( #para cargar y preprocesar las imágenes directamente desde un directorio
    './kaggle/insects/train', #ruta directorio con las imágenes
    labels="inferred", #las etiquetas se infieren de la estructura de subdirectorios:butterfly, etc...
    label_mode="categorical", #definimos que las etiquetas son categorias
    color_mode="rgb",#imágenes en color	
    batch_size=32, #número imágenes procesadas en cada lote
    image_size=(64, 64), #tamaño a redimensionar las imágenes
    shuffle=True, #barajamos las imágenes para que el modelo no aprneda el orden
    interpolation="bilinear", #metodo de interpolación para redimensionar las imágenes
)

#Creamos conjunto de validación
validation_set = tf.keras.utils.image_dataset_from_directory(
    './kaggle/insects/validation', 
    labels="inferred", 
    label_mode="categorical", 
    color_mode="rgb",	
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    interpolation="bilinear"
)

# Ruta del directorio que contiene el conjunto de datos de entrenamiento 
dataset_path = './kaggle/insects/train'

# Cogemos los nombres de las clases de los subdirectorios(butterfly, etc...)
class_names = os.listdir(dataset_path)

####### Para mostrar las imagenes en formato original BGR #######
num_rows = 5
num_cols = (len(class_names) + num_rows - 1) // num_rows

# Creamos una figura y subplots para mostrar las imágenes
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

# Iteramos sobre cada clase y seleccionamos una imagen aleatoria para mostrar
for i, class_name in enumerate(class_names):
    class_dir = os.path.join(dataset_path, class_name)
    image_files = os.listdir(class_dir)
    random_image = random.choice(image_files)
    image_path = os.path.join(class_dir, random_image)
    image = cv2.imread(image_path)
    axes[i].imshow(image)
    axes[i].set_title(class_name)
    axes[i].axis('off')

# Ocultamos los subgráficos restantes si no se usan
for ax in axes[len(class_names):]:
    ax.axis('off')

# Ajustamos el espaciado entre subgráficos y mostramos la figura
plt.tight_layout()
plt.savefig('./charts/insects1.png')
plt.show()

####### Para mostrar las imagenes en color RGB #######
num_rows = 5
num_cols = (len(class_names) + num_rows - 1) // num_rows

# Creamos una figura y subplots para mostrar las imágenes
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

# Iteramos sobre cada clase y seleccionamos una imagen aleatoria para mostrar
for i, class_name in enumerate(class_names):
    class_dir = os.path.join(dataset_path, class_name)
    image_files = os.listdir(class_dir)
    random_image = random.choice(image_files)
    image_path = os.path.join(class_dir, random_image)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[i].imshow(image)
    axes[i].set_title(class_name)
    axes[i].axis('off')

# Ocultamos los subgráficos restantes si no se usan
for ax in axes[len(class_names):]:
    ax.axis('off')

# Ajustamos el espaciado entre subgráficos y mostramos la figura
plt.tight_layout()
plt.savefig('./charts/insects2.png')
plt.show()

####### Definimos red neuronal convolucional (CNN) utilizando TensorFlow y Keras #######
cnn = tf.keras.models.Sequential() # Creamos tipo de modelo en Keras donde las capas se apilan linealmente

#Capa conv2D con 32 filtros, tam kernel 3x3, padding same para mantener tam de la imagen,
#F.activacion = Relu y la forma en la que entran las imagenes 64x64 pixeles y 3 canales RGB
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[64,64,3]))
#Capa conv2D con 32 filtros, tam kernel 3x3, F.activacion = Relu
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
#Capa MaxPool2D con max pooling(para reducir dimensaiones espaciales(ancho y alto)de la imagen)
#2x2 y stride 2
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
# tecnica para regular el sobreajuste durante el entrenamiento apaga aleatoriamente un porcentaje
# de unidades durante el entrenamiento, y evita que aprenda demasiado ciertas características 
cnn.add(tf.keras.layers.Dropout(0.3))
# Agrega una capa de convolución con 64 filtros, cada kernel de 3x3, relleno 'same' (mismas dimensiones) 
# y función de activación ReLU.
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
# Agregamos otra capa, igual que la anterior pero sin relleno
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
# Capa MaxPool2D con max pooling 2x2 y stride 2
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
# Agregamos otra capa de dropout con tasa 0.25
cnn.add(tf.keras.layers.Dropout(0.3))
# Agregamos capa de aplanamiento para convertir los mapas de características 2D en un vector 1D
cnn.add(tf.keras.layers.Flatten())
regularizer = tf.keras.regularizers.l2(0.001)  # Regularización L2
cnn.add(tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=regularizer))  # Reducción de neuronas a 256 y regularización

# Agregamos otra capa de Dropout final con tasa 0.5 - prevenir sobreajuste
cnn.add(tf.keras.layers.Dropout(0.5)) 
     
# Capa de Salida, con 5 unidades finales y función de activación softmax, que dará la probabilidad de cada clase
cnn.add(tf.keras.layers.Dense(units=5,activation='softmax'))

# Compilamos el modelo con optimizador Adam, función de pérdida entropía cruzada categórica y métrica de precisión (accuracy)
# Adam: algoritmo de optimizacion para entrenar modelos, que ajusta automaticamente ciertas caracteristicas del  descenso de gradiente estocastico
cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Se imprime en consola un resumen del modelo, con el tipo de capa, parametros y forma de la salida
cnn.summary()


# Entrenamos el modelo con el conjunto de entrenamiento y validación, durante 5 épocas
training_history = cnn.fit(x=training_set,validation_data=validation_set,epochs=args.epochs)

# Evaluamos el rendimiento del modelo con el conjunto de entrenamiento y validación
train_loss, train_acc = cnn.evaluate(training_set)
print("Training set Accuracy: {} %".format(train_acc*100))
val_loss, val_acc = cnn.evaluate(validation_set)
print("Validation set Accuracy: {} %".format(val_acc*100))

# Guardamos el modelo entrenado 
cnn.save('./models/trained_model_insects.keras')



# Creamos una grafica para visualizar la precision del conjunto de entrenamiento
epochs = [i for i in range(1, len(training_history.history['accuracy']) + 1)]
plt.plot(epochs,training_history.history['accuracy'],color='red')
plt.xlabel('Number of Epochs')
plt.ylabel('Training Accuracy')
plt.title('Visualization of Training Accuracy Result')
plt.savefig('./charts/accuracytraining.png')
#plt.show()

# Creamos una grafica para visualizar la precision del conjunto de validacion
plt.plot(epochs,training_history.history['val_accuracy'],color='blue')
plt.xlabel('Number of Epochs')
plt.ylabel('Validation Accuracy')
plt.title('Visualization of Validation Accuracy Result')
plt.savefig('./charts/accuracyvalidation.png')
#plt.show()

# Cargamos y preprocesamos las imágenes directamente desde un directorio
test = tf.keras.utils.image_dataset_from_directory( 
    './kaggle/insects/test', #ruta directorio con las imágenes
    labels="inferred", #las etiquetas tomadas de la estructura del subdirectorio
    label_mode="categorical", #definimos que las etiquetas son categorias
    color_mode="rgb",#imágenes en color	
    batch_size=32, #número imágenes procesadas en cada lote
    image_size=(64, 64), #tamaño a redimensionar las imágenes
    shuffle=True, #barajamos las imágenes para que el modelo no aprenda el orden
    interpolation="bilinear", #metodo de interpolación para redimensionar las imágenes
)

# Evaluamos el rendimiento del modelo con el conjunto de prueba
test_loss,test_acc = cnn.evaluate(test) #Devuelve la tupla con la perdida y la precision calculada sobre el conjunto de prueba
print("Test set Accuracy: {} %".format(test_acc*100))
