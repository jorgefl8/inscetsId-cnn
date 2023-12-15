import argparse #Biblioteca para crear interfaces de línea de comandos
import numpy as np
import tensorflow as tf  #Bibliotecas para construir y entrenar modelos de aprendizaje profundo
import matplotlib.pyplot as plt
import cv2 #Biblioteca para el procesamiento de imágenes


#Difinimos el argumento image_path que representa la ruta de la imagen que se va a clasificar
parser = argparse.ArgumentParser(description='Clasificar una imagen')
parser.add_argument('image_path', type=str, help='Ruta de la imagen a clasificar')
args = parser.parse_args()


#Cargamos el conjunto de datos de prueba desde un directorio
test_set = tf.keras.utils.image_dataset_from_directory(
    './kaggle/insects/test', 
    labels="inferred", 
    label_mode="categorical", 
    color_mode="rgb",	
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    interpolation="bilinear"
)

#Cargamos el modelo CNN preentrenado	
cnn = tf.keras.models.load_model('./models/trained_model_insects.keras')


#image_path = './kaggle/insects/test/Butterfly/google0.jpg'
image_path = args.image_path

img = cv2.imread(image_path) #Lee la imagen de prueba en formato BGR(default)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #lo conviertimos a formato RGB

# Mostramos la imagen de prueba
plt.imshow(img)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.show()

image = tf.keras.preprocessing.image.load_img(image_path,target_size=(64,64)) #cargamos la imagen de prueba
input_arr = tf.keras.preprocessing.image.img_to_array(image) 
input_arr = np.array([input_arr])  #la convertimos a un array para que sea compatible con la entrada del modelo

predictions = cnn.predict(input_arr) #realizamos predicciones 
result_index = np.argmax(predictions) 
print(result_index) #imprimimos el indice de la clase con mayor probabilidad

print("It's a {}".format(test_set.class_names[result_index]))
