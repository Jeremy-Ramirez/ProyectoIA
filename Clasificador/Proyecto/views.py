from django.shortcuts import render

from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


# Create your views here.
def index(request):
    context={'a':1}
    return render(request, 'index.html', context)

altura=100
anchura=100
modelo='modelo/modelo.h5'
pesos='modelo/pesos.h5'
cnn=load_model(modelo)
cnn.load_weights(pesos)


def predictImage(request):

    print(request.FILES['filePath']) #nos da el nombre del archivo 'molinillo.png'

    fileObjt=request.FILES['filePath'] #guardamos la imagen en nuestro directorio
    fs=FileSystemStorage() #permite almacenar objetos localmente para que puedan ser servidos como media en el desarrollo
    filePathName=fs.save(fileObjt.name, fileObjt) # guarda el nombre del archivo y el contenido archivo. 
    #Esta funcion almacena el contenido bajo el nombre asignado. Si existe ya el nombre lo modifica para q sea unico
    
    filePathName=fs.url(filePathName) #almacenamos la url del archivo, lo que nos sirve para servir la imagen
    testimage='.'+ filePathName #agregamos un punto por motivo se seguridad de path

    print(filePathName) #/media/nombrearchivo.jpg
    print(testimage) #./media/nombrearchivo.jpg
    #CODIGO 
    x=load_img(testimage, target_size=(anchura, altura))
    print(x) #Imagen, modo=rgb, size=100x100  
    x=img_to_array(x) 
    print(x)# convierte una instancia de imagen PIL a un arreglo de numpy que contiene pixeles
    x=np.expand_dims(x,axis=0)
    print(x) # en nuestro eje 0 (primera dimension), queremos añadir una dimension extra, para procesar nuestra información
    arreglo=cnn.predict(x) #llamamos a nuestra red y queremos predecir, regresa un arreglo de 2 dimensiones
    print(arreglo) #[1,0,0,0,0]
    resultado= arreglo[0] #solo necesitamos 1 dimension, la que trae la información
    print(resultado) # nos presenta el arreglo con valores de 0 y 1 como one hot encoding
    respuesta=np.argmax(resultado) #retorna el índice del máximo valor del arreglo
    print(respuesta)
    nombre=''
    audio=''
    if respuesta == 0:
      nombre= 'cacao'
      print('cacao')
      audio="/media/audios/cacao.mp3"
    elif respuesta == 1:
      nombre= 'metate'  
      print('metate')
      audio="/media/audios/metate.mp3"
    elif respuesta == 2:
      nombre='molinillo'
      print('molinillo')
      audio="/media/audios/molinillo.mp3"
    elif respuesta == 3:
      nombre='mortero'
      print('mortero')
      audio="/media/audios/mortero.mp3"
    elif respuesta == 4:
      nombre='silla con forma de U'
      print('sillau')
      audio="/media/audios/silla.mp3"
    


    context={'filePathName':filePathName,'nombre':nombre,'audio':audio}

    return  render(request, 'index.html', context)
