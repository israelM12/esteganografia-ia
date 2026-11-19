# Esteganografía con imágenes e IA

Proyecto de la materia Inteligencia Artificial 2. La idea es ocultar un mensaje de texto dentro de una imagen y después recuperarlo. Lo trabajé de dos formas: una clásica, cambiando los bits de la imagen, y otra con una red neuronal hecha en TensorFlow.

## Qué hace

- Escondes un texto dentro de una imagen y guardas la imagen resultante.
- Abres esa imagen y recuperas el texto escondido.
- Todo desde una interfaz sencilla hecha en Tkinter.

## Las dos formas que probé

**1. Método clásico (LSB).** Cambio el último bit de cada pixel para guardar ahí el mensaje. A simple vista la imagen se ve igual, pero el texto queda adentro. Este método es confiable y recupera el mensaje completo.

**2. Red neuronal (autoencoder).** Entrené un modelo en TensorFlow/Keras con dos partes:

- Un encoder con capas convolucionales que recibe la imagen junto con el mensaje y genera una imagen muy parecida a la original.
- Un decoder que toma esa imagen y trata de recuperar los 256 bits del mensaje.

Esta parte es experimental. La entrené como prueba de concepto con pocos datos, así que sirve para mostrar cómo funcionaría la idea, no como un modelo listo para usar en serio.

## Cómo está armado

- `preprocesamiento.py` convierte el texto a bits y de bits de vuelta a texto.
- `procesamiento_imagenes.py` prepara la imagen (128x128, RGB, normalizada) y la junta con el mensaje.
- `modelo_esteganografia.py` define el encoder y el decoder.
- `entrenamiento.py` arma y entrena el autoencoder y guarda los pesos.
- `interfaz_esteganografia.py` es la interfaz para ocultar y recuperar mensajes.

## Tecnologías

Python, TensorFlow/Keras, OpenCV, NumPy, Pillow y Tkinter.

## Cómo correrlo

```bash
pip install tensorflow opencv-python numpy pillow
python interfaz_esteganografia.py
```
