import tensorflow as tf
from tensorflow.keras import Model, layers  # <--- Agregamos 'layers' aquí
import numpy as np

# Importamos nuestros módulos anteriores
from modelo_esteganografia import crear_encoder, crear_decoder

def construir_autoencoder(encoder, decoder):
    # Ahora 'layers' ya está reconocido
    entradas_globales = layers.Input(shape=(128, 128, 4))
    imagen_oculta = encoder(entradas_globales)
    bits_extraidos = decoder(imagen_oculta)
    
    # Keras necesita saber cómo se llaman estas salidas para el 'loss dict'
    autoencoder = Model(
        inputs=entradas_globales, 
        outputs=[imagen_oculta, bits_extraidos], 
        name="Autoencoder_Completo"
    )
    return autoencoder

# ... (El resto de tu código igual)

# =============================================================================
# PRUEBA DE UNIÓN DEL MODELO
# =============================================================================
if __name__ == "__main__":
    print("--- Inicializando Redes Individuales ---")
    enc = crear_encoder()
    dec = crear_decoder()
    
    print("\n--- Ensamblando Autoencoder Global ---")
    modelo_completo = construir_autoencoder(enc, dec)
    
    modelo_completo.compile(
        optimizer='adam',
        loss={
            'Encoder_Esteganografia': 'mse',
            'Decoder_Esteganografia': 'binary_crossentropy'
        },
        loss_weights={
            'Encoder_Esteganografia': 1.0,
            'Decoder_Esteganografia': 1.0
        }
    )
    
    # =============================================================================
    # SIMULACIÓN DE DATOS PARA ENTRENAMIENTO (¡NUEVO!)
    # =============================================================================
    print("\n--- Generando datos sintéticos para prueba de entrenamiento ---")
    num_muestras = 16  # Cantidad pequeña de imágenes para probar que no explote la RAM
    
    # Simulamos las imágenes originales (X_imagenes) -> (16, 128, 128, 3)
    X_imagenes = np.random.rand(num_muestras, 128, 128, 3).astype(np.float32)
    
    # Simulamos los vectores de bits (X_bits) -> (16, 256)
    X_bits = np.random.randint(0, 2, size=(num_muestras, 256)).astype(np.float32)
    
    # Para la ENTRADA de la red necesitamos los 4 canales combinados (como en el paso anterior)
    # Expandimos los bits para que tengan forma (16, 128, 128, 1)
    X_bits_expandidos = np.zeros((num_muestras, 128, 128, 1), dtype=np.float32)
    for i in range(num_muestras):
        matriz_plana = np.resize(X_bits[i], (128 * 128))
        X_bits_expandidos[i] = matriz_plana.reshape((128, 128, 1))
        
    # Concatenamos en el eje de canales para obtener el set de entrada (16, 128, 128, 4)
    X_entrada_global = np.concatenate([X_imagenes, X_bits_expandidos], axis=3)
    
    print(f"Forma de la matriz de entrada global: {X_entrada_global.shape}")
    print("Iniciando una época de entrenamiento de prueba...")
    
    # Entrenamos con fit. Pasamos la entrada y un diccionario con los objetivos de salida.
    modelo_completo.fit(
        x=X_entrada_global,
        y={
            'Encoder_Esteganografia': X_imagenes,  # El encoder debe imitar la foto original
            'Decoder_Esteganografia': X_bits       # El decoder debe recuperar los bits originales
        },
        epochs=1,
        batch_size=4
    )
print("\n¡Ciclo de entrenamiento completado con éxito!")
    
 # =============================================================================
    # GUARDAR EL MODELO ENTRENADO (¡NUEVO!)
    # =============================================================================
# Cámbialo a esto:
ruta_pesos = "pesos_esteganografia.weights.h5" 
modelo_completo.save_weights(ruta_pesos)
print(f"Pesos guardados exitosamente en: {ruta_pesos}")