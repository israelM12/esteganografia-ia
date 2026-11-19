import tensorflow as tf
from tensorflow.keras import layers, Model

def crear_encoder(input_shape=(128, 128, 4)):
    """
    Encoder: Recibe (128, 128, 4) y genera imagen (128, 128, 3).
    """
    entradas = layers.Input(shape=input_shape)
    
    # Extraer características
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(entradas)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    
    # Salida: RGB normalizado [0, 1]
    salida_imagen_oculta = layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid', name="Encoder_Esteganografia")(x)
    
    modelo_encoder = Model(inputs=entradas, outputs=salida_imagen_oculta, name="Encoder_Esteganografia")
    return modelo_encoder

def crear_decoder(input_shape=(128, 128, 3)):
    """
    Decoder: Recibe imagen (128, 128, 3) y extrae mensaje (256 bits).
    Optimizado con GlobalAveragePooling2D para reducir drásticamente el peso del archivo.
    """
    entradas = layers.Input(shape=input_shape)
    
    # Detectar patrones
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(entradas)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    
    # Reducción eficiente (evita el gigantismo del archivo)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # Salida: Vector de 256 bits con nombre explícito para evitar errores en entrenamiento
    salida_bits = layers.Dense(256, activation='sigmoid', name="Decoder_Esteganografia")(x)
    
    modelo_decoder = Model(inputs=entradas, outputs=salida_bits, name="Decoder_Esteganografia")
    return modelo_decoder

# Test de estructura
if __name__ == "__main__":
    print("=== ARQUITECTURA AUTOENCODER OPTIMIZADA ===")
    encoder = crear_encoder()
    decoder = crear_decoder()
    encoder.summary()
    print("\n" + "="*50 + "\n")
    decoder.summary()