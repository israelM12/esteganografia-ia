import numpy as np

def texto_a_bits(texto, max_caracteres=32):
    """
    Toma una cadena de texto y la convierte en un vector de NumPy 
    de ceros y unos con un tamaño fijo de 256 bits.
    """
    texto = texto[:max_caracteres]
    lista_bits = []
    for letra in texto:
        binario_letra = f"{ord(letra):08b}"
        for bit in binario_letra:
            lista_bits.append(int(bit))
            
    bits_totales_requeridos = max_caracteres * 8  # 256
    bits_faltantes = bits_totales_requeridos - len(lista_bits)
    
    if bits_faltantes > 0:
        lista_bits.extend([0] * bits_faltantes)
        
    return np.array(lista_bits, dtype=np.float32)


def bits_a_texto(vector_bits):
    """
    Toma un vector de 256 bits (ceros y unos) y lo transforma
    de vuelta a una cadena de texto legible.
    """
    bits = vector_bits.astype(int)
    texto_reconstruido = ""
    
    for i in range(0, len(bits), 8):
        bloque_letra = bits[i:i+8]
        
        if np.all(bloque_letra == 0):
            break
            
        cadena_binaria = "".join(str(b) for b in bloque_letra)
        numero_ascii = int(cadena_binaria, 2)
        texto_reconstruido += chr(numero_ascii)
        
    return texto_reconstruido


# =============================================================================
# PRUEBA DEL PUENTE DE DATOS (Siempre al final del archivo)
# =============================================================================
if __name__ == "__main__":
    mensaje_original = "Hola Profe, esto es IA 10/10"
    
    # 1. Probamos la ida: Texto -> Bits
    vector_bits = texto_a_bits(mensaje_original)
    print(f"Texto original: '{mensaje_original}'")
    print(f"Convertido a vector de bits con forma: {vector_bits.shape}\n")
    
    # 2. Probamos la vuelta: Bits -> Texto
    mensaje_recuperado = bits_a_texto(vector_bits)
    print(f"Texto recuperado por el sistema: '{mensaje_recuperado}'")