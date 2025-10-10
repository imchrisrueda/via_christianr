import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
import cv2

def procesar_diccionario_imagenes(muestra_imagenes, modelo, conf_threshold=0.25):
    """
    Procesa las imágenes contenidas en un diccionario usando el modelo YOLO.
    
    Args:
        muestra_imagenes (dict): Diccionario donde las llaves son clases ('0' a '62') 
                              y los valores son rutas a imágenes
        modelo: Modelo YOLO cargado
        conf_threshold (float): Umbral de confianza para las detecciones (0-1)
        
    Returns:
        pd.DataFrame: DataFrame con los resultados de las predicciones
    """
    # Lista para almacenar los resultados
    resultados = []
    
    # Configurar dispositivo para usar GPU si está disponible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    
    # Procesar cada clase en el diccionario
    for clase, rutas_imagenes in tqdm(muestra_imagenes.items(), desc="Procesando clases"):
        # Si rutas_imagenes es una sola ruta (string), convertirla a lista
        if isinstance(rutas_imagenes, str):
            rutas_imagenes = [rutas_imagenes]
        
        # Procesar cada imagen de la clase
        for ruta_imagen in rutas_imagenes:
            # Verificar que la imagen existe
            if not os.path.exists(ruta_imagen):
                print(f"Advertencia: La imagen {ruta_imagen} no existe. Saltando...")
                continue
                
            try:
                # Realizar la detección con el modelo YOLO
                results = modelo(ruta_imagen, conf=conf_threshold, device=device)
                result = results[0]  # Tomamos el primer resultado (una sola imagen)
                
                # Si no hay detecciones, añadir una fila con valores nulos para las detecciones
                if len(result.boxes) == 0:
                    resultados.append({
                        'clase_real': clase,
                        'ruta_imagen': ruta_imagen,
                        'clase_predicha': None,
                        'confianza': None,
                        'bbox': None
                    })
                else:
                    # Para cada detección en la imagen
                    for box in result.boxes:
                        cls_id = int(box.cls.item())
                        cls_name = result.names[cls_id]
                        conf = box.conf.item()
                        bbox = box.xyxy.tolist()[0]  # Convertir a lista [x1, y1, x2, y2]
                        
                        # Añadir resultado a la lista
                        resultados.append({
                            'clase_real': clase,
                            'ruta_imagen': ruta_imagen,
                            'clase_predicha': cls_name,
                            'confianza': conf,
                            'bbox': bbox
                        })
            except Exception as e:
                print(f"Error al procesar la imagen {ruta_imagen}: {str(e)}")
                # Añadir fila con error
                resultados.append({
                    'clase_real': clase,
                    'ruta_imagen': ruta_imagen,
                    'clase_predicha': 'ERROR',
                    'confianza': None,
                    'bbox': None,
                    'error': str(e)
                })
    
    # Crear DataFrame con los resultados
    df_resultados = pd.DataFrame(resultados)
    
    return df_resultados

def visualizar_resultados_muestra(df_resultados, num_muestras=3):
    """
    Visualiza una muestra de imágenes con sus predicciones.
    
    Args:
        df_resultados (pd.DataFrame): DataFrame con los resultados de las predicciones
        num_muestras (int): Número de imágenes a visualizar
    """
    # Filtrar solo filas con detecciones válidas
    df_con_detecciones = df_resultados[df_resultados['clase_predicha'].notna() & 
                                     (df_resultados['clase_predicha'] != 'ERROR')]
    
    # Si no hay detecciones válidas, mostrar mensaje
    if len(df_con_detecciones) == 0:
        print("No hay detecciones válidas para visualizar.")
        return
    
    # Seleccionar imágenes únicas para visualizar
    rutas_unicas = df_con_detecciones['ruta_imagen'].unique()
    num_muestras = min(num_muestras, len(rutas_unicas))
    rutas_muestra = np.random.choice(rutas_unicas, num_muestras, replace=False)
    
    # Configurar el tamaño de la figura
    plt.figure(figsize=(15, 5 * num_muestras))
    
    # Visualizar cada imagen seleccionada
    for i, ruta_imagen in enumerate(rutas_muestra):
        # Obtener todas las detecciones para esta imagen
        detecciones = df_con_detecciones[df_con_detecciones['ruta_imagen'] == ruta_imagen]
        clase_real = detecciones['clase_real'].iloc[0]
        
        # Cargar la imagen
        imagen = cv2.imread(ruta_imagen)
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        
        # Dibujar las detecciones en la imagen
        for _, det in detecciones.iterrows():
            if det['bbox'] is not None:
                x1, y1, x2, y2 = map(int, det['bbox'])
                cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Añadir etiqueta con clase y confianza
                etiqueta = f"{det['clase_predicha']}: {det['confianza']:.2f}"
                cv2.putText(imagen, etiqueta, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Mostrar la imagen con las detecciones
        plt.subplot(num_muestras, 1, i + 1)
        plt.imshow(imagen)
        plt.title(f"Imagen: {os.path.basename(ruta_imagen)} | Clase real: {clase_real}")
        plt.axis('off')
        
        # Mostrar detalle de las detecciones
        info_detecciones = "\n".join([
            f"Detección {i+1}: {det['clase_predicha']}, Confianza: {det['confianza']:.2f}"
            for i, (_, det) in enumerate(detecciones.iterrows())
        ])
        plt.figtext(0.5, 0.01 + (i / num_muestras), info_detecciones, 
                   ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5})
    
    plt.tight_layout()
    plt.show()

# Ejemplo de uso en un notebook:
"""
# Cargar el modelo YOLO
model = YOLO('traffic_sign_detector.pt')

# Procesar las imágenes del diccionario
df_resultados = procesar_diccionario_imagenes(muestra_imagenes, model, conf_threshold=0.25)

# Mostrar el DataFrame con los resultados
print("Resultados de la detección:")
display(df_resultados)

# Estadísticas básicas
print("\nEstadísticas:")
print(f"Total de imágenes procesadas: {df_resultados['ruta_imagen'].nunique()}")
print(f"Total de detecciones: {len(df_resultados[df_resultados['clase_predicha'].notna()])}")

# Mostrar distribución de clases predichas
if 'clase_predicha' in df_resultados.columns and df_resultados['clase_predicha'].notna().any():
    print("\nDistribución de clases predichas:")
    display(df_resultados['clase_predicha'].value_counts().head(10))

# Visualizar algunos resultados
visualizar_resultados_muestra(df_resultados, num_muestras=3)
"""