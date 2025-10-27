#!/usr/bin/env python3
"""
Script para preparar datos y fine-tunar un modelo YOLOv11 usando ultralytics.
Optimizado para ejecución en terminal.

Uso:
    python yolo_train.py --epochs 10 --img_size 640

Argumentos:
    --epochs: Número de epochs (default: 50)
    --img_size: Tamaño de imagen (default: 640)
"""

import os
import sys
import shutil
import random
import argparse
import logging
from pathlib import Path
import pandas as pd
from PIL import Image
import yaml
import kagglehub
from ultralytics import YOLO

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def download_dataset():
    """Descarga el dataset desde Kaggle si no existe."""
    try:
        descarga_dir = kagglehub.dataset_download("msalman97/dataset-for-traffic-sign-master-app")
        dataset_dir = os.path.join(descarga_dir, "Dataset")
        logger.info(f"Dataset descargado en: {dataset_dir}")
        return dataset_dir
    except Exception as e:
        logger.error(f"Error descargando dataset: {e}")
        sys.exit(1)

def prepare_and_finetune_yolo11_ultralytics(train_dir: str,
                                           labels_csv: str,
                                           model_path: str = "models/yolo11n.pt",
                                           out_dir: str = "dataset_prepared",
                                           val_split: float = 0.1,
                                           epochs: int = 50,
                                           batch_size: int = 16,
                                           img_size: int = 640,
                                           seed: int = 42):
    """
    Flujo para preparar datos y lanzar el fine-tuning de un modelo YOLOv11 usando ultralytics.
    - train_dir: directorio raíz que contiene las imágenes (ej. carpeta padre de "Train").
    - labels_csv: CSV con las anotaciones en formato: Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path
      (Width/Height son del tamaño de la imagen; Roi.X1/Y1/X2/Y2 son xmin/ymin/xmax/ymax absolutos; ClassId es numérico; Path es ruta relativa).
    Salida:
    - Crea una estructura preparada en ./dataset_prepared/{train,val}/images & labels
    - Genera data.yaml compatible con ultralytics
    - Llama a model.train() de ultralytics con los parámetros indicados
    """
    random.seed(seed)
    train_dir = Path(train_dir)
    labels_csv = Path(labels_csv)
    out_dir = Path(out_dir)
    
    logger.info("Iniciando preparación de datos...")
    
    # Leer CSV y mapear columnas
    df = pd.read_csv(labels_csv)
    df.columns = [c.strip() for c in df.columns]  # Normalizar nombres
    required_cols = ["Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId", "Path"]
    if not all(c in df.columns for c in required_cols):
        raise ValueError(f"CSV debe contener columnas: {required_cols}")
    
    # Renombrar para consistencia
    df = df.rename(columns={
        "Roi.X1": "xmin", "Roi.Y1": "ymin", "Roi.X2": "xmax", "Roi.Y2": "ymax",
        "ClassId": "class_id", "Path": "filename"
    })
    df["class_id"] = df["class_id"].astype(int)
    
    # Obtener clases únicas
    classes = sorted(df["class_id"].unique().astype(str).tolist())
    logger.info(f"Clases encontradas: {len(classes)}")
    
    # Crear estructura de salida
    for split in ("train", "val"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Encontrar imágenes disponibles
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    filenames = sorted(df["filename"].unique().tolist())
    available = {}
    for p in train_dir.rglob("*"):
        if p.suffix.lower() in img_exts:
            available[p.name] = p
    image_files = [f for f in filenames if Path(f).name in available]
    if not image_files:
        raise FileNotFoundError("No se encontraron imágenes referenciadas en el CSV dentro de train_dir")
    
    logger.info(f"Imágenes encontradas: {len(image_files)}")
    
    # Split train/val
    random.shuffle(image_files)
    val_count = max(1, int(len(image_files) * val_split))
    val_imgs = set(image_files[:val_count])
    train_imgs = set(image_files[val_count:])
    
    logger.info(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}")
    
    # Función para convertir bbox a YOLO normalizado
    def xyxy_to_yolo(xmin, ymin, xmax, ymax, iw, ih):
        x_center = ((xmin + xmax) / 2.0) / iw
        y_center = ((ymin + ymax) / 2.0) / ih
        w = (xmax - xmin) / iw
        h = (ymax - ymin) / ih
        return x_center, y_center, w, h
    
    # Procesar imágenes y generar labels
    logger.info("Procesando imágenes y generando labels...")
    for name in image_files:
        p_src = available[Path(name).name]
        with Image.open(p_src) as im:
            iw, ih = im.size
        split = "val" if name in val_imgs else "train"
        dst_img = out_dir / split / "images" / Path(name).name
        shutil.copy2(p_src, dst_img)
        
        rows = df[df["filename"] == name]
        label_lines = []
        for _, r in rows.iterrows():
            cls_id = int(r["class_id"])
            xmin = float(r["xmin"])
            ymin = float(r["ymin"])
            xmax = float(r["xmax"])
            ymax = float(r["ymax"])
            x_c, y_c, w, h = xyxy_to_yolo(xmin, ymin, xmax, ymax, iw, ih)
            label_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        
        dst_lbl = out_dir / split / "labels" / (Path(name).stem + ".txt")
        with open(dst_lbl, "w", encoding="utf8") as fh:
            fh.write("\n".join(label_lines))
    
    # Generar data.yaml
    data_yaml = {
        "train": str((out_dir / "train" / "images").resolve()),
        "val": str((out_dir / "val" / "images").resolve()),
        "nc": len(classes),
        "names": classes
    }
    data_yaml_path = out_dir / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf8") as fh:
        yaml.safe_dump(data_yaml, fh, sort_keys=False)
    
    logger.info("Datos preparados. Iniciando entrenamiento...")
    
    # Cargar modelo y entrenar con ultralytics
    model = YOLO(model_path)
    model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        seed=seed
    )
    
    logger.info("Fine-tuning completado. Resultados en el directorio runs/detect/train (por defecto de ultralytics).")
    return {"prepared_dataset": str(out_dir.resolve()), "data_yaml": str(data_yaml_path.resolve())}

def main():
    parser = argparse.ArgumentParser(description="Entrenar modelo YOLO para detección de señales de tráfico")
    
    # Argumentos opcionales
    parser.add_argument('--epochs', type=int, default=50, help='Número de epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=16, help='Tamaño de batch (default: 16)')
    parser.add_argument('--img_size', type=int, default=640, help='Tamaño de imagen (default: 640)')
    
    args = parser.parse_args()
    
    # Descargar dataset
    dataset_dir = download_dataset()
    
    # Configurar rutas fijas
    train_dir = os.path.join(dataset_dir, "Train")
    labels_csv = os.path.join(dataset_dir, "Train.csv")
    model_path = "models/yolo11n.pt"
    out_dir = "dataset_prepared"
    
    # Verificar que los archivos existen
    if not os.path.exists(train_dir):
        logger.error(f"Directorio de entrenamiento no existe: {train_dir}")
        sys.exit(1)
    if not os.path.exists(labels_csv):
        logger.error(f"Archivo CSV no existe: {labels_csv}")
        sys.exit(1)
    if not os.path.exists(model_path):
        logger.error(f"Modelo base no existe: {model_path}")
        sys.exit(1)
    
    logger.info(f"Configuración: epochs={args.epochs}, img_size={args.img_size}")
    
    # Ejecutar entrenamiento
    try:
        result = prepare_and_finetune_yolo11_ultralytics(
            train_dir=train_dir,
            labels_csv=labels_csv,
            model_path=model_path,
            out_dir=out_dir,
            epochs=args.epochs,
            img_size=args.img_size
        )
        logger.info(f"Entrenamiento completado exitosamente: {result}")
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()