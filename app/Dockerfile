# syntax=docker/dockerfile:1.4
FROM python:3.10 AS builder

# Definir el directorio de trabajo
WORKDIR /app

# Copiar el archivo de requerimientos
COPY requirements.txt .

# Instalar dependencias desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Descargar archivos adicionales de NLTK
RUN python -m nltk.downloader vader_lexicon punkt

# Copiar el resto de los archivos de la aplicación
COPY . .

# Configurar el punto de entrada
ENTRYPOINT ["python"]
CMD ["app.py"]
