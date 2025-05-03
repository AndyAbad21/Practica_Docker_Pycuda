# Usa la imagen base de NVIDIA CUDA
FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

# Instalar Conda
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH

# Crear un directorio de trabajo para el contenedor
WORKDIR /app

# Copiar los archivos del proyecto Flask al contenedor
COPY . /app

# Copiar el entorno de Conda para Flask
COPY pycuda_env.yml /app

# Crear el entorno de Conda dentro del contenedor
RUN conda env create -f pycuda_env.yml

# Activar el entorno de Conda
SHELL ["conda", "run", "-n", "pycuda_env", "/bin/bash", "-c"]

# Instalar dependencias adicionales si es necesario
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto 5000
EXPOSE 5000

# Comando para ejecutar la aplicación Flask
CMD ["conda", "run", "-n", "pycuda_env", "flask", "run", "--host=0.0.0.0"]
