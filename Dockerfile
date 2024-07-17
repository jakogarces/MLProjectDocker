#FROM python:3.9-slim-buster #Funciona perfectamente si quieres DEBIAN
#Voy a usar el de acontinuación porque quiero UBUNTU
FROM ubuntu:20.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y wget git sox libsox-fmt-all

# Descargar e instalar Miniconda
#Para WINDOWS o Linux
#RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
#Para MAC
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh \
    # Crea un directorio .conda en el directorio home del usuario root.
    && mkdir /root/.conda \
    # Ejecuta el script de instalación de Miniconda en modo silencioso (sin interacción del usuario, con -b).
    && bash /tmp/miniconda.sh -b -p /root/miniconda3 \
    # Elimina el script de instalación después de que Miniconda ha sido instalado para mantener limpio el entorno Docker.
    && rm -f /tmp/miniconda.sh

#Test 1 - hasta aqui

#cd /root
WORKDIR /root

# Copiar el archivo environment.yml al directorio de trabajo
COPY environment.yml .

#Test 2 - hasta aqui

# Actualizar conda e instalar pip
RUN conda update -n base -c defaults conda -y
RUN conda install pip -y

# Crear el entorno conda 'datapath_mlops_cvenv' a partir del archivo environment.yml
RUN conda env create -f environment.yml

#Test 3 - hasta aqui

#Instalamos esto para poder lanzar el comando "lsb_release -a" y verificar la version de ubuntu
RUN apt-get install -y lsb-release

# paso 1: Comando para crear la imagen del Docker (solo una vez)
#docker build -t datapath_ubuntu20_python39 .

#paso 2: Crear contenedor de docker (cada vez que trabaje)
# docker run --rm -it --userns=host --shm-size 64G -v /Users/jacobogarces/Documents/MLOpsBootcamp/drugs/mlprojectdocker:/workspace/datapath_mlops/ -p '8484:8484' --name DATAPATH_CONTAINER datapath_ubuntu20_python39:latest /bin/bash

# Paso 3: Para entrar al Entorno Virtual de Anaconda Configurado en Environment.yml (Se realiza cada vez que trabaje):
#source activate datapath_mlops_cvenv

#Paso 4: Para entrar al Workspace configurado (Se realiza cada vez que trabaje):
#cd /workspace/datapath_mlops/

#Comandos extra que veremos en la sesión siguiente:
#pip install notebook
#jupyter notebook --allow-root --no-browser --ip 0.0.0.0 --port 8383