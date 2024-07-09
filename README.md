# Proyecto de Ciencia de Datos para Servicio de Streaming

Este proyecto tiene como objetivo analizar y dar recomendaciones de peliculas. El proyecto esta vigente en internet a traves de una API.

## Tabla de Contenidos

- [Descripción](#descripción)
- [Instalación](#instalación)
- [Trabajo Realizado](#Trabajo-Realizado)
- [Uso De la API](#Uso-de-la-API)

## Descripción

En este proyecto, utilizamos diversas técnicas de ciencia de datos para analizar y predecir el comportamiento del contenido de un servicio de streaming. Esto incluye:

- **Limpieza de datos**: Transformamos los datos y utilizando algoritmos para desanidad los datos.
- **Analisis de los datos**: Los datos limpios lo analizamos para ver su calidad. 
- **Recomendación de películas**: Creamos un sistema de recomendacion basada en la vectorizacion de texto y en la similutud del coseno..
- **Desarrollo de API**: Todo los datos y el sistema de recomendacion estan deployadas en un servicio web Llamada render.

## Trabajo Realizado
En el comiendo de este proyecto se nos dio la tarea de realizar la limpieza de los datos y transformarlos en los datasets movies.csv y credits.csv, ambos estaban con muchos datos erroneos y con columnas anidadas. Con algoritmos y tecnicas pudimos limpiar los datos y convertirlos en un dataset limpio. Terminado el proceso de ETL, se nos dio la tarea de realizar el analisis de los datos. Con el analisis de los datos notamos que datos nos seria de utilizar y cuales habia que desechar. Por ultimo, se nos dio la tarea de crear un sistema de recomendacion, Se realizo basada en la vectorizacion de texto y en la similitud del coseno a partir del titulo de la pelicula.

Una ves terminado con todo los procesos se nos pidio crear una API que nos permitiera interactuar con algunos endpoits y con el sistema de recomendacion.

## Uso de la API
Para interactuar con la API se utilizó la librería FastAPI.

Esta API contiene 6 endpoints que nos permiten interactuar.

- **GET** `/films/month/{month}`: Retorna una respuesta de la cantidad de peliculas que se estrenaron en ese mes.
- **GET** `/films/day/{day}`: Retorna una respuesta de la cantidad de peliculas que se estrenaron en ese dia.
- **GET** `/titles/{title}/score`: Retorna el puntaje de una pelicula.
- **GET** `/titles/{title}/votes`: Retorna la cantidad de votos de una pelicula.
- **GET** `/actors/{actor}`: Retorna la informacion del actor y la cantidad de piliculas en las que aparece.
- **GET** `/directors/{director}`: Retorna la informacion del director y la informacion de cada pelicula en la que participo.
- **GET** `/films/recomendaciones/{title}`: Retorna las recomendaciones de una pelicula.

Link: https://proyecto-individual-1-ga7m.onrender.com



## Instalación

Sigue estos pasos para configurar el entorno y ejecutar el proyecto:

1. Clona el repositorio:
    ```bash
    git clone https://github.com/tu-usuario/nombre-del-proyecto.git
    ```
2. Crear un entorno virtual:
    ```bash
    python -m virtualenv venv
    ```
3. activa un entorno virtual:
    ```bash
    python -m venv env
    En linux: source env/bin/activate  
    En Windows: .\env\Scripts\activate
    ```
4. Instala las dependencias necesarias:
    ```bash
    pip install -r requirements.txt
    ```
