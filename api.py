from fastapi import FastAPI 
import pandas as pd
app = FastAPI()

movies = pd.read_csv("Dataset/movies_clean.csv", parse_dates=["release_date"])
meses_dict = {
    'enero': 1,
    'febrero': 2,
    'marzo': 3,
    'abril': 4,
    'mayo': 5,
    'junio': 6,
    'julio': 7,
    'agosto': 8,
    'septiembre': 9,
    'octubre': 10,
    'noviembre': 11,
    'diciembre': 12
}

@app.get("/")
def index():

    return {}


@app.get("/filmaciones/{mes}")
def cantidad_filmaciones_mes(mes: str):
    month = movies[movies["release_date"].dt.month == meses_dict[mes.lower()]].to_dict()
    
    return f"Cantidad de filmaciones en {mes}: {len(month)}"


@app.get("/filmaciones/{dia}")
def cantidad_filmaciones_dia(dia):
    return {"": ""}


@app.get("/titulos/{titulo}/score")
def score_titulo(titulo):
    film = movies[movies["title"] == titulo]
    score = film["vote_average"].values[0]
    anio = film["release_year"].values[0]
    del film
    return f"La película {titulo} fue estrenada en el año {anio} con un score/popularidad de {score}"


@app.get("/titulos/{titlo}/votos")
def votos_titulo(titulo):
    film = movies[movies["title"] == titulo]
    votes = film["vote_count"].values[0]
    if votes > 2000:
        score = film["vote_average"].values[0]
        anio = film["release_year"].values[0]
        del film
        return f"La película {titulo} fue estrenada en el año {anio}. La misma cuenta con un total de {votes} valoraciones, con un promedio de {score}"

    else:
        del film
        return f"La película {titulo} no cuenta con suficientes votos"


@app.get("/actores/{actor}")
def nombre_actor (actor):
    return {"": ""}


@app.get("/directores/{nombre_director}")
def get_director(nombre_director):
    return {"": ""}