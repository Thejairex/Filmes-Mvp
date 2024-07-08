from fastapi import FastAPI
import pandas as pd
import datetime

app = FastAPI()

movies = pd.read_csv("Dataset/Cleaned/movies_clean.csv", parse_dates=["release_date"])
acthors = pd.read_csv("Dataset/Cleaned/credits_clean.csv")
crew = pd.read_csv("Dataset/Cleaned/crew_clean.csv")
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
dias_dict = {
    'lunes': 0,
    'martes': 1,
    'miercoles': 2,
    'jueves': 3,
    'viernes': 4,
    'sabado': 5,
    'domingo': 6
}

@app.get("/")
def index():

    return {}


@app.get("/filmaciones/mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    mes_num = meses_dict[mes.lower().strip()]
    total = len(movies[movies["release_date"].dt.month == mes_num])

    return f"Cantidad de filmaciones en {mes}: {total}"


@app.get("/filmaciones/dia/{dia}")
def cantidad_filmaciones_dia(dia):
    dia_num = dias_dict[dia.lower().strip()]
    dia_semana = movies[movies["release_date"].dt.weekday == dia_num]
    return f'La cantidad de filmaciones en {dia} fue de {len(dia_semana)}.'


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
def nombre_actor(actor):
    actor = actor.title().strip()
    actor_films = acthors[acthors["name"] == actor]
    films = movies[movies["id"].isin(actor_films["id_film"])]
    retorno = films["return"].sum().round(2)
    retorno_medio = films["return"].mean().round(2)
    total_film = films.shape[0]
    return "El actor {} ha ganado un retorno de {}% con un retorno promedio de {}% en {} filmaciones".format(actor, retorno, retorno_medio, total_film)


@app.get("/directores/{director}")
def get_director(director):
    director = director.title().strip()
    director_films = crew[(crew["name"] == director) & (crew["job"] == "Director")]
    films = movies[movies["id"].isin(director_films["id_film"])]
    retorno = films["return"].sum().round(2)
    data = {
        "director": director,
        "exito": retorno,
        "filmaciones": [],
        
    }
    for index, row in films.iterrows():
        film = {
            "titulo": row["title"],
            "fecha de lanzamiento": row["release_date"],
            "puntuacion": row["vote_average"],
            "costo": row["budget"],
            "recaudacion" : row["revenue"],
            "ganacia %" : round(row["return"], 2) ,
        }
        data["filmaciones"].append(film)
    return data
