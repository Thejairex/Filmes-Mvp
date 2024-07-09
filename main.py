from fastapi import FastAPI
import pandas as pd


from recommender import MovieRecommender

app = FastAPI()

movies = pd.read_csv("Dataset/Cleaned/movies_clean.csv",
                     parse_dates=["release_date"])
actors = pd.read_csv("Dataset/Cleaned/actors_clean.csv")
crew = pd.read_csv("Dataset/Cleaned/crew_clean.csv")
recommendations = MovieRecommender(movies)


months_dict = {
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
days_dict = {
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
    """
    Mensaje de bienvenida.
    """
    return "Bienvenido a la API de Peliculas de Jesus Yair Juarez"


@app.get("/films/month/{month}")
def count_films_month(month: str):
    """
    Devuelve la cantidad de peliculas de un mes.
    """
    month_num = months_dict[month.lower().strip()]

    # Hacemos una mascara para obtener los datos de ese mes
    total = len(movies[movies["release_date"].dt.month == month_num])

    # borramos la variable temporal
    del month_num
    return f"Cantidad de peliculas en {month}: {total}"


@app.get("/films/day/{day}")
def count_films_day(day):
    """
    Devuelve la cantidad de peliculas de un dia de la semana.
    """
    day_num = days_dict[day.lower().strip()]
    # Obtenemos los datos del dia a través de la mascara
    weekday = movies[movies["release_date"].dt.weekday == day_num]

    # Borramos la variable temporal
    del day_num
    return f'La cantidad de peliculas en {day} fue de {len(weekday)}.'


@app.get("/titles/{title}/score")
def score_title(title):
    """
    Devuelve la puntuacion de una pelicula.
    """
    film = movies[movies["title"] == title]
    score = film["vote_average"].values[0]
    year = film["release_year"].values[0]

    # borramos la variable temporal
    del film
    return f"La película {title} fue estrenada en el año {year} con un score/popularidad de {score}"


@app.get("/titles/{title}/votes")
def votes_title(title):
    """
    Devuelve la informacion de los votos de una pelicula.
    """
    film = movies[movies["title"] == title]
    votes = film["vote_count"].values[0]

    # verificamos si hay mas de 2000 votos
    if votes > 2000:
        score = film["vote_average"].values[0]
        anio = film["release_year"].values[0]
        # borramos la variable temporal
        del film
        return f"La película {title} fue estrenada en el año {anio}. La misma cuenta con un total de {votes} valoraciones, con un promedio de {score}"

    else:
        del film, votes
        return f"La película {title} no cuenta con suficientes votos"


@app.get("/actors/{actor}")
def get_actor(actor):
    """
    Devuelve la informacion de un actor.
    """
    actor = actor.title().strip()
    # Obtenemos los datos de los actores, peliculas y retornos
    actor_films = actors[actors["name"] == actor]
    films = movies[movies["id"].isin(actor_films["id_film"])]
    returned = films["return"].sum().round(2)
    returned_mean = films["return"].mean().round(2)

    # obtenemos el total de filmaciones de ese actor
    total_film = films.shape[0]

    # borramos la variable temporal
    del actor_films, films
    return f"El actor {actor} ha ganado un retorno de {returned}% con un retorno promedio de {returned_mean}% en {total_film} filmaciones"


@app.get("/directores/{director}")
def get_director(director):
    """
    Devuelve la informacion de un director.
    """
    # Obtenemos los datos de los directores, peliculas y retornos
    director = director.title().strip()
    director_films = crew[(crew["name"] == director) &
                          (crew["job"] == "Director")]
    films = movies[movies["id"].isin(director_films["id_film"])]
    returned = films["return"].sum().round(2)

    # Creamos el diccionario con la informacion
    data = {
        "director": director,
        "exito": returned,
        "peliculas": [],

    }

    # Obtenemos la informacion de las peliculas de ese director
    for index, row in films.iterrows():
        film = {
            "title": row["title"],
            "fecha de lanzamiento": row["release_date"],
            "puntuacion": row["vote_average"],
            "costo": row["budget"],
            "recaudacion": row["revenue"],
            "ganacia %": round(row["return"], 2),
        }
        data["peliculas"].append(film)

    # borramos la variable temporal
    del director_films, films, film
    return data


@app.get("/recomendaciones/{title}")
def recomendar_title(title):
    """
    Devuelve 5 peliculas recomendadas a partir de una pelicula.
    """
    title = title.title().strip()
    rec = recommendations.recommend(title)
    return rec.to_dict()
