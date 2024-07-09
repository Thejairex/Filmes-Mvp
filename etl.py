import requests as rq
import gdown
import pandas as pd
import numpy as np
from io import StringIO
import ast
import json
from sklearn.preprocessing import MultiLabelBinarizer


def main():
    # Cargamos los datos
    movies = pd.read_csv(get_csv("movies"))
    credits = pd.read_csv(get_csv("credits"))

    cast = credits[["id", "cast"]]
    crew = credits[["id", "crew"]]

    del credits

    transform_movies(movies)
    transform_casts(cast)
    transform_crew(crew)


def get_csv(name: str):
    """
    Conseguimos los csv de google drive.
    """

    files_id = {
        "movies": "1Rp7SNuoRnmdoQMa5LWXuK4i7W1ILblYb",
        "credits": "1lMGJUWVVVRPO00ZWqzEJZIxFhSqtfmAB",
    }
    url = f"https://drive.google.com/uc?export=download&id={files_id[name]}"

    if name == "credits":
        filename = gdown.download(
            url, output="Dataset/credits.csv", quiet=False)
        return filename

    elif name == "movies":
        response = rq.get(url)
        csv_data = StringIO(response.text)
        return csv_data
    else:
        raise ValueError


def normalize_list_column(df, column_name, key: str = "name"):
    """
    Normaliza la columna que se le pase del dataset a una lista.

    Parameters:
    -----------
    df: DataFrame
    column_name: Nombre de la columna
    key: Clave que queremos extraer de cada elemento de la lista
    """
    def normalize(data):
        """
        Obtiene los items de un stringificado JSON y los convierte en una lista.
        Si no hay items, devuelve una lista vacía.

        Parameters:
        ------------
        data: str (Conteniendo un stringificado JSON)
        """

        # verifica si es nulo
        if pd.isnull(data):
            return None

        # Creamos una lista vacía
        items = []

        # Intentamos parsear el string a JSON
        try:
            genres_list = json.loads(data.replace("'", "\""))
            for genre in genres_list:
                items.append(genre[key])
        except (json.JSONDecodeError, KeyError):
            # En caso de error, no se agrega nada
            pass
        return items

    # Aplicamos la normalizacion y eliminamos la columna original
    df[f'normalized_{column_name}'] = df[f'{column_name}'].apply(normalize)
    df.drop(f'{column_name}', axis=1, inplace=True)
    return df


def normalize_collection(df):
    """
    Normaliza la columna belongs_to_collection del dataset a un diccionario y las agrega como nuevas columnas al dataset.
    """
    def normalize_columns(str_column):
        """
        Obtiene el diccionario de un stringificado JSON y lo devuelve.
        Si la 'str_column' es nula, devuelve un diccionario vacío.

        Parameters:
        ------------
        str_column: str (Conteniendo un stringificado JSON)
        """

        # verifica si es nulo
        if pd.isnull(str_column):
            return {}

        # Intentamos parsear el string a JSON e devolver el diccionario
        try:
            collection_dict = json.loads(str_column.replace("'", "\""))
            return collection_dict
        except (json.JSONDecodeError, KeyError):
            # En caso de error, no devuelve nada
            return {}
    # Aplica la normalización a la columna 'belongs_to_collection'
    df['normalized_collection'] = df['belongs_to_collection'].apply(
        normalize_columns)

    # Creamos las columnas nuevas para cada columna del diccionario
    df['collection_id'] = df['normalized_collection'].apply(
        lambda x: x.get('id'))
    df['collection_name'] = df['normalized_collection'].apply(
        lambda x: x.get('name'))
    df['collection_poster_path'] = df['normalized_collection'].apply(
        lambda x: x.get('poster_path'))
    df['collection_backdrop_path'] = df['normalized_collection'].apply(
        lambda x: x.get('backdrop_path'))

    # Borramos las columnas 'belongs_to_collection' y 'normalized_collection'
    df.drop(['belongs_to_collection', 'normalized_collection'],
            axis=1, inplace=True)

    return df


def normalize_to_onehot(column_str, key="name"):
    """
    Normaliza una columna que se le pase del dataset a una lista de items. Y las convierte en un vector de 0 y 1.
    """
    if pd.isnull(column_str):
        return []
    items = []
    try:
        # Parseamos el string a JSON
        items_list = json.loads(column_str.replace("'", "\""))
        for item in items_list:
            items.append(item[key])
    except (json.JSONDecodeError, KeyError):
        pass
    return items


def clean_str(data):
    """
    Limpia los datos de un string.
    """

    # Eliminamos las comillas dobles y simples
    if not data:
        data = "null"
    else:
        if '"' in data:
            data = data.replace('"', "")
        if "'" in data:
            data = data.replace("'", "")
        return data


def transform_movies(df: pd.DataFrame):
    """
    Procesamos el dataset movies y lo transformamos en un archivo limpio que puede ser usado en el modelo

    Parameters:
    -----------
    df: movies dataset
    """
    # Drop rows with non-numeric IDs. The dataset contains movies with IDs that are not numeric is tiny.
    # Borramos los datos que no son numericos
    non_numeric = df["id"].apply(
        lambda x: pd.to_numeric(x, errors='coerce')).isna()
    df = df[~non_numeric]
    df["id"] = df["id"].astype(int)

    # Llenamos los valores faltantes con 0 en las columnas 'revenue' y 'budget'
    columnas = ["revenue", "budget"]
    df[columnas] = df[columnas].fillna(0)

    # Borramos columnas que no nos sirven
    df.drop(["video", "imdb_id", "adult",
             "original_title", "poster_path", "homepage"], axis=1, inplace=True)

    # Transforma la columna release_date al formato datetime
    df["release_date"] = pd.to_datetime(
        df["release_date"], format='%Y-%m-%d', errors='coerce')
    df.dropna(subset=["release_date"], inplace=True)

    # Creamos la columna release_year a partir de release_date
    df["release_year"] = df["release_date"].dt.year

    # Asigna tipo int a la columna budget y crea la columna return
    df["budget"] = df["budget"].astype(int)
    df["return"] = np.where(df["budget"] != 0, df["revenue"] / df["budget"], 0)

    # normalize columns nested
    # Normalizamos las columnas agrupadas
    df = normalize_collection(df)
    df["genres"] = df["genres"].apply(normalize_to_onehot)
    df = normalize_list_column(df, "production_companies")
    df = normalize_list_column(df, "production_countries", "iso_3166_1")
    df = normalize_list_column(df, "spoken_languages", "iso_639_1")

    # One hot encoding de los generos
    mtl = MultiLabelBinarizer()
    genres_hot = pd.DataFrame(mtl.fit_transform(
        df["genres"]), columns=mtl.classes_, index=df.index)

    # Borramos la columna genres
    df.drop("genres", axis=1, inplace=True)
    # Concatenamos los dataframes
    df = pd.concat([df, genres_hot], axis=1)

    # Guardamos el dataset limpio
    df.to_csv("Dataset/movies_etl.csv", index=False)


def transform_casts(df: pd.DataFrame):
    """
    Procesamos el dataset casts y lo transformamos en un archivo limpio que puede ser usado en el modelo

    Parameters:
    -----------
    df: casts dataset
    """
    def preprocess_json(json_str: str):
        """
        Preprocesamos el json de las columnas `cast` extrayendo los campos y limpiando los datos.
        """
        # Pasamos el string a una lista
        datas = list(ast.literal_eval(json_str))

        # Creamos la lista temporal
        temp = []

        # Recorremos la lista limpiando los datos y guardando en un diccionario
        for data in datas:
            datas_dict = {
                "cast_id": int(data['cast_id']),
                "caracter": str(clean_str(data["character"])),
                "credit_id": str(clean_str(data["credit_id"])),
                "gender": int(data["gender"]),
                "name": str(clean_str(data["name"])),
                "order": int(data["order"]),
                "profile_path": str(clean_str(data["profile_path"]))
            }
            # Guardamos el diccionario en una lista temporal
            temp.append(datas_dict)

        return json.loads(json.dumps(temp))

    # Renombrar la columna `id` para que no tengamos problemas
    df.rename(columns={"id": "id_film"}, inplace=True)

    # Aplicar la función `preprocess_json` a la columna `cast`
    df['cast'] = df['cast'].apply(preprocess_json)

    # Expandir la columna `cast`
    cast_expanded = df.explode('cast')

    # normalizar la columna `cast` con el `pd.json_normalize`
    cast_normalized = pd.json_normalize(cast_expanded['cast'])

    # Agregar el `id` a cada fila del elenco
    cast_normalized['id_film'] = cast_expanded['id_film'].values

    # Ordenar las columnas para que `id` sea la primera columna
    columns = ['id_film'] + \
        [col for col in cast_normalized.columns if col != 'id_film']
    cast_normalized = cast_normalized[columns]

    cast_normalized.to_csv("Dataset/actors_etl.csv", index=False)


def transform_crew(df: pd.DataFrame):
    def preprocess_json(json_str: str):
        """
        Preprocesamos el json de las columnas `cast` extrayendo los campos y limpiando los datos.
        """
        # Pasamos el string a una lista
        datas = list(ast.literal_eval(json_str))

        # Creamos la lista temporal
        temp = []

        # Recorremos la lista limpiando los datos y guardando en un diccionario
        for data in datas:
            datas_dict = {
                "id": int(data['id']),
                "department": str(clean_str(data["department"])),
                "credit_id": str(clean_str(data["credit_id"])),
                "gender": int(data["gender"]),
                "name": str(clean_str(data["name"])),
                "job": str(clean_str(data["job"])),
                "profile_path": str(clean_str(data["profile_path"]))
            }
            # Guardamos el diccionario en una lista temporal
            temp.append(datas_dict)

        return json.loads(json.dumps(temp))

    # Renombrar la columna `id` para que no tengamos problemas
    df.rename(columns={"id": "id_film"}, inplace=True)

    # Aplicar la función `preprocess_json` a la columna `cast`
    df['crew'] = df['crew'].apply(preprocess_json)

    # Expandir la columna `cast`
    cast_expanded = df.explode('crew')

    # normalizar la columna `cast` con el `pd.json_normalize`
    cast_normalized = pd.json_normalize(cast_expanded['crew'])
    cast_normalized['id_film'] = cast_expanded['id_film'].values

    # Ordenar las columnas para que `id_film` sea la primera columna
    columns = ['id_film'] + \
        [col for col in cast_normalized.columns if col != 'id_film']
    cast_normalized = cast_normalized[columns]

    # Guardamos el dataset
    cast_normalized.to_csv("Dataset/crew_etl.csv", index=False)


if __name__ == "__main__":
    main()
