import pandas as pd
import numpy as np
import ast


def main():
    movies = pd.read_csv("Dataset/movies_dataset.csv")
    transform_movies(movies)


def transform_movies(df: pd.DataFrame):
    print(df.shape)
    columnas = ["revenue", "budget"]
    df[columnas] = df[columnas].fillna(0)

    df.drop(["video", "imdb_id", "adult",
             "original_title", "poster_path", "homepage"], axis=1, inplace=True)

    df["release_date"] = pd.to_datetime(
        df["release_date"], format='%Y-%m-%d', errors='coerce')
    df.dropna(subset=["release_date"], inplace=True)

    df["release_year"] = df["release_date"].dt.year

    df["budget"] = df["budget"].astype(int)

    df["return"] = np.where(df["budget"] != 0, df["revenue"] / df["budget"], 0)

    json = df["belongs_to_collection"]
    json_collec = {
        "id_collection": [],
        "name_collection": [],
        "poster_path_collection": [],
        "backdrop_path_collection": []
    }

    for item in json:
        if pd.notna(item) and type(ast.literal_eval(item)) != float:
            item = dict(ast.literal_eval(item))
            json_collec["id_collection"].append(item.get("id", None))
            json_collec["name_collection"].append(item.get("name", None))
            json_collec["poster_path_collection"].append(
                item.get("poster_path", None))
            json_collec["backdrop_path_collection"].append(
                item.get("backdrop_path", None))
        else:
            json_collec["id_collection"].append(None)
            json_collec["name_collection"].append(None)
            json_collec["poster_path_collection"].append(None)
            json_collec["backdrop_path_collection"].append(None)

    collection = pd.DataFrame(json_collec)
    collection.index = df["belongs_to_collection"].index
    
    df.drop(["belongs_to_collection"], axis=1, inplace=True)
    df = df.join(collection)
    
    
    check_info(df)
    
    df.to_csv("Dataset/movies_clean.csv", index=False)


def check_info(df):
    print(df.info())
    print(df.isna().sum())
    print("Cantidad total de valores faltantes: ", df.isna().sum().sum())
    print("-"*50)
    print(df.duplicated())
    print("Cantidad de duplicados: ", df.duplicated().sum())
    print("-"*50)
    print(df.head(10))
    print("Dimensiones: ", df.shape)


if __name__ == "__main__":
    main()
