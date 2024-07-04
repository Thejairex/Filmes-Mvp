import pandas as pd
import numpy as np
import ast
import json
from sklearn.preprocessing import MultiLabelBinarizer

def main():
    movies = pd.read_csv("Dataset/movies_dataset.csv")
    credits = pd.read_csv("Dataset/credits.csv")

    cast = credits[["id", "cast"]]
    crew = credits[["id", "crew"]]
    transform_movies(movies)
    transform_casts(cast)
    transform_crew(crew)


def normalize_list_column(df, column_name, key: str = "name"):
    """
    Normalize a column to a list in the dataset and add them to the dataset as new columns.

    Parameters:
    -----------
    df: movies dataset
    column_name: name of the column to normalize
    key: key to use in the list
    """
    def normalize(data):
        """
        Extract items from stringified JSON and convert it to a list.
        If there are no items, return an empty list.
        """
        if pd.isnull(data):
            return None
        items = []
        try:
            genres_list = json.loads(data.replace("'", "\""))
            for genre in genres_list:
                items.append(genre[key])
        except (json.JSONDecodeError, KeyError):
            pass
        return items

    df[f'normalized_{column_name}'] = df[f'{column_name}'].apply(normalize)
    df.drop(f'{column_name}', axis=1, inplace=True)
    return df


def normalize_collection(df):
    """
    Normalize belongs_to_collection column to a dictionary and add it to the dataset as new columns.
    """
    def normalize_columns(str_column):
        """
        Get the dictionary from the 'str_column' and return it.
        If the 'str_column' is null, return an empty dictionary.
        Parameters:
        ------------
        str_column: str (containing a stringified JSON)
        """
        if pd.isnull(str_column):
            return {}
        try:
            collection_dict = json.loads(str_column.replace("'", "\""))
            return collection_dict
        except (json.JSONDecodeError, KeyError):
            return {}
    # Apply the normalization function to the 'belongs_to_collection' column
    df['normalized_collection'] = df['belongs_to_collection'].apply(
        normalize_columns)

    # Create new columns for each column in the dictionary
    df['collection_id'] = df['normalized_collection'].apply(
        lambda x: x.get('id'))
    df['collection_name'] = df['normalized_collection'].apply(
        lambda x: x.get('name'))
    df['collection_poster_path'] = df['normalized_collection'].apply(
        lambda x: x.get('poster_path'))
    df['collection_backdrop_path'] = df['normalized_collection'].apply(
        lambda x: x.get('backdrop_path'))

    # Drop the original 'belongs_to_collection' and 'normalized_collection' columns
    df.drop(['belongs_to_collection', 'normalized_collection'],
            axis=1, inplace=True)

    return df

def normalize_to_onehot(column_str, key="name"):
    if pd.isnull(column_str):
        return []
    items = []
    try:
        items_list = json.loads(column_str.replace("'", "\""))
        for item in items_list:
            items.append(item[key])
    except (json.JSONDecodeError, KeyError):
        pass
    return items

def transform_movies(df: pd.DataFrame):
    """
    Process the dataset movies and transform it into a clean file that can be used in the model.

    Parameters:
    -----------
    df: movies dataset
    """
    # Drop rows with non-numeric IDs. The dataset contains movies with IDs that are not numeric is tiny.
    non_numeric = df["id"].apply(
        lambda x: pd.to_numeric(x, errors='coerce')).isna()
    df = df[~non_numeric]
    df["id"] = df["id"].astype(int)

    # Fill in missing values with 0 at the 'revenue' and 'budget' columns
    columnas = ["revenue", "budget"]
    df[columnas] = df[columnas].fillna(0)

    # drop columns useless
    df.drop(["video", "imdb_id", "adult",
             "original_title", "poster_path", "homepage"], axis=1, inplace=True)

    # Transform release_date column to datetime
    df["release_date"] = pd.to_datetime(
        df["release_date"], format='%Y-%m-%d', errors='coerce')
    df.dropna(subset=["release_date"], inplace=True)

    # Create release_year column from release_date
    df["release_year"] = df["release_date"].dt.year

    # assign type int to budget and create return column
    df["budget"] = df["budget"].astype(int)
    df["return"] = np.where(df["budget"] != 0, df["revenue"] / df["budget"], 0)

    # normalize columns nested
    df = normalize_collection(df)
    df["genres"] = df["genres"].apply(normalize_to_onehot)
    df = normalize_list_column(df, "production_companies")
    df = normalize_list_column(df, "production_countries", "iso_3166_1")
    # df["spoken_languages"] = df["spoken_languages"].apply(normalize_to_onehot, key="iso_639_1")
    df = normalize_list_column(df, "spoken_languages", "iso_639_1")

    mtl = MultiLabelBinarizer()
    genres_hot = pd.DataFrame(mtl.fit_transform(df["genres"]), columns=mtl.classes_, index=df.index)
    # spoken_hot = pd.DataFrame(mtl.fit_transform(df["spoken_languages"]), columns=mtl.classes_, index=df.index)
    df.drop("genres", inplace=True)
    df = pd.concat([df, genres_hot], axis=1)
    check_info(df)

    df.to_csv("Dataset/movies_clean.csv", index=False)


def clean_data(data):
    if not data: data = "null" 
    else:
        if '"' in data: data = data.replace('"', "")
        if "'" in data: data = data.replace("'", "")
        return data

def transform_casts(df: pd.DataFrame):
    def preprocess_json(json_str: str):
        
        datas = list(ast.literal_eval(json_str))
        temp = []
        for data in datas:
            datas_dict = {
                "cast_id": int(data['cast_id']),
                "caracter": str(clean_data(data["character"])),
                "credit_id": str(clean_data(data["credit_id"])),
                "gender": int(data["gender"]),
                "name": str(clean_data(data["name"])),
                "order": int(data["order"]),
                "profile_path": str(clean_data(data["profile_path"]))
            }
            temp.append(datas_dict)
        
        return json.loads(json.dumps(temp))
    
    df.rename(columns={"id": "id_film"}, inplace=True)
    df['cast'] = df['cast'].apply(preprocess_json)
    
    # Expandir la columna `cast`
    cast_expanded = df.explode('cast')

    # Convertir cada diccionario en una fila separada
    cast_normalized = pd.json_normalize(cast_expanded['cast'])

    # Agregar el `id` a cada fila del elenco
    cast_normalized['id_film'] = cast_expanded['id_film'].values

    # Ordenar las columnas para que `id` sea la primera columna
    columns = ['id_film'] + [col for col in cast_normalized.columns if col != 'id_film']
    cast_normalized = cast_normalized[columns]    
    # check_info_credits(cast_normalized)

    cast_normalized.to_csv("Dataset/credits_clean.csv", index=False)

def transform_crew(df: pd.DataFrame):
    def preprocess_json(json_str: str):
        datas = list(ast.literal_eval(json_str))
        temp = []
        for data in datas:
            datas_dict = {
                "id": int(data['id']),
                "department": str(clean_data(data["department"])),
                "credit_id": str(clean_data(data["credit_id"])),
                "gender": int(data["gender"]),
                "name": str(clean_data(data["name"])),
                "job": str(clean_data(data["job"])),
                "profile_path": str(clean_data(data["profile_path"]))
            }
            temp.append(datas_dict)
        
        return json.loads(json.dumps(temp))
    
    
    df.rename(columns={"id": "id_film"}, inplace=True)
    df['crew'] = df['crew'].apply(preprocess_json)
    
    cast_expanded = df.explode('crew')

    cast_normalized = pd.json_normalize(cast_expanded['crew'])

    cast_normalized['id_film'] = cast_expanded['id_film'].values

    columns = ['id_film'] + [col for col in cast_normalized.columns if col != 'id_film']
    cast_normalized = cast_normalized[columns]
    
    cast_normalized.to_csv("Dataset/crew_clean.csv", index=False)
    # check_info(cast_normalized)


def check_info(df):
    print(df.info())
    print(df.isna().sum())
    print("Cantidad total de valores faltantes: ", df.isna().sum().sum())
    print("-"*50)
    # print(df.duplicated())
    # print(df.duplicated().sum())
    print("Dimensiones: ", df.shape)    
    print(df[0:5])
def check_info_credits(df):
    print(df.info())
    print(df.isna().sum())
    print("Número total de valores faltantes: ", df.isna().sum().sum())
    print("-"*50)
    # print(df.duplicated())
    # print(df.duplicated().sum())
    print("Dimensiones: ", df.shape)
    # print(df[0:5])
    # test = df[0:6]
    # test.to_csv("test.csv", index=False)


def check_info_dev(df):
    print(df.info())
    print(df.head())


if __name__ == "__main__":
    main()

    # check_info_dev(pd.read_csv("Dataset/movies_clean.csv"))
