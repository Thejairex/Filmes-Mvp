import pandas as pd
import json
from sklearn.preprocessing import MultiLabelBinarizer

# Ejemplo de datos
data = {
    'genres': [
        "[{'id': 16, 'name': 'Animation'}, {'id': 10751, 'name': 'Family'}]",
        "[{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]",
        "[{'id': 35, 'name': 'Comedy'}]",
        None
    ],
    'belongs_to_collection': [
        "{'id': 10194, 'name': 'Toy Story Collection', 'poster_path': '/7G9915LfUQ2lVfwMEEhDsn3kT4B.jpg', 'backdrop_path': '/9FBwqcd9IRruEDUrTdcaafOMKUq.jpg'}",
        None,
        "{'id': 10196, 'name': 'Matrix Collection', 'poster_path': '/1O30sF2yhhK9a2q0GB2tOyozlBb.jpg', 'backdrop_path': '/s4A0lWV5H5T3uP7PjQdd0xyJQKf.jpg'}",
        None
    ],
    'production_companies': [
        "[{'name': 'Pixar Animation Studios', 'id': 3}]",
        "[{'name': 'Warner Bros. Pictures', 'id': 174}]",
        "[{'name': 'Universal Pictures', 'id': 33}, {'name': 'Amblin Entertainment', 'id': 56}]",
        None
    ]
}

df = pd.DataFrame(data)

# Función genérica para normalizar columnas JSON
def normalize_json_column(column_str):
    if pd.isnull(column_str):
        return []
    items = []
    try:
        items_list = json.loads(column_str.replace("'", "\""))
        for item in items_list:
            items.append(item['name'])
    except (json.JSONDecodeError, KeyError):
        pass
    return items

# Aplicar la función a las columnas 'genres' y 'production_companies'
df['normalized_genres'] = df['genres'].apply(normalize_json_column)
df['normalized_companies'] = df['production_companies'].apply(normalize_json_column)

# Utilizar MultiLabelBinarizer para crear las columnas one-hot de manera eficiente
mlb = MultiLabelBinarizer()

# Transformar y crear DataFrames one-hot para 'genres'
genres_one_hot = pd.DataFrame(mlb.fit_transform(df['normalized_genres']), columns=mlb.classes_, index=df.index)

# Transformar y crear DataFrames one-hot para 'production_companies'
companies_one_hot = pd.DataFrame(mlb.fit_transform(df['normalized_companies']), columns=mlb.classes_, index=df.index)

# Concatenar las columnas one-hot con el DataFrame original
df = pd.concat([df, genres_one_hot, companies_one_hot], axis=1)

print(df)
