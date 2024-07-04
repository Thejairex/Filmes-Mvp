import pandas as pd
import json
import re
# Leer el archivo CSV
df = pd.read_csv('test.csv')

def escape_quotes_in_values(json_str):
    # Expresión regular para encontrar las comillas dobles dentro de los valores
    pattern = r'(?<=: \")([^\"]*?\"[^\"]*?)(?=\")'
    return re.sub(pattern, lambda x: x.group(0).replace('"', '\\"'), json_str)

# Function to preprocess and load JSON
def preprocess_json(json_str):
    # Escapar comillas dobles dentro de los valores
    
    # print(json_str)
    splited = json_str.split('"')
    if len(splited) > 1:
        for i in range(1, len(splited)):
            # print( splited[i])
            if "'" in splited[i]:
                print( splited[i])
    # Reemplazar 'None' con 'null'
    json_str = json_str.replace("None", 'null')
    
    # Reemplazar comillas simples con comillas dobles
    json_str = json_str.replace("'", '"')
    
    # Eliminar comillas dobles repetidas
    json_str = re.sub(r'""(?=\w)', '"', json_str)
    
    
    return json.loads(json_str)

def check_missing(obj):
    for key, value in obj.items():
        if isinstance(value, str) and '"' in value:
            # print("key acutal: ", key)
            obj[key] = value.replace('"', "")
    return obj

# Convertir la columna `cast` a una lista de diccionarios
df['cast'] = df['cast'].apply(preprocess_json)

# Expandir la columna `cast`
cast_expanded = df.explode('cast')

# Convertir cada diccionario en una fila separada
cast_normalized = pd.json_normalize(cast_expanded['cast'])

# Agregar el `id` a cada fila del elenco
cast_normalized['id'] = cast_expanded['id'].values

# Ordenar las columnas para que `id` sea la primera columna
columns = ['id'] + [col for col in cast_normalized.columns if col != 'id']
cast_normalized = cast_normalized[columns]

# Guardar el resultado en un nuevo archivo CSV
cast_normalized.to_csv('cast_normalizado.csv', index=False)
