import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    """
    Clase para recomendar pelúculas usando la similitud coseno y tf-idf
    """

    def __init__(self, data: pd.DataFrame = None):
        """
        Inicializa la clase MovieRecommender
        """
        # Limitamos el dataset a 2000 pelúculas para ahorrar memoria.
        self.data = data.iloc[:2000]
        self.train()

    def train(self):
        """
        Calcula la matriz de similitud coseno y la almacena en self.cosine_sim
        """
        # Crea la matriz tf-idf y calcula la similitud coseno
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(self.data["title"])
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def recommend(self, title) -> pd.Series:
        """

        Recomienda peliculas basado en la similitud coseno y tf-idf
        encontrando los 5 mejores puntuaciones y devolviendo los tiútulos de las pelúculas.
        Si la pelúcula no se encuentra, devuelve "Pelicula no encontrada"

        Parameters
        ----------
        title: str
            El tiútulo de la pelúcula
        """
        try:
            # Obtiene el índice de la pelúcula en el dataframe
            idx = self.data[self.data["title"] == title].index[0]

            # obtiene la similitud coseno de las 5 pelúcula
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:6]

            # obtiene los índices de las pelúculas en el dataframe y devuelve los tiútulos
            movie_indices = [i[0] for i in sim_scores]
            return self.data["title"].iloc[movie_indices]
        except:
            # Devuelve "Pelicula no encontrada" si la pelúcula no se encuentra
            return "Pelicula no encontrada"
