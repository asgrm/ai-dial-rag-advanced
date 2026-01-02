import os
from enum import StrEnum
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    #TODO:
    # provide method `process_text_file` that will:
    #   - apply file name, chunk size, overlap, dimensions and bool of the table should be truncated
    #   - truncate table with vectors if needed
    #   - load content from file and generate chunks (in `utils.text` present `chunk_text` that will help do that)
    #   - generate embeddings from chunks
    #   - save (insert) embeddings and chunks to DB
    #       hint 1: embeddings should be saved as string list
    #       hint 2: embeddings string list should be casted to vector ({embeddings}::vector)

    def process_text_file(
            self,
            file_name: str,
            chunk_size: int,
            truncate_table: bool = True,
            dimensions: int = 1536,
            overlap: int = 10
    ):
        # default checks
        if chunk_size < 10:
            raise ValueError("chunk_size must be at least 10")
        if overlap < 0:
            raise ValueError("overlap must be at least 0")
        if overlap >= chunk_size:
            raise ValueError("overlap should be lower than chunkSize")

        if truncate_table:
            self._truncate_table()

        base_dir = os.path.dirname(__file__)
        project_root = Path(__file__).resolve().parents[1]
        print(base_dir)
        print(project_root)
        manual_path = project_root / file_name
        # manual_path = os.path.join(base_dir, file_name)
        print(manual_path)

        with open(manual_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        chunks = chunk_text(text=file_content, chunk_size=chunk_size, overlap=overlap)

        embeddings = self.embeddings_client.get_embeddings(inputs=chunks, dimensions=dimensions)

        # self._save_chunks(
        #     document_name=file_name,
        #     chunks=chunks,
        #     embeddings=embeddings
        # )

        for i in range(len(chunks)):
            self._save_chunk(embedding=embeddings.get(i), chunk=chunks[i], document_name=file_name)


    #TODO:
    # provide method `search` that will:
    #   - apply search mode, user request, top k for search, min score threshold and dimensions
    #   - generate embeddings from user request
    #   - search in DB relevant context
    #     hint 1: to search it in DB you need to create just regular select query
    #     hint 2: Euclidean distance `<->`, Cosine distance `<=>`
    #     hint 3: You need to extract `text` from `vectors` table
    #     hint 4: You need to filter distance in WHERE clause
    #     hint 5: To get top k use `limit`
    def search(
            self,
            search_mode: SearchMode,
            user_request: str,
            top_k: int,
            score_threshold: float,
            dimensions: int = 1536
    ):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if score_threshold < 0 or score_threshold > 1:
            raise ValueError("score_threshold must be in [0.0..., 0.99...] range")

        embeddings = self.embeddings_client.get_embeddings(inputs=user_request, dimensions=dimensions)[0]

        vector_string = f"[{','.join(map(str, embeddings))}]"

        if search_mode == SearchMode.COSINE_DISTANCE:
            max_distance = 1.0 - score_threshold
        else:
            max_distance = float('inf') if score_threshold == 0 else (1.0 / score_threshold) - 1.0

        retrieved_chunks = []

        with self._get_connection() as connection:
            try:
                with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(self._get_search_query(search_mode=search_mode), (vector_string, vector_string, max_distance, top_k))
                    results = cursor.fetchall()

                    for row in results:
                        if search_mode == SearchMode.COSINE_DISTANCE:
                            similarity = 1.0 - row['distance']
                        else:
                            similarity = 1.0 / (1.0 + row['distance'])

                        print(f"---Similarity score: {similarity:.2f}---")
                        print(f"Data: {row['text']}\n")
                        retrieved_chunks.append(row['text'])

            except psycopg2.Error as e:
                connection.rollback()
                raise e

        return retrieved_chunks

    def _truncate_table(self):
        with self._get_connection() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("TRUNCATE TABLE vectors")
                connection.commit()
                print('"vectors" table has been truncated')

            except psycopg2.Error as e:
                connection.rollback()
                raise RuntimeError("Failed to truncate vectors table") from e

    def _save_chunk(self, embedding: list[float], chunk: str, document_name):
        vector_string = f"[{','.join(map(str, embedding))}]"

        with self._get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector)",
                    (document_name, chunk, vector_string)
                )
                connection.commit()

    def _get_search_query(self, search_mode: SearchMode) -> str:
        return """SELECT text, embedding {mode} %s::vector AS distance
                  FROM vectors
                  WHERE embedding {mode} %s::vector <= %s
                  ORDER BY distance
                  LIMIT %s""".format(mode='<->' if search_mode == SearchMode.EUCLIDIAN_DISTANCE else '<=>')