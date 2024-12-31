import json
from typing import Any, List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

class QdrantEngineManager:
    """
    A manager class for Qdrant functionality, providing methods to initialize the connection
    and perform operations like creating collections, inserting/upserting data,
    and searching for vectors. 

    Attributes:
        host (str): Host name of the Qdrant server.
        port (int): Port number of the Qdrant server.
        client (QdrantClient): The Qdrant client instance for all operations.
    """
    def __init__(self, host: str, port: int):
        """
        Initialize the Qdrant client with host and port configuration.

        Args:
            host (str): Hostname of the Qdrant server.
            port (int): Port number of the Qdrant server.
        """
        self.host = host
        self.port = port
        self.client = QdrantClient(host=self.host, port=self.port)

    def create_collection_if_not_exists(self, collection_name: str, vector_size: int, distance: str = "COSINE"):
        """
        Create a Qdrant collection if it does not already exist.

        Args:
            collection_name (str): The name of the collection to create.
            vector_size (int): The size of the vectors stored in the collection.
            distance (str): The distance function to use for similarity search. Default is "COSINE".
        """
        try:
            self.client.get_collection(collection_name=collection_name)
            print(f"Collection '{collection_name}' already exists. Skipping creation.")
        except Exception:
            print(f"Collection '{collection_name}' doesnt exists. Creating new collection.")
            # If collection doesn't exist, create it
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "image_embeddings": VectorParams(size=vector_size, distance=getattr(Distance, distance))
                },
            )
            print(f"Collection '{collection_name}' created successfully.")

    def upsert_data(self, collection_name: str, points: List[PointStruct]):
        """
        Upsert (insert/update) data points in the specified collection.

        Args:
            collection_name (str): The name of the collection where data is upserted.
            points (List[PointStruct]): A list of `PointStruct` objects representing the data.
        """
        self.client.upsert(collection_name=collection_name, points=points)
        print(f"Upserted {len(points)} points into collection '{collection_name}'.")

    def search(self, collection_name: str, query_vector: Any, limit: int = 10, with_payload: bool = True) -> List[Any]:
        """
        Search for the most similar vectors in a given collection.

        Args:
            collection_name (str): The name of the collection to search in.
            query_vector (Any): The query vector to compare.
            limit (int): The maximum number of results to return.
            with_payload (bool): Whether to include payload information in the search results.

        Returns:
            List[Any]: A list of query result items.
        """
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_vectors=False,
            with_payload=with_payload,
        )

    def retrieve(self, collection_name: str, ids: List[str]) -> Optional[List[Any]]:
        """
        Retrieve specific data points by their IDs from a collection.

        Args:
            collection_name (str): The name of the collection to query.
            ids (List[str]): A list of point IDs to retrieve.

        Returns:
            Optional[List[Any]]: The retrieved points, or None if nothing matches.
        """
        return self.client.retrieve(collection_name=collection_name, ids=ids)

    def delete_point(self, collection_name: str, point_id: str):
        """
        Delete a specific point from a collection by ID.

        Args:
            collection_name (str): The name of the collection.
            point_id (str): The ID of the point to delete.
        """
        self.client.delete(collection_name=collection_name, points_selector={"id": point_id})
        print(f"Deleted point with ID '{point_id}' from collection '{collection_name}'.")