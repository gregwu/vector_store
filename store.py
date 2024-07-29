import numpy as np
from typing import List, Callable
from sentence_transformers import SentenceTransformer

class SimMetric:
    @staticmethod
    def cosine_similarity(a, b):
        """
        Compute the cosine similarity between two vectors.

        Args:
            a (np.ndarray): First vector.
            b (np.ndarray): Second vector.

        Returns:
            float: Cosine similarity between the two vectors.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class Vectorstore:
    """
    A lightweight vectorstore built with numpy.
    """
    def __init__(
        self,
        docs: List[str],
        embedder: SentenceTransformer = None,
        similarity_metric: Callable = SimMetric.cosine_similarity
    ) -> None:
        """
        Initialize the Vectorstore.

        Args:
            docs (List[str]): List of documents.
            embedder (SentenceTransformer, optional): Sentence transformer model for embedding the documents. Defaults to None.
            similarity_metric (Callable, optional): Similarity metric function. Defaults to SimMetric.cosine_similarity.
        """
        self.docs = np.array(docs)
        self.embedder = embedder if embedder else SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_metric = similarity_metric
        self._store = None

        # Embed documents if embedder is provided
        if self.embedder:
            self._store = np.array([self.embedder.encode(doc) for doc in self.docs])
        else:
            raise ValueError("Embedder is not provided")

    def query(self, text: str, top_k: int = 5) -> List[int]:
        """
        Query the vector store with a text and return indices of top_k similar documents.

        Args:
            text (str): Query text.
            top_k (int, optional): Number of top similar documents to return. Defaults to 5.

        Returns:
            List[int]: List of indices of top_k similar documents.
        """
        query_vec = self.embedder.encode(text)
        similarities = np.array([self.similarity_metric(query_vec, doc_vec) for doc_vec in self._store])
        top_k_indices = similarities.argsort()[-top_k:][::-1]
        return top_k_indices.tolist()

    def add_documents(self, new_docs: List[str]) -> None:
        """
        Add new documents to the vector store and update embeddings.

        Args:
            new_docs (List[str]): List of new documents to add.
        """
        new_doc_vecs = np.array([self.embedder.encode(doc) for doc in new_docs])
        self.docs = np.append(self.docs, new_docs)
        self._store = np.append(self._store, new_doc_vecs, axis=0)

    def remove_document(self, index: int) -> None:
        """
        Remove a document from the vector store by its index.

        Args:
            index (int): Index of the document to remove.

        Raises:
            IndexError: If the index is out of range.
        """
        if index < 0 or index >= len(self.docs):
            raise IndexError("Index out of range")
        self.docs = np.delete(self.docs, index)
        self._store = np.delete(self._store, index, axis=0)

    def get_document_by_index(self, index: int) -> str:
        """
        Get a document by its index from the vector store.

        Args:
            index (int): Index of the document.

        Returns:
            str: Document at the given index.

        Raises:
            IndexError: If the index is out of range.
        """
        if index < 0 or index >= len(self.docs):
            raise IndexError("Index out of range")
        return self.docs[index]

    def print_query_results(self, text: str, top_k: int = 5) -> None:
        """
        Print the documents of the query results.

        Args:
            text (str): Query text.
            top_k (int, optional): Number of top similar documents to print. Defaults to 5.
        """
        indices = self.query(text, top_k)
        print("Query Results:")
        for index in indices:
            print(self.get_document_by_index(index))

# Example usage:
docs = ["Hello world", "Hi there", "Greetings", "Good day", "How are you"]
vectorstore = Vectorstore(docs)
vectorstore.print_query_results("Hello", top_k=2)

vectorstore.add_documents(["Nice to meet you", "Good evening"])
vectorstore.print_query_results("Good evening", top_k=2)

vectorstore.remove_document(0)
print(vectorstore.docs)

