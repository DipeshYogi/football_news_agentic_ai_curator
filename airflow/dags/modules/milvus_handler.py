import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility


class TextChunker:
    def __init__(self, chunk_size=400, chunk_overlap=80):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )
    
    def chunk_text(self, text):
        return self.splitter.split_text(text)


class TextEmbedder:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model = HuggingFaceEmbeddings(model_name=model_name)

    def embed_texts(self, texts):
        embeddings = self.model.embed_documents(texts)
        # Normalize embeddings for cosine similarity
        return np.array([e/np.linalg.norm(e) for e in embeddings])
    

class MilvusHandler:
    def __init__(self, host="localhost", port="19530", collection_name="football_articles"):
        connections.connect("default", host=host, port=port)
        self.collection_name = collection_name
        self.collection = self.create_or_get_collection()

    def create_or_get_collection(self):
        if utility.has_collection(self.collection_name):
            collection = Collection(self.collection_name)
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=32)
            ]
            schema = CollectionSchema(fields, description="Football news article chunks")
            collection = Collection(self.collection_name, schema)
            # create HNSW index
            index_params = {"index_type": "HNSW", "metric_type": "IP", "params": {"M":16, "efConstruction":200}}
            collection.create_index("embedding", index_params)
            collection.load()

        return collection
    
    def insert_chunks(self, embeddings, chunks, metadata_list):
        self.collection.insert([
            embeddings.tolist(),
            chunks,
            [m["title"] for m in metadata_list],
            [m["date"] for m in metadata_list],
        ])
        self.collection.flush()