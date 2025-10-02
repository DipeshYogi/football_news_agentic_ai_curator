import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient, DataType
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
    

class MilvusHandler(TextEmbedder):
    def __init__(self, host="standalone", port="19530", collection_name="football_articles_milvus"):
        super().__init__()
        self.client = MilvusClient(uri=f"http://{host}:{port}")
        self.collection_name = collection_name

        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        if not self.client.has_collection(self.collection_name):
            schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)      
            schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=768)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=64000)
            schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
            schema.add_field(field_name="date", datatype=DataType.VARCHAR, max_length=32)

            self.client.create_collection(collection_name=self.collection_name, schema=schema)
            
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding", metric_type="IP", index_type="AUTOINDEX", params={}
            )
            self.client.create_index(collection_name=self.collection_name, index_params=index_params)

        self.client.load_collection(collection_name=self.collection_name)

    def insert_chunks(self, embeddings, chunks, metadata_list):
        records = [
            {
                "embedding": emb.tolist(),
                "text": chunk,
                "title": meta["title"],
                "date": str(meta["date"])
            }
            for emb, chunk, meta in zip(embeddings, chunks, metadata_list)
        ]

        self.client.insert(
            collection_name=self.collection_name,
            data=records
        )

    def search(self, query, top_k=5):
        return self.client.search(
            collection_name=self.collection_name,
            data=self.embed_texts([query]),
            search_params={"metric_type": "IP", "params": {}},
            limit=top_k,
            output_fields=["title", "text", "date"]
        )
