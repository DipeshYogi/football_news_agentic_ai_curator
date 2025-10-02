import time
import random
import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from modules.fetcher_utils import RssFetcher
from modules.milvus_handler import TextChunker, TextEmbedder, MilvusHandler


def ingest_embed_article_task(**kwargs):
    rss_feeds = RssFetcher(["http://feeds.bbci.co.uk/sport/football/rss.xml"])
    articles = rss_feeds.collect_articles()
    try:
        chunker = TextChunker()
        embedder = TextEmbedder()
        milvus = MilvusHandler()
        for article in articles:
            chunks = chunker.chunk_text(article["text"])
            embeddings = embedder.embed_texts(chunks)
            milvus.insert_chunks(embeddings, chunks, [dict(article) for _ in range(len(chunks))])
    except Exception as e:
        print(f"[INFO] EXECUTION FAILED = {e}")


default_args = {
    'owner': 'airflow',
    'start_date': datetime.datetime.now(),
    'retries': 0
}

with DAG('milvus_ingest_football_news',
         default_args=default_args,
         schedule_interval='0 */3 * * *',
         catchup=False) as dag:

    t1 = PythonOperator(
        task_id="ingest_embed_article_task",
        python_callable=ingest_embed_article_task,
        provide_context=True
    )

    t1
