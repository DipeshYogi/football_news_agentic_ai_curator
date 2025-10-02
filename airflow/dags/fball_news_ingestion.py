import time
import random
import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

import feedparser
from newspaper import Article


def scrape_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return {
            "title": article.title,
            "url": url,
            "published": article.publish_date,
            "text": article.text
        }
    except Exception as e:
        print(f"[ERROR] Failed to scrape {url}: {e}")
        return None

def insert_articles_to_db():
    RSS_FEEDS = [
        "http://feeds.bbci.co.uk/sport/football/rss.xml",
        "https://www.espn.com/espn/rss/soccer/news",
        "https://www.transfermarkt.co.uk/rss/news",
        "https://www.theguardian.com/football/rss",
        "https://www.eyefootball.com/football_news.xml"
    ]

    articles = []

    for feed_url in RSS_FEEDS:
        print(f"\nFetching RSS feed: {feed_url}")
        feed = feedparser.parse(feed_url)
        
        for entry in feed.entries:  # limit to top 5 articles per feed for demo
            url = entry.link
            print(f"\n[INFO] Scraping article: {url}")
            
            article_data = scrape_article(url)
            if article_data:
                articles.append({
                    "title": article_data['title'],
                    "published": datetime.datetime.now(),
                    "content": article_data['text']
                })

            sleep_time = random.uniform(1, 5)
            print(f"[INFO] Sleeping for {sleep_time:.2f} seconds to avoid getting blocked...")
            time.sleep(sleep_time)

    # Connect to Postgres
    hook = PostgresHook(postgres_conn_id='fball_postgres')

    for article in articles:
        try:
            hook.run(
                """
                INSERT INTO football_news (title, published, content) 
                VALUES (%s, %s, %s)
                """,
                parameters=(article['title'], article['published'], article['content'])
            )
        except Exception as e:
            pass

default_args = {
    'owner': 'airflow',
    'start_date': datetime.datetime(2023,1,1),
    'retries': 1
}

with DAG('ingest_football_news',
         default_args=default_args,
         schedule_interval='0 */3 * * *',
         catchup=False) as dag:

    ingest_task = PythonOperator(
        task_id='ingest_articles',
        python_callable=insert_articles_to_db
    )
