import time
import random
from datetime import datetime

import feedparser
from newspaper import Article


class RssFetcher:
    def __init__(self, rss_feeds):
        self.rss_feed = rss_feeds

    def scrape_articles(self, url):
        try:
            article = Article(url)
            article.download()
            article.parse()
            return {
                "title": article.title,
                "text": article.text
            }
        except Exception as e:
            print(f"[ERROR] Failed to scrape {url}: {e}")
            return None
    
    def collect_articles(self):
        articles = []

        for feed_url in self.rss_feed:
            print(f"\n[INFO] Fetching RSS feed: {feed_url}")
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries:
                url = entry.link
                print(f"\n[INFO] Scraping article: {url}")
                
                article_data = self.scrape_articles(url)
                if article_data:
                    articles.append({
                        "title": article_data['title'],
                        "date": datetime.now(),
                        "text": article_data['text']
                    })

                sleep_time = random.uniform(1, 5)
                time.sleep(sleep_time)
        
        return articles