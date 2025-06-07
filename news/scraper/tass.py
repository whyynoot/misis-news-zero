import requests
import time
import random

class TassScraper:
    def __init__(self):
        self.base_url = "https://tass.ru/tbp/api/v1/content"
        self.parse_pages = 10


    def get_news(self):
        titles_and_leads = []
        """Получение новостей агентсва ТАСС"""
        params = {
            'limit': 20,
            'offset': 0,
            'lang': 'ru',
            'rubrics': random.choice(['/socialnaya-zaschita', '/zdorove', '/demografiya', '/nacionalnye-proekty', '/ekonomika']),
            'sort': '-es_updated_dt'
        }
        pages = self.parse_pages
        while pages > 0:

            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()

                news_items = data.get('result', [])
                for item in news_items:
                    if 'title' in item and 'lead' in item:
                        titles_and_leads.append(f"{item['title']}: {item['lead']}")
            
                params['offset'] += 20
                pages -= 1

            except requests.exceptions.RequestException as e:
                print(f"Error fetching news from TASS API: {e}")
                return []
        
        return titles_and_leads