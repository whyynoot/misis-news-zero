from news.scraper.tass import TassScraper

scrappers = {
    "tass": TassScraper(),
}

def get_scrappers():
    return scrappers