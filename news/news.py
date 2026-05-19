from news.sources import get_available_scrapers, get_scrapers

scrapers = get_available_scrapers()
scrappers = scrapers


def get_news_scrapers(source_names=None):
    if source_names is None:
        return get_scrapers()
    return get_scrapers(source_names)


def get_scrappers(source_names=None):
    return get_news_scrapers(source_names)
