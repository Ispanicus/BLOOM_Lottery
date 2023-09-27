import requests
from bs4 import BeautifulSoup

def scrape_urls():
    # Send GET request
    website_urls = [
        'https://es.wikipedia.org/wiki/Wikipedia:Art%C3%ADculos_buenos',
        'https://es.wikipedia.org/wiki/Wikipedia:Art%C3%ADculos_destacados',
    ]

    all_urls = []

    for website_url in website_urls:
        response = requests.get(website_url)

        # If the GET request is successful, the status code will be 200
        if response.status_code != 200:
            raise ValueError(f"Failed to retrieve the web page. Status code: {response.status_code}")
        
        # Create a BeautifulSoup object and specify the parser
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all <a> tags
        a_tags = soup.find_all('a')
        
        # Extract href attributes and store them in a list
        urls = [f"https://es.wikipedia.org{link.get('href')}" for link in a_tags if link.get('href')]
        
        all_urls.extend(urls)
    
    with open('wiki.txt', 'w') as file:
            for url in all_urls:
                file.write(url + '\n')

if __name__ == "__main__":
    urls = scrape_urls()

