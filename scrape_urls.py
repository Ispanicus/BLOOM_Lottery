import requests
from bs4 import BeautifulSoup

def scrape_urls(website_url):
    # Send GET request
    response = requests.get(website_url)
    
    # If the GET request is successful, the status code will be 200
    if response.status_code != 200:
        raise ValueError(f"Failed to retrieve the web page. Status code: {response.status_code}")
    
    # Create a BeautifulSoup object and specify the parser
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all <a> tags
    a_tags = soup.find_all('a')
    
    # Extract href attributes and store them in a list
    urls = [link.get('href') for link in a_tags if link.get('href')]
    
    return set(urls)

if __name__ == "__main__":
    website_url = input("Enter the website URL: ")
    urls = scrape_urls(website_url)
    print("\nExtracted URLs:")
    for url in urls:
        print(url)


