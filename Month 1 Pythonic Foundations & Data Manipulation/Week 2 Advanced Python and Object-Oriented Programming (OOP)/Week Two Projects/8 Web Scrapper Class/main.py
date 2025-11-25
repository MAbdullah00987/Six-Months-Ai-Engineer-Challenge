
#Web Scraper Class Project
#Web Scraper Class: Encapsulate web scraping logic into a reusable class

import requests
from main import BeautifulSoup
import json
import csv
import time
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse


class WebScraper:
    """
    A reusable web scraping class that encapsulates common scraping logic.
    
    Features:
    - Fetch web pages with custom headers
    - Parse HTML content
    - Extract data using CSS selectors
    - Save data to JSON or CSV
    - Handle errors and retries
    """
    
    def __init__(self, base_url: str, headers: Optional[Dict] = None, delay: float = 1.0):
        """
        Initialize the web scraper.
        
        Args:
            base_url: The base URL to scrape from
            headers: Optional custom headers for requests
            delay: Delay between requests in seconds (be respectful!)
        """
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        
        # Default headers to mimic a browser
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        self.session.headers.update(self.headers)
        
        self.data = []
        self.soup = None
    
    def fetch_page(self, url: str = None, max_retries: int = 3) -> Optional[str]:
        """
        Fetch a web page and return its HTML content.
        
        Args:
            url: URL to fetch (uses base_url if not provided)
            max_retries: Number of retry attempts on failure
            
        Returns:
            HTML content as string, or None if failed
        """
        target_url = url or self.base_url
        
        for attempt in range(max_retries):
            try:
                print(f"Fetching: {target_url}")
                response = self.session.get(target_url, timeout=10)
                response.raise_for_status()  # Raise exception for bad status codes
                
                time.sleep(self.delay)  # Be respectful to servers
                return response.text
                
            except requests.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print(f"Failed to fetch {target_url} after {max_retries} attempts")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def parse_html(self, html: str) -> BeautifulSoup:
        """
        Parse HTML content using BeautifulSoup.
        
        Args:
            html: HTML content as string
            
        Returns:
            BeautifulSoup object
        """
        self.soup = BeautifulSoup(html, 'lxml')
        return self.soup
    
    def extract_data(self, selector: str, attribute: str = 'text', limit: int = None) -> List:
        """
        Extract data from parsed HTML using CSS selectors.
        
        Args:
            selector: CSS selector to find elements
            attribute: 'text' to get text content, or attribute name (e.g., 'href')
            limit: Maximum number of elements to extract
            
        Returns:
            List of extracted data
        """
        if not self.soup:
            print("Error: HTML not parsed. Call parse_html() first.")
            return []
        
        elements = self.soup.select(selector)
        
        if limit:
            elements = elements[:limit]
        
        extracted = []
        for element in elements:
            if attribute == 'text':
                value = element.get_text(strip=True)
            else:
                value = element.get(attribute)
            
            if value:
                extracted.append(value)
        
        return extracted
    
    def extract_structured_data(self, config: Dict[str, Dict]) -> List[Dict]:
        """
        Extract structured data using multiple selectors.
        
        Args:
            config: Dictionary mapping field names to selector configs
                   Example: {
                       'title': {'selector': 'h2.title', 'attribute': 'text'},
                       'link': {'selector': 'a.link', 'attribute': 'href'}
                   }
        
        Returns:
            List of dictionaries containing structured data
        """
        if not self.soup:
            print("Error: HTML not parsed. Call parse_html() first.")
            return []
        
        # Find the container elements (assuming first selector determines count)
        first_field = list(config.keys())[0]
        containers = self.soup.select(config[first_field]['selector'])
        
        results = []
        for container in containers:
            item = {}
            for field, conf in config.items():
                selector = conf['selector']
                attribute = conf.get('attribute', 'text')
                
                # Try to find element within container first
                element = container.select_one(selector)
                
                if element:
                    if attribute == 'text':
                        item[field] = element.get_text(strip=True)
                    else:
                        item[field] = element.get(attribute)
                else:
                    item[field] = None
            
            results.append(item)
        
        return results
    
    def scrape(self, url: str = None, selector_config: Dict = None) -> List:
        """
        Main scraping method that orchestrates the entire process.
        
        Args:
            url: URL to scrape (uses base_url if not provided)
            selector_config: Configuration for structured data extraction
            
        Returns:
            List of scraped data
        """
        html = self.fetch_page(url)
        if not html:
            return []
        
        self.parse_html(html)
        
        if selector_config:
            self.data = self.extract_structured_data(selector_config)
        
        return self.data
    
    def save_to_json(self, filename: str, data: List = None):
        """Save scraped data to JSON file."""
        data_to_save = data or self.data
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"Data saved to {filename}")
    
    def save_to_csv(self, filename: str, data: List[Dict] = None):
        """Save scraped data to CSV file."""
        data_to_save = data or self.data
        
        if not data_to_save:
            print("No data to save")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data_to_save[0].keys())
            writer.writeheader()
            writer.writerows(data_to_save)
        
        print(f"Data saved to {filename}")
    
    def get_all_links(self, internal_only: bool = True) -> List[str]:
        """
        Extract all links from the current page.
        
        Args:
            internal_only: If True, only return links from the same domain
            
        Returns:
            List of URLs
        """
        if not self.soup:
            return []
        
        links = []
        base_domain = urlparse(self.base_url).netloc
        
        for link in self.soup.find_all('a', href=True):
            url = urljoin(self.base_url, link['href'])
            
            if internal_only:
                if urlparse(url).netloc == base_domain:
                    links.append(url)
            else:
                links.append(url)
        
        return list(set(links))  # Remove duplicates


# Example usage
if __name__ == "__main__":
    # Example 1: Simple scraping
    scraper = WebScraper("https://example.com")
    
    # Fetch and parse a page
    html = scraper.fetch_page()
    if html:
        scraper.parse_html(html)
        
        # Extract all headings
        headings = scraper.extract_data('h1, h2, h3')
        print("Headings:", headings)
        
        # Extract all links
        links = scraper.get_all_links()
        print(f"Found {len(links)} internal links")
    
    # Example 2: Structured data scraping
    # This would work on a real e-commerce or blog site
    """
    scraper = WebScraper("https://example-shop.com/products")
    
    config = {
        'title': {'selector': '.product-title', 'attribute': 'text'},
        'price': {'selector': '.product-price', 'attribute': 'text'},
        'link': {'selector': 'a.product-link', 'attribute': 'href'},
        'image': {'selector': 'img.product-image', 'attribute': 'src'}
    }
    
    products = scraper.scrape(selector_config=config)
    
    # Save results
    scraper.save_to_json('products.json', products)
    scraper.save_to_csv('products.csv', products)
    """