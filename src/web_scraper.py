import requests
from bs4 import BeautifulSoup
from newspaper import Article
import re
from urllib.parse import urlparse

class NewsWebScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def extract_from_url(self, url):
        """Extract article content from URL"""
        try:
            # Validate URL
            if not self._is_valid_url(url):
                return None, "Invalid URL format"
            
            # Try newspaper3k first (best for news articles)
            try:
                article = Article(url)
                article.download()
                article.parse()
                
                if article.text and len(article.text.strip()) > 100:
                    return {
                        'title': article.title or 'No title found',
                        'text': article.text,
                        'source': self._extract_domain(url),
                        'authors': article.authors,
                        'publish_date': str(article.publish_date) if article.publish_date else None,
                        'url': url
                    }, None
            except Exception as e:
                print(f"Newspaper3k failed: {e}")
            
            # Fallback to BeautifulSoup
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title
                title = self._extract_title(soup)
                
                # Extract main content
                text = self._extract_content(soup)
                
                if text and len(text.strip()) > 100:
                    return {
                        'title': title,
                        'text': text,
                        'source': self._extract_domain(url),
                        'authors': [],
                        'publish_date': None,
                        'url': url
                    }, None
                else:
                    return None, "Could not extract sufficient content from the article"
                    
            except requests.RequestException as e:
                return None, f"Failed to fetch URL: {str(e)}"
            except Exception as e:
                return None, f"Error parsing content: {str(e)}"
                
        except Exception as e:
            return None, f"Unexpected error: {str(e)}"
    
    def _is_valid_url(self, url):
        """Check if URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _extract_domain(self, url):
        """Extract domain name from URL"""
        try:
            domain = urlparse(url).netloc
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return "Unknown"
    
    def _extract_title(self, soup):
        """Extract article title"""
        # Try different title selectors
        title_selectors = [
            'h1',
            '.headline',
            '.title',
            '[data-testid="headline"]',
            '.entry-title',
            '.post-title'
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem and title_elem.get_text().strip():
                return title_elem.get_text().strip()
        
        # Fallback to page title
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        return "No title found"
    
    def _extract_content(self, soup):
        """Extract main article content"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
            element.decompose()
        
        # Try different content selectors
        content_selectors = [
            'article',
            '.article-content',
            '.entry-content',
            '.post-content',
            '.content',
            '[data-testid="article-body"]',
            '.story-body',
            '.article-body'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                text = content_elem.get_text(separator=' ', strip=True)
                if len(text) > 200:  # Minimum content length
                    return self._clean_text(text)
        
        # Fallback: extract all paragraph text
        paragraphs = soup.find_all('p')
        if paragraphs:
            text = ' '.join([p.get_text(strip=True) for p in paragraphs])
            if len(text) > 200:
                return self._clean_text(text)
        
        return None
    
    def _clean_text(self, text):
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common unwanted patterns
        text = re.sub(r'(Subscribe|Sign up|Newsletter|Advertisement|Cookie Policy).*?(?=\.|$)', '', text, flags=re.IGNORECASE)
        
        # Remove very short sentences (likely navigation/UI text)
        sentences = text.split('.')
        cleaned_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        return '. '.join(cleaned_sentences).strip()
    
    def get_source_info(self, url):
        """Get additional source information"""
        domain = self._extract_domain(url)
        
        # Known credible sources
        credible_sources = {
            'reuters.com': 'Reuters',
            'bbc.com': 'BBC News',
            'cnn.com': 'CNN',
            'nytimes.com': 'The New York Times',
            'washingtonpost.com': 'The Washington Post',
            'theguardian.com': 'The Guardian',
            'npr.org': 'NPR',
            'pbs.org': 'PBS',
            'wsj.com': 'The Wall Street Journal',
            'apnews.com': 'Associated Press',
            'nature.com': 'Nature',
            'science.org': 'Science Magazine'
        }
        
        # Known suspicious patterns
        suspicious_patterns = [
            'blog', 'conspiracy', 'truth', 'exposed', 'secret', 'hidden',
            'scam', 'miracle', 'shocking', 'unbelievable'
        ]
        
        source_type = "Unknown"
        if domain in credible_sources:
            source_type = "Credible"
        elif any(pattern in domain.lower() for pattern in suspicious_patterns):
            source_type = "Suspicious"
        
        return {
            'domain': domain,
            'source_type': source_type,
            'display_name': credible_sources.get(domain, domain.title())
        }