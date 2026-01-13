from readability import Document
import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
from urllib.parse import urlparse, urljoin
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import dateutil.parser
import logging
import fitz  # PyMuPDF
import os
import yaml
from typing import List, Dict, Optional, Union
from collections import defaultdict
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentScraper:
    def __init__(self, headless=True, wait_time=10):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.headless = headless
        self.wait_time = wait_time
        self.driver = None
        

    def _setup_driver(self):
            """Setup Selenium WebDriver for dynamic content"""
            if self.driver is None:
                chrome_options = Options()
                if self.headless:
                    chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument("--window-size=1920,1080")
                chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
                
                self.driver = webdriver.Chrome(options=chrome_options)
                self.driver.implicitly_wait(self.wait_time)

    
    def _close_driver(self):
        """Close Selenium WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def _identify_platform(self, url):
        """Identify the platform/website type"""
        domain = urlparse(url).netloc.lower()
        
        social_platforms = {
            'twitter.com': 'Twitter',
            'x.com': 'Twitter',
            'facebook.com': 'Facebook',
            'instagram.com': 'Instagram',
            'linkedin.com': 'LinkedIn',
            'youtube.com': 'YouTube',
            'tiktok.com': 'TikTok',
            'reddit.com': 'Reddit'
        }
        
        for platform_domain, platform_name in social_platforms.items():
            if domain == platform_domain or domain.endswith(f".{platform_domain}"):
                return platform_name, True

        return domain, False
    

    def _extract_readable_content(self, html):
        try:
            doc = Document(html)
            title = doc.short_title()
            summary_html = doc.summary()

            soup = BeautifulSoup(summary_html, 'html.parser')
            lines = []

            for elem in soup.find_all(['h1', 'h2', 'h3','h4','h5','h6' ,'p']):
                text = elem.get_text(strip=True)
                if not text:
                    continue
                if elem.name in ['h1', 'h2', 'h3','h4','h5','h6']:
                    lines.append(f"\n\n## {text}")
                else:
                    lines.append(text)

            full_text = '\n\n'.join(lines)
            return title, full_text
        except Exception as e:
            logger.warning(f"Readability failed: {e}")
            return "", ""


    
    def _extract_with_requests(self, url):
        """Try to extract content using requests + BeautifulSoup"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.warning(f"Requests extraction failed: {e}")
            return None
    
    def _extract_with_selenium(self, url):
        try:
            self._setup_driver()
            self.driver.get(url)

            WebDriverWait(self.driver, self.wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            time.sleep(3)  # wait for JS content

            # Click read more buttons if present
            self._click_read_more_buttons()

            return BeautifulSoup(self.driver.page_source, 'html.parser')
        except Exception as e:
            logger.warning(f"Selenium extraction failed: {e}")
            return None



    def _click_read_more_buttons(self):
        """Click any 'Read More' buttons like Taboola, etc."""
        try:
            # Match buttons that expand content, like Taboola
            read_more_buttons = self.driver.find_elements(By.CSS_SELECTOR, 'a.tbl-read-more-btn')
            for btn in read_more_buttons:
                if btn.is_displayed() and btn.is_enabled():
                    try:
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", btn)
                        time.sleep(1)
                        btn.click()
                        time.sleep(2)  # Wait for content to load
                    except Exception as click_err:
                        logger.warning(f"Could not click read-more button: {click_err}")
        except Exception as e:
            logger.warning(f"Error while trying to click 'Read More': {e}")

    
    def _extract_meta_tags(self, soup):
        """Extract metadata from meta tags"""
        meta_data = {}
        
        # Common meta tags
        meta_mappings = {
            'og:title': 'title',
            'twitter:title': 'title',
            'og:description': 'description',
            'twitter:description': 'description',
            'og:site_name': 'site_name',
            'og:url': 'canonical_url',
            'article:published_time': 'published_time',
            'article:author': 'author',
            'og:type': 'content_type'
        }
        
        for meta_tag in soup.find_all('meta'):
            property_val = meta_tag.get('property') or meta_tag.get('name')
            content = meta_tag.get('content')
            
            if property_val and content:
                if property_val in meta_mappings:
                    meta_data[meta_mappings[property_val]] = content
        
        return meta_data
    
    def _parse_date(self, date_string):
        """Parse various date formats"""
        if not date_string:
            return None
        
        try:
            # Try parsing with dateutil (handles most formats)
            parsed_date = dateutil.parser.parse(date_string)
            return parsed_date.isoformat()
        except:
            # Try common patterns
            patterns = [
                r'\d{4}-\d{2}-\d{2}',
                r'\d{2}/\d{2}/\d{4}',
                r'\d{2}-\d{2}-\d{4}',
                r'\w+ \d{1,2}, \d{4}'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, date_string)
                if match:
                    try:
                        parsed_date = dateutil.parser.parse(match.group())
                        return parsed_date.isoformat()
                    except:
                        continue
        
        return date_string  # Return original if parsing fails


    def _extract_text_from_element(self, element):
        lines = []

        if element.name in ['h1', 'h2', 'h3', 'h4']:
            lines.append(f"\n\n## {element.get_text(strip=True)}")
        elif element.name == 'li':
            lines.append(f"- {element.get_text(strip=True)}")
        elif element.name == 'p':
            lines.append(element.get_text(strip=True))
        elif element.name in ['ul', 'ol', 'div']:
            # Recursively process only direct children
            for child in element.find_all(recursive=False):
                lines.extend(self._extract_text_from_element(child))

        return lines


    def _extract_authors(self, soup):
        """Extract and clean author names."""
        author_selectors = [
            '[class*="byline"]',
            '[class*="author"]',
            '[itemprop="author"]',
            'a[href*="/author/"]',
        ]

        authors = set()

        for selector in author_selectors:
            for tag in soup.select(selector):
                text = tag.get_text(" ", strip=True)
                if not text:
                    continue

                cleaned = (
                    text.replace("By ", "")
                    .replace("BY ", "")
                    .replace("by ", "")
                    .replace(" and ", ",")
                    .replace("&", ",")
                )

                for part in cleaned.split(","):
                    part = part.strip()
                    if part and len(part.split()) <= 4:
                        authors.add(part.title())

        return sorted(authors)
    
    def _classify_content(self, text, platform, soup, extract_tags_enabled=True):
        """Classify content type and extract tags"""
        text_lower = text.lower() if text else ""
        classification = {
            'content_type': 'unknown',
            'category': 'general',
            'tags': []
        }
        
        # Content type classification
        if platform in ['Twitter', 'Facebook', 'Instagram', 'TikTok']:
            classification['content_type'] = 'social_media_post'
            classification['category'] = 'social'
        elif platform == 'Reddit':
            classification['content_type'] = 'forum_post'
            classification['category'] = 'discussion'
        elif platform == 'YouTube':
            classification['content_type'] = 'video'
            classification['category'] = 'media'
        elif len(text) > 2000:
            classification['content_type'] = 'article'
            classification['category'] = 'news' if 'news' in text_lower else 'blog'
        elif len(text) > 500:
            classification['content_type'] = 'blog_post'
            classification['category'] = 'blog'
        else:
            classification['content_type'] = 'short_content'
            classification['category'] = 'general'
        
        # Extract tags/keywords if enabled
        if extract_tags_enabled:
            tags = set()
            
            # Extract from meta tags
            for meta_tag in soup.find_all('meta'):
                property_val = meta_tag.get('property') or meta_tag.get('name', '')
                content = meta_tag.get('content', '')
                
                if property_val in ['keywords', 'article:tag']:
                    tags.update([tag.strip() for tag in content.split(',') if tag.strip()])
            
            # Add platform as tag
            if platform:
                tags.add(platform.lower())
            
            classification['tags'] = sorted(list(tags)[:10])  # Limit to 10 tags
        
        return classification


    def _extract_article_content(self, soup, platform):
        """Extract structured article content without duplication, handling nested structures."""
        # --- inside _extract_article_content ---------------------------------
        content_selectors = [
            '[itemprop="articleBody"]',
            '[class*="article-content"]',
            '[class*="article-body"]',
            '[class*="story-body"]',
            '[class*="story-content"]',
            '[id^="story-content-"]',
            '[id^="article-content"]',
            '[class*="entryContent"]',
            '[class*="abstract"]',
            '[class*="container"]',
            '[id^="bodyContent"]',
            '[class*="story-section"]',
            '[class*="post-content"]',
            '[class*="wysiwyg"]',
            '[class*="primary"]',
            '[class*="article-body__content__17Yit"]'
            '[class*="responsiveSkin ifp-doc-type-oxencycl-entry"]',
            '[class*="text-component"]',
            '[class*="entry-content"]',
            '[class*="e-tab-content tab-content"]'
            '[class*="e-content-block"]'
            '[class*="elementor-post-content"]',      # WordPress/Elementor
            '[class*="elementor-widget-container"]',  # ← add this
            '[class*="elementor-element elementor-element-d72ee69 single-post-paragraph elementor-widget elementor-widget-text-editor"]',
            'article',
            'main'
        ]

        best_block = None
        max_score = 0

        for selector in content_selectors:
            for element in soup.select(selector):
                for tag in element(['script', 'style', 'nav', 'footer', 'aside', 'form', 'iframe', '.adsbygoogle']):
                    tag.decompose()

                score = len(element.find_all('p')) + len(element.find_all(['h2', 'h3', 'li']))
                if score > max_score:
                    best_block = element
                    max_score = score

        if not best_block:
            return ""

        lines = []
        processed = set()

        for tag in best_block.find_all(['h1', 'h2', 'h3', 'li', 'p', 'ul', 'ol'], recursive=True):
            text = tag.get_text(strip=True)
            if text and text not in processed:
                if tag.name in ['h1', 'h2', 'h3']:
                    lines.append(f"\n\n## {text}")
                elif tag.name == 'li':
                    lines.append(f"- {text}")
                else:
                    lines.append(text)
                processed.add(text)

        return '\n'.join(lines)


    def _extract_twitter_content(self, soup):
        """Extract Twitter-specific content"""
        data = {}
        
        # Twitter selectors (may need updates as Twitter changes)
        tweet_selectors = [
            '[data-testid="tweetText"]',
            '.tweet-text',
            '[data-testid="tweet"]'
        ]
        
        for selector in tweet_selectors:
            elements = soup.select(selector)
            if elements:
                data['text'] = ' '.join([elem.get_text(strip=True) for elem in elements])
                break
        
        # Extract author
        author_selectors = [
            '[data-testid="User-Name"]',
            '.username',
            '.fullname'
        ]
        
        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                data['author'] = element.get_text(strip=True)
                break

        return data

    def _extract_facebook_content(self, soup):
        """Extract Facebook-specific content"""
        data = {}
        
        # Facebook post content
        post_selectors = [
            '[data-ad-preview="message"]',
            '.userContent',
            '[data-testid="post_message"]'
        ]
        
        for selector in post_selectors:
            element = soup.select_one(selector)
            if element:
                data['text'] = element.get_text(strip=True)
                break

        return data


    
    def _extract_reddit_content(self, soup):
        """Extract Reddit-specific content"""
        data = {}

        def safe_get(selector_list, attr=None, text=True):
            for selector in selector_list:
                element = soup.select_one(selector)
                if element:
                    if attr:
                        value = element.get(attr)
                        if value:
                            return value.strip()
                    elif text:
                        return element.get_text(strip=True)
            return ''

        # Reddit post title from new Reddit format (shreddit-title tag)
        data['title'] = safe_get(['shreddit-title'])

        # Reddit post text (if available)
        data['text'] = safe_get([
            '[data-test-id="post-content"] div[class*="text"]',
            '.Post div[class*="text"]',
            '[data-click-id="text"] div',
            'div[class*="usertext-body"]',
            'shreddit-post[post-title]'  # Custom attribute
        ], attr='post-title', text=False)

        # Reddit author (from custom attribute)
        data['author'] = safe_get([
            'shreddit-post'
        ], attr='author', text=False)

        # Subreddit name
        data['subreddit'] = safe_get([
            'shreddit-post'
        ], attr='subreddit-name', text=False)

        return data


    def _extract_pdf_content(self, url, extract_tags_enabled=True):
        logger.info(f"Downloading PDF: {url}")
        response = self.session.get(url)
        response.raise_for_status()

        with open("temp.pdf", "wb") as f:
            f.write(response.content)

        text = ""
        doc = fitz.open("temp.pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
        os.remove("temp.pdf")
        
        scrape_timestamp = datetime.now().isoformat()
        platform = urlparse(url).netloc
        
        # Create empty soup for classification
        dummy_soup = BeautifulSoup("<html></html>", 'html.parser')
        
        classification = self._classify_content(text, platform, dummy_soup, extract_tags_enabled)
        classification['content_type'] = 'document'
        classification['category'] = 'document'
        if 'pdf' not in classification['tags']:
            classification['tags'].append('pdf')

        return {
            'url': url,
            'platform': platform,
            'is_social_media': False,
            'title': "",
            'text': text.strip(),
            'author': "",
            'publisher': platform,
            'published_date': None,
            'site_name': platform,
            'data_source': 'pdf',
            'scrape_timestamp': scrape_timestamp,
            'extraction_method': 'pdf_extractor',
            'content_classification': classification,
            'metadata': {
                'author': '',
                'publisher': platform,
                'platform': platform,
                'data_source': 'pdf',
                'scrape_timestamp': scrape_timestamp,
                'extraction_method': 'pdf_extractor',
                'published_date': None,
                'content_classification': classification
            }
        }

    def scrape_content(self, url, classify_content=True, extract_tags=True):
        """Main scraping function"""
        logger.info(f"Starting to scrape: {url}")
        scrape_timestamp = datetime.now().isoformat()
        
        try:
            head = self.session.head(url, allow_redirects=True, timeout=10)
            content_type = head.headers.get('Content-Type', '').lower()
            if content_type.startswith('application/pdf') or url.lower().endswith('.pdf'):
                return self._extract_pdf_content(url, extract_tags)
        except Exception as e:
            logger.warning(f"Failed to check content type: {e}")

        # Identify platform
        platform, is_social = self._identify_platform(url)
        
        # Try requests first, then Selenium if needed
        soup = self._extract_with_requests(url)
        
        if not soup or is_social:
            logger.info("Using Selenium for dynamic content extraction")
            soup = self._extract_with_selenium(url)
        
        if not soup:
            raise Exception("Failed to extract content from URL")
        
        # Extract meta data
        meta_data = self._extract_meta_tags(soup)

        # Initialize result structure
        result = {
            'url': url,
            'platform': platform,
            'is_social_media': is_social,
            'title': '',
            'text': '',
            'author': '',
            'publisher': '',
            'published_date': None,
            'site_name': platform,
            'data_source': 'web_scraper',
            'scrape_timestamp': scrape_timestamp,
            'extraction_method': 'selenium' if (not soup or is_social) else 'requests'
        }
        
        # Extract title
        title_sources = [
            meta_data.get('title'),
            soup.find('title').get_text(strip=True) if soup.find('title') else None,
            soup.find('h1').get_text(strip=True) if soup.find('h1') else None
        ]
        
        for title_source in title_sources:
            if title_source and title_source.strip():
                result['title'] = title_source.strip()
                break
        
        # Platform-specific extraction
        if platform == 'Twitter':
            twitter_data = self._extract_twitter_content(soup)
            result.update(twitter_data)
        elif platform == 'Facebook':
            facebook_data = self._extract_facebook_content(soup)
            result.update(facebook_data)
        elif platform == 'Reddit':
            reddit_data = self._extract_reddit_content(soup)
            result.update(reddit_data)
        else:
            # Generic article extraction
            result['text'] = self._extract_article_content(soup, platform)
            result['author'] = ', '.join(self._extract_authors(soup))
            if not result['text'] or len(result['text']) < 200:
                title, text = self._extract_readable_content(str(soup))
                if title and not result['title']:
                    result['title'] = title
                if text:
                    result['text'] = text

        # For non-social media, try to find publisher
        if not is_social:
            result['publisher'] = meta_data.get('site_name') or platform
        
        # Extract author from meta tags if not found
        author_from_meta = meta_data.get('author', '')
        if not result.get('author') and author_from_meta:
            result['author'] = author_from_meta
        
        # Extract date
        date_sources = [
            meta_data.get('published_time'),
            soup.select_one('time')['datetime'] if soup.select_one('time') and soup.select_one('time').get('datetime') else None,
            soup.select_one('.date, .publish-date, .post-date').get_text(strip=True) if soup.select_one('.date, .publish-date, .post-date') else None
        ]
        
        for date_source in date_sources:
            if date_source:
                parsed_date = self._parse_date(date_source)
                if parsed_date:
                    result['published_date'] = parsed_date
                    break
        
        # Add content classification and tags if enabled
        if classify_content:
            result['content_classification'] = self._classify_content(
                result.get('text', ''), platform, soup, extract_tags
            )
        else:
            result['content_classification'] = {
                'content_type': 'unknown',
                'category': 'general',
                'tags': []
            }
        
        # Add comprehensive metadata
        # result['metadata'] = {
        #     'author': result.get('author', ''),
        #     'publisher': result.get('publisher', ''),
        #     'platform': platform,
        #     'data_source': result.get('data_source', 'web_scraper'),
        #     'scrape_timestamp': scrape_timestamp,
        #     'extraction_method': result.get('extraction_method', 'unknown'),
        #     'published_date': result.get('published_date'),
        #     'content_classification': result.get('content_classification', {})
        # }
        result['metadata'] = {}
        
        # Clean up driver
        self._close_driver()
        
        return result
    
    def save_to_json(self, data, filename=None, output_dir=None):
        """Save scraped data to JSON file"""
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if filename:
                filename = os.path.join(output_dir, filename)
            else:
                filename = os.path.join(output_dir, "data.json")
        elif filename is None:
            filename = "data.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Data saved to {filename}")
        return filename

def scrape_url(url, output_file=None, output_dir=None, classify_content=True, extract_tags=True):
    """Convenience function to scrape a single URL"""
    scraper = ContentScraper()
    try:
        result = scraper.scrape_content(url, classify_content=classify_content, extract_tags=extract_tags)
        filename = scraper.save_to_json(result, output_file, output_dir)
        return result, filename
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        raise


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================
class ScrapingConfig:
    """Load scraping configuration from YAML file."""
    
    def __init__(self, config_path: str = "scraping_config.yaml"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        # Default config path relative to script location
        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, config_path)
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Load input settings
            input_config = config.get('input', {})
            self.URLS_FILE = input_config.get('urls_file')
            self.URLS = input_config.get('urls', [])
            
            # Load output settings
            output_config = config.get('output', {})
            self.OUTPUT_DIR = output_config.get('output_dir', 'scraping_output')
            self.OUTPUT_FILE = output_config.get('file', 'scraping_results.json')
            self.CHECKPOINT_FILE = output_config.get('checkpoint_file', 'scraping_checkpoint.json')
            
            # Load pipeline settings
            pipeline_config = config.get('pipeline', {})
            self.PARALLEL = pipeline_config.get('parallel', False)
            self.MAX_WORKERS = pipeline_config.get('max_workers', 3)
            self.MAX_RETRIES = pipeline_config.get('max_retries', 3)
            self.RETRY_DELAY = pipeline_config.get('retry_delay', 2.0)
            self.RATE_LIMIT_DELAY = pipeline_config.get('rate_limit_delay', 1.0)
            self.RESUME = pipeline_config.get('resume', True)
            
            # Load scraper settings
            scraper_config = config.get('scraper', {})
            self.HEADLESS = scraper_config.get('headless', True)
            self.WAIT_TIME = scraper_config.get('wait_time', 10)
            self.USER_AGENT = scraper_config.get('user_agent', 
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            
            # Load extraction settings
            extraction_config = config.get('extraction', {})
            self.CLASSIFY_CONTENT = extraction_config.get('classify_content', True)
            self.EXTRACT_TAGS = extraction_config.get('extract_tags', True)
            self.EXTRACT_METADATA = extraction_config.get('extract_metadata', True)
            self.MIN_TEXT_LENGTH = extraction_config.get('min_text_length', 200)
            
            logger.info(f"Configuration loaded from: {config_path}")
        else:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            # Set defaults
            self.URLS_FILE = None
            self.URLS = []
            self.OUTPUT_DIR = 'scraping_output'
            self.OUTPUT_FILE = 'scraping_results.json'
            self.CHECKPOINT_FILE = 'scraping_checkpoint.json'
            self.PARALLEL = False
            self.MAX_WORKERS = 3
            self.MAX_RETRIES = 3
            self.RETRY_DELAY = 2.0
            self.RATE_LIMIT_DELAY = 1.0
            self.RESUME = True
            self.HEADLESS = True
            self.WAIT_TIME = 10
            self.USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            self.CLASSIFY_CONTENT = True
            self.EXTRACT_TAGS = True
            self.EXTRACT_METADATA = True
            self.MIN_TEXT_LENGTH = 200


# Example usage
if __name__ == "__main__":
    # Load configuration
    config = ScrapingConfig("scraping_config.yaml")
    
    # Determine URLs to scrape
    urls_to_scrape = []
    
    if config.URLS_FILE and os.path.exists(config.URLS_FILE):
        logger.info(f"Reading URLs from file: {config.URLS_FILE}")
        with open(config.URLS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    urls_to_scrape.append(line)
    elif config.URLS:
        logger.info(f"Using URLs from config ({len(config.URLS)} URLs)")
        urls_to_scrape = config.URLS
    else:
        # Fallback to interactive input
        url = input("Please enter a URL to scrape: ")
        urls_to_scrape = [url]
    
    if not urls_to_scrape:
        logger.error("No URLs provided. Please configure URLs in scraping_config.yaml or provide via input.")
        exit(1)
    
    # Create scraper with config settings
    scraper = ContentScraper(
        headless=config.HEADLESS,
        wait_time=config.WAIT_TIME
    )
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    
    # Scrape all URLs
    results = []
    for i, url in enumerate(urls_to_scrape, 1):
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Scraping URL {i}/{len(urls_to_scrape)}: {url}")
            logger.info(f"{'='*60}")
            
            result = scraper.scrape_content(
                url,
                classify_content=config.CLASSIFY_CONTENT,
                extract_tags=config.EXTRACT_TAGS
            )
            
            results.append(result)
            
            # Print summary
            print(f"\n✅ Successfully scraped:")
            print(f"  Platform: {result['platform']}")
            print(f"  Title: {result['title'][:100] if result.get('title') else 'N/A'}...")
            print(f"  Text length: {len(result.get('text', ''))} characters")
            print(f"  Author: {result.get('author', 'N/A')}")
            print(f"  Published: {result.get('published_date', 'N/A')}")
            if result.get('content_classification'):
                classification = result['content_classification']
                print(f"  Content Type: {classification.get('content_type', 'N/A')}")
                print(f"  Category: {classification.get('category', 'N/A')}")
                if classification.get('tags'):
                    print(f"  Tags: {', '.join(classification['tags'][:5])}")
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            results.append({
                'url': url,
                'error': str(e),
                'scrape_timestamp': datetime.now().isoformat()
            })
    
    # Save all results to JSON file in output directory
    output_file = os.path.join(config.OUTPUT_DIR, config.OUTPUT_FILE)
    scraper.save_to_json(
        results if len(results) > 1 else results[0] if results else {},
        config.OUTPUT_FILE,
        config.OUTPUT_DIR
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Scraping completed!")
    logger.info(f"  Total URLs: {len(urls_to_scrape)}")
    logger.info(f"  Successful: {len([r for r in results if 'error' not in r])}")
    logger.info(f"  Failed: {len([r for r in results if 'error' in r])}")
    logger.info(f"  Results saved to: {output_file}")
    logger.info(f"{'='*60}")

