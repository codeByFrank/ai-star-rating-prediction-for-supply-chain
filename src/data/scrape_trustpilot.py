"""
Trustpilot Review Scraper for Customer Satisfaction Analysis
=========================================================

This script scrapes English reviews from Trustpilot for e-commerce platforms
like Temu, AliExpress, and Wish for star rating prediction analysis.

Author: Data Science Team: Sebastian, Frank, and Mohamed
Project: Customer Rating Prediction using NLP
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
import logging
from typing import Dict, List, Optional
import sys
import lxml
from pathlib import Path

# =============================================================================
# CONFIGURATION CONSTANTS 
# =============================================================================

# Target platform (change to 'aliexpress', 'wish', or 'temu')
TARGET_PLATFORM = 'temu'

# Platform URLs
PLATFORM_URLS = {
    'temu': 'https://www.trustpilot.com/review/temu.com',
    'aliexpress': 'https://www.trustpilot.com/review/aliexpress.com',
    'wish': 'https://www.trustpilot.com/review/wish.com'
}

# Scraping settings
MAX_PAGES = None  # Maximum pages to scrape (fallback limit)
MAX_REVIEWS = None  # Maximum number of reviews to scrape
PROGRESS_INTERVAL = 1000  # Show progress every N reviews
DELAY_BETWEEN_REQUESTS = 1  # Seconds between requests 
REQUEST_TIMEOUT = 10  # Request timeout in seconds

# =============================================================================
# OUTPUT / LOG PATHS 
# =============================================================================

# All paths are relative to src/data/ (where this script lives)
BASE_DIR = Path(__file__).resolve().parent       # …/src/data
RAW_DIR  = BASE_DIR / "raw"                      # CSV goes here
LOG_DIR  = BASE_DIR / "logs"                     # log files go here
RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Output settings
OUTPUT_FILE = RAW_DIR / f"{TARGET_PLATFORM}_reviews.csv"
LOG_FILE    = LOG_DIR / f"{TARGET_PLATFORM}_scraping.log"

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_date(date_str: str) -> Optional[str]:
    """
    Parse datetime and return ISO-8601 formatted string.
    Keeps full ISO-8601 format with T-separator and timezone for strict conformance.
    
    Args:
        date_str: String like '2025-07-06T10:05:59.000Z', '2 days ago', etc.
        
    Returns:
        ISO-8601 formatted string or None if parsing fails
    """
    if not date_str:
        return None
        
    date_str = date_str.strip()
    
    # If already ISO-8601 format, return as-is (preserve full format)
    iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z?$'
    if re.match(iso_pattern, date_str, re.IGNORECASE):
        return date_str
    
    # Handle relative dates by converting to ISO format
    date_str_lower = date_str.lower()
    now = datetime.now()
    
    # Handle specific patterns
    if 'hour' in date_str_lower:
        match = re.search(r'(\d+)', date_str_lower)
        if match:
            hours = int(match.group(1))
            dt = now - timedelta(hours=hours)
            return dt.isoformat() + 'Z'
    elif 'day' in date_str_lower:
        match = re.search(r'(\d+)', date_str_lower)
        if match:
            days = int(match.group(1))
            dt = now - timedelta(days=days)
            return dt.isoformat() + 'Z'
    elif 'week' in date_str_lower:
        match = re.search(r'(\d+)', date_str_lower)
        if match:
            weeks = int(match.group(1))
            dt = now - timedelta(weeks=weeks)
            return dt.isoformat() + 'Z'
    elif 'month' in date_str_lower:
        match = re.search(r'(\d+)', date_str_lower)
        if match:
            months = int(match.group(1))
            dt = now - timedelta(days=months * 30)  # Approximate
            return dt.isoformat() + 'Z'
    elif 'year' in date_str_lower:
        match = re.search(r'(\d+)', date_str_lower)
        if match:
            years = int(match.group(1))
            dt = now - timedelta(days=years * 365)  # Approximate
            return dt.isoformat() + 'Z'
    
    # Try common date formats and convert to ISO
    try:
        date_formats = [
            '%B %d, %Y',
            '%b %d, %Y',
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%d-%m-%Y',
            '%Y/%m/%d'
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.isoformat() + 'Z'
            except ValueError:
                continue
                
    except Exception:
        pass
    
    logger.warning(f"Could not parse date: {date_str}")
    return None


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters that might break CSV
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Remove HTML entities
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    return text


def extract_user_id(user_link: str) -> str:
    """Extract user ID from profile link."""
    if not user_link:
        return ""
    
    # Extract ID from URL patterns like /users/abc123
    match = re.search(r'/users/([^/]+)', user_link)
    if match:
        return match.group(1)
    return ""


def extract_star_rating(element) -> str:
    """Extract star rating from various possible elements."""
    if not element:
        return ""
    
    # Look for img alt text first
    img = element.find('img')
    if img and img.get('alt'):
        alt_text = img['alt'].lower()
        match = re.search(r'(\d+)', alt_text)
        if match:
            return match.group(1)
    
    # Look for aria-label
    if element.get('aria-label'):
        aria_text = element['aria-label'].lower()
        match = re.search(r'(\d+)', aria_text)
        if match:
            return match.group(1)
    
    # Look for data attributes
    for attr in ['data-rating', 'data-star', 'data-score']:
        if element.get(attr):
            match = re.search(r'(\d+)', element[attr])
            if match:
                return match.group(1)
    
    # Look for title attribute
    if element.get('title'):
        title_text = element['title'].lower()
        match = re.search(r'(\d+)', title_text)
        if match:
            return match.group(1)
    
    return ""


# =============================================================================
# MAIN SCRAPING CLASS
# =============================================================================

class TrustpilotScraper:
    """Main scraper class for Trustpilot reviews."""
    
    def __init__(self, platform: str):
        self.platform = platform
        self.base_url = PLATFORM_URLS.get(platform)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        self.reviews_data = []
        
    def get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a single page."""
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'lxml')
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def extract_review_data(self, review_element) -> Dict:
        """Extract all review data from a single review element using updated selectors."""
        
        # Initialize review data dictionary with ISO-8601 string fields for dates
        review_data = {
            'UserId': '',
            'UserName': '',
            'UserCountry': '',
            'ReviewCount': '',
            'ReviewRating': '',
            'ReviewTitle': '',
            'ReviewText': '',
            'ReviewDate': None,
            'ReviewExperienceDate': None,
            'ReplyText': '',
            'ReplyDate': None
        }
        
        try:
            # User information, using updated selectors based on HTML structure
            user_info = review_element.find('aside', class_=re.compile(r'styles_consumerInfoWrapper'))
            if user_info:
                # User name
                user_name = user_info.find('span', {'data-consumer-name-typography': 'true'})
                if user_name:
                    review_data['UserName'] = clean_text(user_name.get_text())
                
                # User profile link for ID
                user_link = user_info.find('a', {'data-consumer-profile-link': 'true'})
                if user_link and user_link.get('href'):
                    review_data['UserId'] = extract_user_id(user_link['href'])
                
                # User country
                user_country = user_info.find('span', {'data-consumer-country-typography': 'true'})
                if user_country:
                    review_data['UserCountry'] = clean_text(user_country.get_text())
                
                # Review count
                review_count = user_info.find('span', {'data-consumer-reviews-count-typography': 'true'})
                if review_count:
                    count_text = review_count.get_text()
                    match = re.search(r'(\d+)', count_text)
                    if match:
                        review_data['ReviewCount'] = match.group(1)
            
            # Star rating, using updated selectors
            rating_element = review_element.find('div', {'data-service-review-rating': True})
            if rating_element:
                rating_value = rating_element.get('data-service-review-rating')
                if rating_value:
                    review_data['ReviewRating'] = rating_value
                else:
                    # Fallback to star rating image
                    star_img = rating_element.find('img')
                    if star_img:
                        review_data['ReviewRating'] = extract_star_rating(star_img)
            
            # Review title
            title_element = review_element.find('h2', {'data-service-review-title-typography': 'true'})
            if title_element:
                review_data['ReviewTitle'] = clean_text(title_element.get_text())
            
            # Review text
            text_element = review_element.find('p', {'data-service-review-text-typography': 'true'})
            if text_element:
                review_data['ReviewText'] = clean_text(text_element.get_text())
            
            # Review date
            date_element = review_element.find('time', {'data-service-review-date-time-ago': 'true'})
            if date_element:
                # Try datetime attribute first
                date_str = date_element.get('datetime')
                if not date_str:
                    # Fallback to text content
                    date_str = date_element.get_text()
                if date_str:
                    review_data['ReviewDate'] = parse_date(date_str)
            
            # Experience date
            exp_date_element = review_element.find('span', {'data-service-review-date-of-experience-typography': 'true'})
            if exp_date_element:
                exp_date_str = exp_date_element.get_text()
                review_data['ReviewExperienceDate'] = parse_date(exp_date_str)
            
            # Company reply, extract only the actual reply text, not the header
            reply_text_elem = review_element.find('p', {'data-service-review-business-reply-text-typography': 'true'})
            if reply_text_elem:
                review_data['ReplyText'] = clean_text(reply_text_elem.get_text())
                
                # Find the reply date from the header section
                reply_header = review_element.find('div', class_=re.compile(r'styles_replyHeader'))
                if reply_header:
                    reply_date_elem = reply_header.find('time', {'data-service-review-business-reply-date-time-ago': 'true'})
                    if reply_date_elem:
                        reply_date_str = reply_date_elem.get('datetime')
                        if reply_date_str:
                            review_data['ReplyDate'] = parse_date(reply_date_str)
            
        except Exception as e:
            logger.error(f"Error extracting review data: {e}")
        
        return review_data
    
    def scrape_reviews(self) -> List[Dict]:
        """Main scraping method with review count limit."""
        logger.info(f"Starting to scrape {self.platform} reviews...")
        logger.info(f"Target: {MAX_REVIEWS} reviews (max {MAX_PAGES} pages)")
        
        current_url = self.base_url
        page_count = 0

        while current_url and (MAX_PAGES is None   or page_count < MAX_PAGES) and (MAX_REVIEWS is None or len(self.reviews_data) < MAX_REVIEWS):
            page_count += 1
            logger.info(f"Scraping page {page_count}: {current_url}")
            
            # Get page content
            soup = self.get_page(current_url)
            if not soup:
                logger.error(f"Failed to get page {page_count}")
                break
            
            # Find review elements using updated selectors
            review_elements = soup.find_all('article', class_=re.compile(r'styles_reviewCard|paper_paper'))
            
            # Alternative selector if first doesn't work
            if not review_elements:
                review_elements = soup.find_all('div', {'data-service-review-card-paper': 'true'})
            
            if not review_elements:
                logger.warning(f"No reviews found on page {page_count}")
                logger.debug(f"Page structure: {soup.find_all('article')[:3]}")
                break
            
            # Extract data from each review, but check limit
            page_reviews = []
            for review_element in review_elements:
                if MAX_REVIEWS is not None and len(self.reviews_data) + len(page_reviews) >= MAX_REVIEWS:
                    logger.info(f"Reached maximum review limit ({MAX_REVIEWS})")
                    break
                    
                review_data = self.extract_review_data(review_element)
                if review_data['ReviewText'] or review_data['ReviewTitle']:  
                    page_reviews.append(review_data)
            
            self.reviews_data.extend(page_reviews)
            
            logger.info(f"Extracted {len(page_reviews)} reviews from page {page_count}")
            
            if MAX_REVIEWS is None:
                logger.info(f"Total reviews collected: {len(self.reviews_data)}")
            else:
                logger.info(f"Total reviews collected: {len(self.reviews_data)}/{MAX_REVIEWS}")
            
            # Progress update
            if len(self.reviews_data) % PROGRESS_INTERVAL == 0:
                logger.info(f"Progress: {len(self.reviews_data)} reviews scraped")
            
            # Check if we've reached the limit
            if MAX_REVIEWS is not None and len(self.reviews_data) >= MAX_REVIEWS:
                logger.info(f"Reached target of {MAX_REVIEWS} reviews")
                break
            
            # Find next page link
            next_link = soup.find('a', {'name': 'pagination-button-next'})
            if not next_link:
                # Alternative selectors for next page
                next_link = soup.find('a', class_=re.compile(r'next'))
                if not next_link:
                    next_link = soup.find('a', {'aria-label': 'Next page'})
            
            if next_link and next_link.get('href'):
                current_url = urljoin(self.base_url, next_link['href'])
            else:
                logger.info("No more pages found")
                break
            
            # Respectful delay between requests
            time.sleep(DELAY_BETWEEN_REQUESTS)
        
        logger.info(f"Scraping completed. Total reviews: {len(self.reviews_data)}")
        return self.reviews_data
    
    def save_to_csv(self, filename: str = None):
        """Save scraped data to CSV file."""
        if not self.reviews_data:
            logger.warning("No data to save")
            return
        
        filename = filename or OUTPUT_FILE
        
        # Create DataFrame
        df = pd.DataFrame(self.reviews_data)
        
        # Data type conversions
        df['ReviewCount'] = pd.to_numeric(df['ReviewCount'], errors='coerce')
        df['ReviewRating'] = pd.to_numeric(df['ReviewRating'], errors='coerce')
        
        # Save to CSV
        df.to_csv(filename, index=False, encoding='utf-8')
        logger.info(f"Data saved to {filename}")
        
        # Print summary statistics
        logger.info(f"Summary Statistics:")
        logger.info(f"- Total reviews: {len(df)}")
        if df['ReviewRating'].notna().any():
            logger.info(f"- Average rating: {df['ReviewRating'].mean():.2f}")
            logger.info(f"- Rating distribution:\n{df['ReviewRating'].value_counts().sort_index()}")
        if df['ReviewDate'].notna().any():
            logger.info(f"- Date range: {df['ReviewDate'].min()} to {df['ReviewDate'].max()}")
        
        # Show sample of replies
        reply_count = df['ReplyText'].notna().sum()
        logger.info(f"- Reviews with company replies: {reply_count}")
        if reply_count > 0:
            logger.info(f"- Sample reply preview: {df[df['ReplyText'].notna()]['ReplyText'].iloc[0][:100]}...")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("TRUSTPILOT REVIEW SCRAPER")
    logger.info("="*60)
    logger.info(f"Target Platform: {TARGET_PLATFORM}")
    logger.info(f"Max Reviews: {MAX_REVIEWS}")
    logger.info(f"Max Pages: {MAX_PAGES}")
    logger.info(f"Output File: {OUTPUT_FILE}")
    logger.info("="*60)
    
    # Initialize scraper
    scraper = TrustpilotScraper(TARGET_PLATFORM)
    
    # Check if platform URL exists
    if not scraper.base_url:
        logger.error(f"Platform '{TARGET_PLATFORM}' not found in PLATFORM_URLS")
        return
    
    try:
        # Scrape reviews
        reviews = scraper.scrape_reviews()
        
        # Save to CSV
        scraper.save_to_csv()
        
        logger.info("Scraping process completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
        if scraper.reviews_data:
            logger.info("Saving partial data...")
            scraper.save_to_csv(f"partial_{OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if scraper.reviews_data:
            logger.info("Saving partial data...")
            scraper.save_to_csv(f"error_{OUTPUT_FILE}")


if __name__ == "__main__":
    main()