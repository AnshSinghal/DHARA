
#!/usr/bin/env python3
"""
Legal Web Scraper - Scrapes legal case links from Indian Kanoon
"""

import time
import traceback
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from logger import DharaLogger

# Initialize dharalogger for web scraping
dharalogger = DharaLogger("web_scraper", log_file="web_scrapping.log")
logger = dharalogger.get_logger()

def setup_driver():
    """Setup Chrome driver with options"""
    logger.info("Setting up Chrome driver...")
    
    try:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        
        logger.debug("Chrome options configured:")
        for arg in options.arguments:
            logger.debug(f"  {arg}")
        
        driver = webdriver.Chrome(options=options)
        logger.info("✅ Chrome driver initialized successfully")
        return driver
        
    except Exception as e:
        logger.error(f"Failed to setup Chrome driver: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

def scrape_page_links(driver, url, page_num):
    """Scrape links from a single page"""
    logger.info(f"Scraping page {page_num}: {url}")
    
    start_time = time.time()
    
    try:
        # Navigate to page
        logger.debug(f"Navigating to URL: {url}")
        driver.get(url)
        
        # Wait for results to load
        logger.debug("Waiting for page results to load...")
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'result')))
        
        # Get page source and parse
        logger.debug("Parsing page content...")
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # Find result divs
        result_divs = soup.find_all('div', class_='result')
        logger.debug(f"Found {len(result_divs)} result divs")
        
        # Extract links
        page_links = []
        for i, div in enumerate(result_divs):
            link_element = div.find('a', href=True)
            if link_element:
                link = link_element['href']
                page_links.append(link)
                logger.debug(f"  Link {i+1}: {link}")
            else:
                logger.warning(f"  No link found in result div {i+1}")
        
        scrape_time = time.time() - start_time
        logger.info(f"✅ Page {page_num} scraped successfully:")
        logger.info(f"   Time: {scrape_time:.2f}s")
        logger.info(f"   Links found: {len(page_links)}")
        
        return page_links
        
    except TimeoutException:
        scrape_time = time.time() - start_time
        logger.error(f"❌ Timeout waiting for page {page_num} to load after {scrape_time:.2f}s")
        return []
        
    except Exception as e:
        scrape_time = time.time() - start_time
        logger.error(f"❌ Error scraping page {page_num} after {scrape_time:.2f}s: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return []

def main():
    """Main scraping function"""
    logger.info("=" * 80)
    logger.info("LEGAL WEB SCRAPER STARTING")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Configuration
    base_url = "https://indiankanoon.org/search/?formInput=commercial%20court%20%20%20doctypes%3A%20judgments"
    max_pages = 50
    output_file = "all_links.txt"
    
    logger.info(f"Configuration:")
    logger.info(f"  Base URL: {base_url}")
    logger.info(f"  Max pages: {max_pages}")
    logger.info(f"  Output file: {output_file}")
    
    driver = None
    
    try:
        # Setup driver
        driver = setup_driver()
        
        # Open output file
        logger.info(f"Opening output file: {output_file}")
        with open(output_file, "w", encoding='utf-8') as f:
            
            all_links = []
            successful_pages = 0
            failed_pages = 0
            
            # Scrape each page
            for page_num in range(0, max_pages):
                logger.info(f"Processing page {page_num + 1}/{max_pages}")
                
                # Construct URL
                if page_num == 0:
                    url = base_url
                else:
                    url = base_url + "&pagenum=" + str(page_num)
                
                # Scrape page
                page_links = scrape_page_links(driver, url, page_num + 1)
                
                if page_links:
                    all_links.extend(page_links)
                    successful_pages += 1
                    logger.info(f"Cumulative links collected: {len(all_links)}")
                else:
                    failed_pages += 1
                    logger.warning(f"No links collected from page {page_num + 1}")
                
                # Add delay between requests to be respectful
                if page_num < max_pages - 1:  # Don't sleep after last page
                    logger.debug("Sleeping 2 seconds between requests...")
                    time.sleep(2)
            
            # Process and save links
            logger.info("Processing and saving collected links...")
            
            processed_links = []
            for i, link in enumerate(all_links):
                try:
                    link_parts = link.split("/")
                    if len(link_parts) >= 3:
                        processed_link = f"https://indiankanoon.org/doc/{link_parts[2]}"
                        processed_links.append(processed_link)
                        f.write(processed_link + "\n")
                        logger.debug(f"  Processed link {i+1}: {processed_link}")
                    else:
                        logger.warning(f"  Invalid link format: {link}")
                except Exception as e:
                    logger.error(f"  Error processing link {i+1} '{link}': {e}")
            
            total_time = time.time() - start_time
            
            # Final statistics
            logger.info("=" * 80)
            logger.info("WEB SCRAPING COMPLETED SUCCESSFULLY")
            logger.info(f"Total processing time: {total_time:.2f}s")
            logger.info(f"Statistics:")
            logger.info(f"  Pages attempted: {max_pages}")
            logger.info(f"  Successful pages: {successful_pages}")
            logger.info(f"  Failed pages: {failed_pages}")
            logger.info(f"  Success rate: {successful_pages/max_pages*100:.1f}%")
            logger.info(f"  Raw links collected: {len(all_links)}")
            logger.info(f"  Processed links saved: {len(processed_links)}")
            logger.info(f"  Output file: {output_file}")
            logger.info(f"  Average time per page: {total_time/max_pages:.2f}s")
            logger.info("=" * 80)
            
    except Exception as e:
        error_time = time.time() - start_time
        logger.error("=" * 80)
        logger.error("WEB SCRAPING FAILED")
        logger.error(f"Error after {error_time:.2f}s: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        logger.error("=" * 80)
        raise
        
    finally:
        # Clean up driver
        if driver:
            try:
                logger.info("Closing web driver...")
                driver.quit()
                logger.info("✅ Web driver closed successfully")
            except Exception as e:
                logger.error(f"Error closing driver: {e}")

if __name__ == "__main__":
    try:
        main()
        print("✅ Web scraping completed successfully!")
        print("Check 'all_links.txt' for the scraped links.")
        
    except Exception as e:
        print(f"❌ Web scraping failed: {e}")
        exit(1)
