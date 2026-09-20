import requests
from bs4 import BeautifulSoup
import json
import time

def scrape_pather_dabi():
    bengali_numerals = ["১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯", "১০", 
                        "১১", "১২", "১৩", "১৪", "১৫", "১৬", "১৭", "১৮", "১৯", "২০", 
                        "২১", "২২", "২৩", "২৪", "২৫", "২৬", "২৭", "২৮", "২৯", "৩০", "৩১"]
    
    base_url = "https://bn.wikisource.org/wiki/পথের_দাবী_(শরৎচন্দ্র_চট্টোপাধ্যায়,_১৯৫৮)/"
    book_data = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    print("Starting book extraction from Wikisource...")
    
    for chapter in bengali_numerals:
        url = base_url + chapter
        print(f"Scraping Chapter: {chapter}...")
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # CRITICAL FIX: Wikisource keeps the raw book text inside this specific div, not in <p> tags
            text_container = soup.find('div', class_='prp-pages-output')
            
            # Fallback just in case
            if not text_container:
                text_container = soup.find('div', class_='mw-parser-output')
            
            if text_container:
                # Extract all raw text and separate lines with a newline
                chapter_text = text_container.get_text(separator='\n', strip=True)
                
                if chapter_text and len(chapter_text) > 100:
                    book_data.append({
                        "chapter": chapter,
                        "text": chapter_text,
                        "url": url
                    })
                    print(f" -> Success! ({len(chapter_text)} characters extracted)")
                else:
                    print(" -> Failed: No text found on page.")
            
            time.sleep(1)
        else:
            print(f" -> Failed to retrieve chapter {chapter} (Status Code: {response.status_code})")

    with open("pather_dabi.json", "w", encoding="utf-8") as f:
        json.dump(book_data, f, ensure_ascii=False, indent=4)
        
    print(f"\nSuccessfully scraped {len(book_data)} chapters and saved to pather_dabi.json")

if __name__ == "__main__":
    scrape_pather_dabi()