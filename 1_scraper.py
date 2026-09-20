import requests
from bs4 import BeautifulSoup
import json
import os

BASE_URL = "https://bn.wikisource.org/wiki/পথের_দাবী_(শরৎচন্দ্র_চট্টোপাধ্যায়,_১৯৫৮)/"

def to_bengali_numeral(n):
    # Converts standard digits to Bengali digits (e.g., 12 -> ১২)
    eng_to_bn = str.maketrans('0123456789', '০১২৩৪৫৬৭৮৯')
    return str(n).translate(eng_to_bn)

def crawl_book():
    os.makedirs("data", exist_ok=True)
    book_data = []
    
    chapter_num = 1
    print("Starting crawler...")
    
    while True:
        bn_chapter = to_bengali_numeral(chapter_num)
        url = BASE_URL + bn_chapter
        
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check if the page is empty or missing (Wikisource returns a 'noarticletext' div)
        noarticletext = soup.find('div', class_='noarticletext')
        if noarticletext or response.status_code == 404:
            print(f"Finished crawling. Total chapters found: {chapter_num - 1}")
            break
            
        # Extract main text body
        content_div = soup.find('div', class_='mw-parser-output')
        paragraphs = content_div.find_all('p') if content_div else []
        
        chapter_text = "\n".join([p.get_text(strip=True) for p in paragraphs])
        
        # Save if content exists
        if chapter_text.strip():
            book_data.append({
                "book_name": "পথের দাবী (Pather Dabi)",
                "chapter_name": bn_chapter,
                "url": url,
                "content": chapter_text
            })
            print(f"Scraped Chapter: {bn_chapter}")
        
        chapter_num += 1
        
        # Failsafe loop breaker
        if chapter_num > 40:
            break

    with open("data/pather_dabi.json", "w", encoding="utf-8") as f:
        json.dump(book_data, f, ensure_ascii=False, indent=4)
    print("Data successfully saved to data/pather_dabi.json")

if __name__ == "__main__":
    crawl_book()