# Get the sources of the plagiarized text
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from time import sleep, time
import random

def extract_sources(texts):
  options = Options()
  options.add_argument("--headless")
  options.add_argument("--ignore-certificate-errors")
  driver = webdriver.Chrome(options=options) 
  
  sources = []
  
  for text in texts:
      inner_sources = []
      search_url = f"https://www.google.com/search?q={text}"
      driver.get(search_url)
      sleep(random.uniform(1, 3)) 
  
      html_content = driver.page_source 
      soup = BeautifulSoup(html_content, "html.parser")
  
      result_links = soup.select(".yuRUbf a") 
      urls = list(set(link["href"] for link in result_links))
      urls = [url for url in urls if 'translate.google' not in url]
      
      for url in urls[:3]:
        inner_sources.append(url)
      
      sources.append(inner_sources)
      
  
  driver.quit() 
  
  return sources

  
  
text = ["Python is a programming language that lets you work quickly", "Dart is a high-level programming language"]


sources = extract_sources(text)

print(sources) 


