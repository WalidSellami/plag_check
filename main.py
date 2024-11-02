

# -- install the following libraries -- #
# pip install fastapi
# pip install uvicorn
# pip install spacy-udpipe
# pip install selenium 

# -- In scikit-leran install last version 1.4.1.post1 to avoid any error -- #
# pip install --upgrade scikit-learn (if you have an old version just upgrade it)
# pip install scikit-learn (if you don't have it automatically it will install the last version) 


from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
# from fastapi.templating import Jinja2Templates  # --> For rendering HTML templates
import re
import string
import pandas as pd
import os
# import nltk
from nltk.corpus import stopwords
import spacy_udpipe
import pickle

# nltk.download('stopwords')  # --> Download it only one time

# Scikit-learn for TF-IDF vectorizer and cosine similarity (Handling the sources)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Initialize the FastAPI app
app = FastAPI()

# templates = Jinja2Templates(directory="templates") # --> For rendering HTML templates


# spacy_udpipe.download("ar") # Download it only one time
nlp = spacy_udpipe.load("ar")
additional_punctuation = ['؛', '-', '؟', '!', '#', '،', "'ٌ؛", "؛'ٌ", "-،", "،-", "_", "ــ", "—", "|", '[', ']', '{', '@',
                          '}', '<', '>', '(', ')', '«', '»', 'ـ', "'ً", 'ٌ', "'ٍ", "'َ", "'ُ", "'ِ", "'ّ", "'ْ", "'ٓ", 'ٔ', "'ٰ",]

pattern = "[" + re.escape("".join(additional_punctuation)) + "]"
stop_words = stopwords.words('arabic')

lexical_df = pd.read_csv('lexical_database.csv')

# Load the model and vectorizer
with open('gb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


# Load the original docs
original_docs = []

data_path = 'E:\\All_Projects\\ML_Projects\\plag_check\\original-docs' # --> your project path

for filename in os.listdir(data_path):
   file_path = os.path.join(data_path, filename)
   with open(file_path, 'r', encoding='utf-8') as f:
      text = f.read().strip()
      original_docs.append(text)


# Preprocess the text
def preprocess(text):
    # lemma_tokens = []

    # Remove inside {} and () that come after it (for quran)
    text = re.sub(r'\{.*?\}\s*\((.*?)\)', '', text)
    text = re.sub(r'\﴾.*?\﴿\s*', '', text)
    # Remove inside {} and () that come before it (for quran)
    text = re.sub(r'\((.*?)\)\s*\{.*?\}', '', text)

    # Remove notations from the word
    text = re.sub(pattern, ' ', text)

    # Remove HTML code
    text = re.sub('<[^<]+?>', '', text)

    # Remove URLs (http and https)
    text = re.sub(r'https?://\S+', '', text)

    # Remove English or French words
    text = re.sub(r'\b[a-zA-Z]+\b', '', text)

    # Remove extar white space
    text = re.sub(r'\s+', ' ', text)

    doc = nlp(text)
    
    cleaned_text = ''

    for sentence in doc.sents:
        tokens = [word.text for word in sentence]
        normalized_tokens = [token for token in tokens if token not in string.punctuation and token not in additional_punctuation]
        cleaned_tokens = [token for token in normalized_tokens if token not in stop_words]
        cleaned_text = ' '.join(cleaned_tokens)
        # for sentence in sentences:
        #   for word in sentence:
        #      found = False
        #      for index, row in lexical_df.iterrows():
        #          if word.text == row['word']:
        #              lemma_tokens.append(row['lemma'])
        #              found = True
        #              break
        #      if not found:
        #          lemma_tokens.append(word.lemma_)

    # new_text = ' '.join(lemma_tokens)

    # Remove < and >
    new_text = re.sub(r'<|>', '', cleaned_text)
    # Remove English or French words (another check)
    new_text = re.sub(r'\b[a-zA-Z]+\b', '', new_text)
    # Remove Greek characters
    new_text = re.sub(r'[\u0370-\u03FF\u1F00-\u1FFF]+', '', new_text)
    # Remove variations of UNK
    new_text = re.sub(r'\s*[U\s]*NK\s*', '', new_text)
    # Remove extra whitespace, including double tabs
    new_text = re.sub(r'\s+', ' ', new_text)

    return new_text




def get_scores(upload_document):
     
   pre_document = preprocess(upload_document)
   doc_vec = vectorizer.transform([pre_document])
#    prediction = model.predict(doc_vec) # -->  0: Original, 1: Plagiarized (class that have higher probability)

   plagiarism_score = round(model.predict_proba(doc_vec)[0][1], 2)
   original_score = round(model.predict_proba(doc_vec)[0][0], 2)

  
   if plagiarism_score < 0.01:  # 1% threshold
       plagiarism_score = 0.0  # 0%
       original_score = 1.0  # 100%
       

   return {
        'plagiarism_score': plagiarism_score,
        'original_score': original_score,
    }



@app.post('/report')
async def get_report(data: dict):
    
    try:
            
        uploaded_document = data.get('uploaded_document') 
        chosen_ratio = data.get('chosen_ratio')
        if not uploaded_document:
            raise HTTPException(status_code=400, detail='No document uploaded')
    
        plagiarism_score = get_scores(uploaded_document)['plagiarism_score']
        original_score = get_scores(uploaded_document)['original_score']
            
        texts = extract_plagiarized_text(uploaded_document, original_docs)
        
        nbr = 0
        max_sents = 20
        doc_type = ''
        sources = []
    
        
        if len(texts) != 0:
            
           if chosen_ratio != 0 and chosen_ratio < plagiarism_score:
             nbr = round(chosen_ratio * max_sents)
           else:  
             nbr = round(plagiarism_score * max_sents)  
        
           if plagiarism_score <= 0.15 or chosen_ratio <= 0.10:   
              nbr += 1
             
           texts = texts[:nbr]  
           sources = extract_sources(texts)
           
        
        if not texts: 
        
             doc_type = 'Original Document'
             original_score = 1.0
             plagiarism_score = 0.0
             chosen_ratio = 0
             sources = []
                
        else:
             doc_type = 'Plagiarized Document'   
           
           
        content = {
            'status': 'success',
            'document_type': doc_type,
            'plagiarism_score': plagiarism_score,
            'original_score': original_score,
            'plagiarized_texts': texts,
            'sources': sources}
        
        
        return JSONResponse(content=content, status_code=200)
    
    except Exception as e:
        
        content = {
            'status': 'failed',
            'message_error': str(e)}
        
        return JSONResponse(content, status_code=400) 
    
    
# Extract plagiarized text
def extract_plagiarized_text(upload_doc, original_docs):
    vectorizer = TfidfVectorizer()
    # Transform the original documents and the uploaded document
    tfidf_matrix = vectorizer.fit_transform([upload_doc] + original_docs)

    # Calculate similarities
    similarity_scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])

    # Find plagiarized parts using n-gram comparison
    threshold = 0.75  # 85% minimum similarity
    n_gram_range = (4, 26) 
    plagiarized_parts = set()

    for i, score in enumerate(similarity_scores[0]):
        if score >= threshold:
            original_text = original_docs[i]
            text_tokens = upload_doc.split()
            original_text_tokens = original_text.split()

            # Create n-grams
            text_ngrams = set(zip(*[text_tokens[i:] for i in range(n_gram_range[1])]))
            original_text_ngrams = set(zip(*[original_text_tokens[i:] for i in range(n_gram_range[1])]))

            # Find matching n-grams
            matching_ngrams = text_ngrams.intersection(original_text_ngrams)

            # Reconstruct potential plagiarized phrases
            for ngram in matching_ngrams:
                plagiarized_phrase = " ".join(ngram)
                plagiarized_parts.add(plagiarized_phrase)

    return list(plagiarized_parts)



# Get the sources of the plagiarized text
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from time import sleep
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
      sleep(random.uniform(1, 3))  # Random sleep time between 1 and 3 seconds
  
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




@app.get('/')
async def welcome():
    return {
        'message': 'Welcome to the Plagiarism Checker APP (PlagCheck)'
    }


# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, port=8000)


# Run in the terminal :

# uvicorn main:app --reload   --> for local server (http://127.0.0.1:8000)

# uvicorn main:app --host=0.0.0.0 --reload   --> for public access (http://your_ip_address:8000)