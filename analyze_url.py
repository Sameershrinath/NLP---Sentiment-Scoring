import os
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import re
import pandas as pd

#Download necessary NLTK data
nltk.download('punkt')


#load stop words
def load_stop_words(folder_path):
    stop_words = set()
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            with open(os.path.join(folder_path, file_name), 'r', encoding='ISO-8859-1') as file:
                stop_words.update(file.read().split())
    return stop_words



# loading Given Files 
stop_words = load_stop_words('StopWords-20241020T074411Z-001/StopWords')

with open(r'MasterDictionary-20241020T074411Z-001\MasterDictionary\positive-words.txt', 'r', encoding='ISO-8859-1') as f:
    positive_words = set(f.read().split())
with open(r'MasterDictionary-20241020T074411Z-001\MasterDictionary\negative-words.txt', 'r', encoding='ISO-8859-1') as f:
    negative_words = set(f.read().split())
input_df = pd.read_excel('Input.xlsx')


# Functions based on given Text analysis task
def clean_text(text):
    """Clean and tokenize text by removing stopwords and punctuation."""
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return tokens


def get_sentiment_scores(tokens):
    """Calculate positive, negative, polarity, and subjectivity scores."""
    positive_score = sum(1 for word in tokens if word in positive_words)
    negative_score = sum(1 for word in tokens if word in negative_words)
    polarity_score = (positive_score - negative_score) / (positive_score + negative_score + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(tokens) + 0.000001)
    return positive_score, negative_score, polarity_score, subjectivity_score

def analyze_readability(text):
    """Calculate readability metrics like average sentence length, fog index, etc."""
    sentences = nltk.sent_tokenize(text)
    words = clean_text(text)
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    complex_words = [word for word in words if len([char for char in word if char in 'aeiou']) > 2]
    percentage_complex_words = len(complex_words) / len(words) if words else 0
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    avg_no_of_words_per_sentence=len(words)/len(sentences) if words and sentences else 0
    return avg_sentence_length, percentage_complex_words, fog_index,avg_no_of_words_per_sentence

def count_personal_pronouns(text):
    """Count personal pronouns in text."""
    pronouns = re.findall(r'\b(I|we|my|ours|us)\b', text, re.I)
    return len(pronouns)

def avg_word_length(words):
    """Calculate average word length."""
    return sum(len(word) for word in words) / len(words) if words else 0

# Scrape article text from URL
def extract_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    article = soup.find('article')
    if article:
        title = article.find('h1').get_text()
        paragraphs = article.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return title, text
    return None, None


# Initialize an empty dataframe before the loop
output_df = pd.DataFrame(columns=['URL_ID','URL', 'Positive_Score', 'Negative_Score', 'Polarity_Score', 'Subjectivity_Score',
                                  'Avg_Sentence_Length', 'Percentage_Complex_Words', 'Fog_Index','AVG NUMBER OF WORDS PER SENTENCE',
                                    'Complex_Word_Count','Word_Count', 'Syllable_Per_Word', 'Personal_Pronouns', 'Avg_Word_Length'])





# Folder where the extracted text files will be saved
output_folder = 'extracted_articles'

# Check if the folder exists, if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Processing...Please wait..")

# Process each URL in the Input file in a loop
for index, row in input_df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    title, text = extract_text(url)
    
    if title and text:
        file_path = os.path.join(output_folder, f'{url_id}.txt')

        # Save the extracted article in the text file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(f"Title: {title}\n\n{text}")
        
        # Perform text analysis (same as before)
        tokens = clean_text(text)
        pos_score, neg_score, polarity, subjectivity = get_sentiment_scores(tokens)
        avg_sent_len, perc_complex, fog_idx, avg_no_of_words_in_sentences = analyze_readability(text)
        complex_word_count = len([word for word in tokens if len([char for char in word if char in 'aeiou']) > 2])
        word_count = len(tokens)
        syll_per_word = sum([len([char for char in word if char in 'aeiou']) for word in tokens]) / word_count if word_count > 0 else 0
        personal_pronoun_count = count_personal_pronouns(text)
        avg_word_len = avg_word_length(tokens)

        # Create a DataFrame for the new row
        new_row = pd.DataFrame([{
            'URL_ID': url_id,
            'URL': url,
            'Positive_Score': pos_score,
            'Negative_Score': neg_score,
            'Polarity_Score': polarity,
            'Subjectivity_Score': subjectivity,
            'Avg_Sentence_Length': avg_sent_len,
            'Percentage_Complex_Words': perc_complex,
            'Fog_Index': fog_idx,
            'AVG NUMBER OF WORDS PER SENTENCE': avg_no_of_words_in_sentences,
            'Complex_Word_Count': complex_word_count,
            'Word_Count': word_count,
            'Syllable_Per_Word': syll_per_word,
            'Personal_Pronouns': personal_pronoun_count,
            'Avg_Word_Length': avg_word_len
        }])

        # Concatenate with the previous made output dataframe
        output_df = pd.concat([output_df, new_row], ignore_index=True)

# Save the output to a CSV file
output_df.to_csv('Output.csv', index=False)
print("Task Completed...")


