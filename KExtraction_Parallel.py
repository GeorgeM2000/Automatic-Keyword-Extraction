import nltk; nltk.data.path.append("/home/pk/Context_Aware_Node_Embeddings/Downloads")
import string
import spacy
import yake
import pytextrank
import gc
import random
import time
random.seed(42)

from nltk.corpus import stopwords
from Parallel_Functions import *


def save_keywords_to_files(keywords, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for k in keywords:
            line = ' '.join(k)
            f.write(line + '\n')


def preprocess(text):
  text = text.lower()  # Lowercase
  text = ''.join([ch for ch in text if ch not in punctuation])  # Remove punctuation
  tokens = text.split()
  tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
  return ' '.join(tokens)


def extract_abstracts(file_path):
    # Read the contents of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        abstracts = file.readlines()

   # Remove any leading or trailing whitespace characters from each line
    abstracts = [abstract.strip() for abstract in abstracts if abstract.strip()]

    # Track the number of abstracts
    num_abstracts = len(abstracts)

    return abstracts, num_abstracts


def process_keywords(keywords, output_file):
    word_counts = []  # To store word count per sublist
    total_word_count = 0
    total_non_word_count = 0

    for sublist in keywords:
        sublist_word_count = 0  # Count words in this sublist
        sublist_non_word_count = 0

        for keyphrase in sublist:
            # Split on whitespace or dash but preserve valid words
            words = [word for part in keyphrase.split('-') for word in part.split()]
            
            for word in words:
                if word.isalpha():  # Check if it's a valid word
                    sublist_word_count += 1
                    total_word_count += 1
                else:
                    sublist_non_word_count += 1
                    total_non_word_count += 1

        word_counts.append(sublist_word_count)
    
    # Save word counts to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for count in word_counts:
            f.write(str(count) + '\n')

    return total_word_count, total_non_word_count

if __name__ == '__main__':
    
    parent_path = 'Datasets/arxiv/graph-v2'
    abstracts, num_abstracts = extract_abstracts(f'{parent_path}/data-v2.txt')

    T = [5, 10]

    
    batch_size_15 = int(num_abstracts / 15)  # Small batch size for other extraction methods
    #batch_size_5 = int(num_abstracts / 5)    # Larger batch size for TF-IDF

    ranges_15 = []
    #ranges_5 = []

    # Generate 15-batch ranges
    start = 0
    while start < num_abstracts:
        end = min(start + batch_size_15, num_abstracts)
        ranges_15.append([start, end])
        start = end

    # Merge last two batches to prevent small last batch
    ranges_15[-2][1] = ranges_15[-1][1]
    del ranges_15[-1]

    # Generate 5-batch ranges
    #start = 0
    #while start < num_abstracts:
    #    end = min(start + batch_size_5, num_abstracts)
    #    ranges_5.append([start, end])
    #    start = end

    # Merge last two batches to prevent small last batch
    #ranges_5[-2][1] = ranges_5[-1][1]
    #del ranges_5[-1]

    print(f'15 ranges: {ranges_15}\n')
    #print(f'5 ranges for TF-IDF: {ranges_5}')


    # Preprocess the abstracts: remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)

    # Preprocess each abstract
    preprocessed_abstracts = [preprocess(abstract) for abstract in abstracts]


    keywords = {
        'RAKE5': None,
        'YAKE5': None,
        #'TFIDF5': None,
        'TextR5': None,
        'PositionR5': None,
        'TopicR5': None,

        'RAKE10': None,
        'YAKE10': None,
        #'TFIDF10': None,
        'TextR10': None,
        'PositionR10': None,
        'TopicR10': None
    }

    """ for t in T:
        keywords[f'TFIDF{t}'] = parallel_tfidf_processing(preprocessed_abstracts, ranges_5, t)
        
	time.sleep(20) """ 

    for t in T:
        keywords[f'RAKE{t}'] = parallel_kextraction_processing(abstracts, ranges_15, t, 'RAKE', kextraction_method_obj=None)
        
    time.sleep(20)

    for t in T:
        yake_custom_keyword_extractor = yake.KeywordExtractor(lan='en', n=3, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=t, features=None)
        keywords[f'YAKE{t}'] = parallel_kextraction_processing(abstracts, ranges_15, t, 'YAKE', kextraction_method_obj=yake_custom_keyword_extractor)
        del yake_custom_keyword_extractor
        
    gc.collect()

    time.sleep(20)

        
    topicrank = spacy.load("en_core_web_lg-3.8.0-py3-none-any/en_core_web_lg/en_core_web_lg-3.8.0")
    topicrank.add_pipe("topicrank")
    for t in T:
        keywords[f'TopicR{t}'] = parallel_kextraction_processing(abstracts, ranges_15, t, 'Topic Rank', kextraction_method_obj=topicrank)
    
    del topicrank
    gc.collect()

    time.sleep(30)

    textrank = spacy.load("en_core_web_lg-3.8.0-py3-none-any/en_core_web_lg/en_core_web_lg-3.8.0")
    textrank.add_pipe("textrank")
    for t in T:
        keywords[f'TextR{t}'] = parallel_kextraction_processing(abstracts, ranges_15, t, 'Text Rank', kextraction_method_obj=textrank)
    
    del textrank
    gc.collect()

    time.sleep(30)

    positionrank = spacy.load("en_core_web_lg-3.8.0-py3-none-any/en_core_web_lg/en_core_web_lg-3.8.0")
    positionrank.add_pipe("positionrank")
    for t in T:
        keywords[f'PositionR{t}'] = parallel_kextraction_processing(abstracts, ranges_15, t, 'Position Rank', kextraction_method_obj=positionrank)
        
    del positionrank
    gc.collect()
    
    time.sleep(20)




    for method, list_of_keywords in keywords.items():
        save_keywords_to_files(list_of_keywords, f'{parent_path}/Keywords/Parallel_Generated/{method}.txt')
        total_word_count, total_non_word_count = process_keywords(list_of_keywords, f'{parent_path}/Word_Counts/{method}.txt')
        print(f'% of Non-Words for {method}: {total_non_word_count / total_word_count}')
