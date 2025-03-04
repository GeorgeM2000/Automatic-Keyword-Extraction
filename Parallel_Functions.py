import concurrent.futures
import pickle
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from RAKE import *


def process_tfidf_batch(batch_abstracts, vectorizer_pickle, feature_names, T, batch_index):
    """
    Process a batch of abstracts and extract the top keywords.
    """
    with open(vectorizer_pickle, 'rb') as f:
        vectorizer = pickle.load(f)  # Load shared vectorizer

    tfidf_matrix = vectorizer.transform(batch_abstracts)  # Use transform
    batch_keywords = []

    # Process each row in the batch
    for row in tfidf_matrix:
        row_array = row.toarray().flatten()
        top_keywords = [feature_names[j] for j in row_array.argsort()[-T:][::-1]]
        batch_keywords.append(top_keywords)
    
    return batch_index, batch_keywords

def parallel_tfidf_processing(abstracts, ranges, T, num_threads=5):
    keywords_per_abstract = [None] * len(abstracts)
    vectorizer_pickle = "vectorizer.pkl"

    start_time = datetime.now()

    # Fit once on the entire data to get document frequencies (IDF values)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(abstracts)  # Only fit to get IDF and vocabulary

    # Get feature names (i.e., words)
    feature_names = vectorizer.get_feature_names_out()

    # Save vectorizer for multiprocessing
    with open(vectorizer_pickle, 'wb') as f:
        pickle.dump(vectorizer, f)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks to the thread pool for each range of abstracts
        futures = [
            executor.submit(process_tfidf_batch, abstracts[start:end], vectorizer_pickle, feature_names, T, batch_index)
            for batch_index, (start, end) in enumerate(ranges)
        ]
        
        # Wait for the results and place them in the correct position in the final list
        for future in concurrent.futures.as_completed(futures):
            batch_index, batch_keywords = future.result()
            start, end = ranges[batch_index]
            keywords_per_abstract[start:end] = batch_keywords  # Place results in the correct range

        end_time = datetime.now()
        print(f'Time TF-IDF {T}: {((end_time - start_time).total_seconds()) / 60.0}')

    return keywords_per_abstract


def process_batch_rake(batch_abstracts, T, batch_index, kextraction_method):
    batch_keywords = []
    
    for abstract in batch_abstracts:        
        oR = Rake(ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO)
        oR.extract_keywords_from_text(abstract)
        unique_ranked_keywords = sorted(set(oR.get_ranked_phrases_with_scores()), key=lambda x: x[0], reverse=True)[:T]
        batch_keywords.append([keyphrase for _, keyphrase in unique_ranked_keywords])

    return batch_index, batch_keywords

def process_batch_yake(batch_abstracts, T, batch_index, kextraction_method):
    batch_keywords = []
    
    for abstract in batch_abstracts:        
        yake_res = kextraction_method.extract_keywords(abstract)
        batch_keywords.append([keyphrase for keyphrase, _ in yake_res])

    return batch_index, batch_keywords

def process_batch_rank(batch_abstracts, T, batch_index, kextraction_method):
    batch_keywords = []
    
    for abstract in batch_abstracts:        
        rank_res = kextraction_method(abstract)
        batch_keywords.append([keyphrase.text for keyphrase in rank_res._.phrases[:T]])

    return batch_index, batch_keywords

# Parallel processing function
def parallel_kextraction_processing(abstracts, ranges, T, kextraction_method, num_threads=15, kextraction_method_obj=None):
    keywords = [None] * len(abstracts)  # Initialize the final list
    kextraction_function = None

    if kextraction_method == "RAKE":
        kextraction_function = process_batch_rake
    elif kextraction_method == "YAKE":
        kextraction_function = process_batch_yake
    else:
        kextraction_function = process_batch_rank


    start_time = datetime.now()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(kextraction_function, abstracts[start:end], T, batch_index, kextraction_method_obj)
            for batch_index, (start, end) in enumerate(ranges)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            batch_index, batch_keywords = future.result()
            start, end = ranges[batch_index]
            keywords[start:end] = batch_keywords  # Place results in the correct range

        end_time = datetime.now()
        print(f'Time {kextraction_method} {T}: {((end_time - start_time).total_seconds()) / 60.0}')

    return keywords