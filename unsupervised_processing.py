
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import pickle

from text_processing import clean_text

def count_all_topics(dataset, total_topics, state=27):
    # Convert the documents to a matrix of word counts
    count_vectorizer = CountVectorizer(stop_words='english')
    count_matrix = count_vectorizer.fit_transform(dataset)
    
    # Apply NMF to the count matrix
    nmf_model = NMF(n_components=total_topics, random_state=state, max_iter=10000)
    nmf_model.fit(count_matrix)

    # Create a mapping from words to topics
    word_topic_mapping = {}
    for word, index in count_vectorizer.vocabulary_.items():
        word_topic_mapping[word] = np.argmax(nmf_model.components_[:, index])

    # Count the occurrences of each topic in each document
    topic_counts_per_document = np.zeros((len(dataset), total_topics))
    for i, document in enumerate(dataset):
        for word in document.split():
            if word in word_topic_mapping:
                topic = word_topic_mapping[word]
                topic_counts_per_document[i][topic] += 1

    return topic_counts_per_document

def create_dominant_topics(dataset1, total_topics, top_topics, state=27):
    """
    Create dominant topics using NMF factorization on two datasets.

    Args:
        dataset1 (list): List of documents for the first dataset.
        dataset2 (list): List of documents for the second dataset.

    Returns:
        tuple: A tuple containing two lists of dominant topics for each dataset.
    """
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(dataset1)
    # NMF factorization
    nmf_model = NMF(n_components=total_topics, random_state=state, max_iter=10000)
    nmf_model.fit(tfidf_matrix)
    
    dominant_topics = np.argsort(nmf_model.transform(tfidf_matrix), axis=1)[:, ::-1][:, :top_topics]
    feature_names = tfidf_vectorizer.get_feature_names_out()
    """ for topic_idx, topic in enumerate(nmf_model.components_):
        top_words_idx = topic.argsort()[-30:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    # Print top words for each topic """

    res = []
    for i in dominant_topics:
        res.append(i)

    
    return res, dominant_topics

# create vectorizer 
def cluster(dataset, num_clusters = 4,tot_pca = 2, plot = False):
    vectorizer = TfidfVectorizer(stop_words='english') 
    
    # vectorizer the text documents 
    vectorized_documents = vectorizer.fit_transform(dataset) 
    print(vectorized_documents.get_shape)
    # reduce the dimensionality of the data using PCA 
    pca = PCA(n_components=tot_pca) 
    reduced_data = pca.fit_transform(vectorized_documents.toarray()) 

    kmeans = KMeans(n_clusters=num_clusters, n_init=5, 
                    max_iter=500, random_state=42) 
    kmeans.fit(vectorized_documents) 
    
    # create a dataframe to store the results 
    results = pd.DataFrame() 
    results['document'] = dataset
    results['cluster'] = kmeans.labels_ 
    
    if plot == True:
      colors = plt.cm.get_cmap('tab10', num_clusters)
      cluster = [f'cluster {i+1}' for i in range(num_clusters)]
      for i in range(num_clusters): 
          plt.scatter(reduced_data[kmeans.labels_ == i, 0], 
                      reduced_data[kmeans.labels_ == i, 1],  
                      s=num_clusters, color=colors(i / num_clusters), 
                      label=f' {cluster[i]}') 
    
      plt.legend() 
      plt.show()
      
    # print the results
    cluster_set = []
    for i in range(len(dataset)):
        cluster_label = results.loc[i, 'cluster']
        cluster_number = cluster_label + 1
        cluster_set.append([cluster_number])
    # Get the cluster label assigned to the element
    

    # Find out which cluster the element belongs to
      # Adding 1 to match cluster numbering starting from 1
    return cluster_set
    
    # plot the results 

    
    