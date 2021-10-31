# -*- coding: utf-8 -*-
import logging
import operator
import re
from collections import defaultdict
import sklearn.feature_extraction

try:
    import RAKE.RAKE
except:
    logging.debug("unable to import the python-rake library")
    
    
    
def _generate_candidate_keyword_scores(phrase_list, word_score, minFrequency):
    """
    Replacement for RAKE.RAKE.generate_candidate_keyword_scores() that runs
    significantly faster.
    """
    keyword_candidates = {}
    # iterate over the list once to count how many times
    # each phrase occurs
    phrase_counts = defaultdict(int)
    for p in phrase_list:
        phrase_counts[p] += 1
    # and then a second time to compute RAKE scores
    for phrase in phrase_list:
        if phrase_counts[phrase] >= minFrequency:
            keyword_candidates.setdefault(phrase, 0)
            word_list = RAKE.RAKE.separate_words(phrase)
            candidate_score = 0
            for word in word_list:
                candidate_score += word_score[word]
            keyword_candidates[phrase] = candidate_score
    return keyword_candidates

def fasterrake(text, max_words=5, min_characters=1, min_frequency=1, stopwords=None):
    """
    Run through the RAKE pipeline
    
    :text: str; concatenated text of your corpus
    :max_words: int; max number of words per phrase
    :min_characters: int; minimum number of characters per token
    :min_frequency: int; minimum number of times a phrase has to appear in the corpus
    :stopwords: list of stopwords; if None use the Smart list
    
    Returns a list of (keyword, RAKE score) tuples sorted in descending order of score
    """
    if stopwords is None:
        stopwords = RAKE.SmartStopList()
    stop_pattern = RAKE.RAKE.build_stop_word_regex(stopwords)
    sentence_list = RAKE.RAKE.split_sentences(text)
    phrase_list = RAKE.RAKE.generate_candidate_keywords(sentence_list, stop_pattern,
                                                        min_characters, max_words)
    word_scores = RAKE.RAKE.calculate_word_scores(phrase_list)
    keyword_candidates = _generate_candidate_keyword_scores(phrase_list, word_scores,
                                                            min_frequency)
    sorted_keywords = sorted(keyword_candidates.items(), key=operator.itemgetter(1),
                             reverse=True)
    return sorted_keywords


def build_rake_tdm(corpus, max_words=5, min_characters=1, min_frequency=1, 
                   stopwords=None, remove_urls=True):
    """
    
    """
    # compile the regex sklearn uses for tokenization
    sklearn_pattern = re.compile('(?u)\\b\\w\\w+\\b')
    # and the reged RAKE uses for word separation
    splitter = re.compile('(?u)\W+') # from RAKE.RAKE.separate_words
    # this substitute function will strip out hyphens and stuff and replace with
    # a space. so RAKE won't map "big time" and "big-time" to separate keywords.
    corpus = [re.sub(splitter, " ", c) for c in corpus]
    # run RAKE on the entire corpus
    keywords = fasterrake("\n".join(corpus), max_words=max_words, 
                           min_characters=min_characters, min_frequency=min_frequency, 
                           stopwords=stopwords)
    # strip out keywords sklearn won't recognize or ones that have a carriage
    # return in them for some reason
    keyword_vocab = [k[0] for k in keywords if 
                     bool(re.match(sklearn_pattern, k[0]))&("\n" not in k[0])]
    if remove_urls:
        keyword_vocab = [k for k in keyword_vocab if "http" not in k]
    # vectorize the corpus to a sparse document-keyword matrix scaled with
    # TF-IDF.
    vec = sklearn.feature_extraction.text.TfidfVectorizer(vocabulary=keyword_vocab,
                                                          ngram_range=(1, max_words))
    tdm = vec.fit_transform(corpus)
    return keyword_vocab, tdm                              