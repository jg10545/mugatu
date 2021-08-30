# -*- coding: utf-8 -*-
import logging
import operator
from collections import defaultdict

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