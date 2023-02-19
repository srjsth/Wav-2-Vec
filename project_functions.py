from string import punctuation
from heapq import nlargest
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import spacy
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import regex as re


def save_2_text_file(text, path):
    """ Save to Text File """
    with open(path, "w", encoding='utf8') as f:
        f.write(text)
    return True


class SummarizeSpaCy(object):
    """ Text Summarization using spayCy
    Summarize Document by summarizing each section
    This Class accepts two input arguments:
        text — The input text (a short paragraph or a big chunk of text)
        limit — The number of sentences to be returned.
    """

    """ Class Variables """
    count = 0  # For All Instances

    def __init__(self):
        """   Class Attributes   """
        SummarizeSpaCy.count += 1
        self.word_frequencies = list()
        self.sentence_scores = {}
        self.summary = ""

    """ Class Methods Declaration """
    def tokenize_text(self, text):
        """   Instance Variables: word_frequencies  """
        nlp = spacy.load('en_core_web_sm')  # Load the model (English) into spaCy
        doc = nlp(text)

        """ 1) Tokenize the input text and find out the important tokens ("keywords") in it  
                1 — Convert the input text to lower case and tokenize it with spaCy’s language model.
                2 — Loop over each of the tokens.
                3 — Ignore the token if it is a stopword or punctuation.
                4 — Append the token to a list if it is the part-of-speech tag that we have defined.
        """
        keyword = []  # Tokens
        pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
        doc = nlp(text.lower())  # 1
        for token in doc:  # 2
            if token.text in nlp.Defaults.stop_words or token.text in punctuation:
                continue  # 3
            if token.pos_ in pos_tag:
                keyword.append(token.text)  # 4

        """ 2) Normalize token's ("keywords") weight  
                for better processing by dividing the token’s frequencies by the maximum frequency. """
        self.word_frequencies = Counter(keyword)  # Counter will convert the list into a dictionary with their respective frequency values.
        max_freq = Counter(keyword).most_common(1)[0][1]  # Get the frequency of the top most-common keyword.
        for w in self.word_frequencies:
            self.word_frequencies[w] = (self.word_frequencies[w] / max_freq)  # Loop over each item in the dictionary and normalize the frequency.

        return self

    def find_sent_strength(self, text):
        """   Instance Variables: sentence_scores """
        """ 3) Calculate the importance of the sentences ("sentence_scores") by identifying the occurrence of 
        important keywords and sum up the value. """
        """ Weigh each sentence based on the frequency of the keywords present in each of the sentences. """
        nlp = spacy.load('en_core_web_sm')  # Load the model (English) into spaCy
        doc = nlp(text)
        for sent in doc.sents:  # Loop over each sentence in the text, that are split by the spaCy model based on full-stop punctuation.
            for word in sent:  # Loop over each word in a sentence based on spaCy’s tokenization.
                if word.text in self.word_frequencies.keys():  # Determine if the word is a keyword based on the keywords that we extracted earlier.
                    if sent in self.sentence_scores.keys():
                        # Add normalized keyword value to the key-value pair of the sentence.
                        self.sentence_scores[sent] += self.word_frequencies[word.text]
                    else:
                        # Create a new key-value in the sentence_scores dictionary using the sentence as key
                        # and the normalized keyword value as value.
                        self.sentence_scores[sent] = self.word_frequencies[word.text]
        return self

    def summarize(self, limit):
        """   Instance Variables: summary   """
        """ 4) Generate the Summary """
        temp_summary = []
        # Sort the dictionary based on the normalized value. Set the reverse parameter to True for descending order.
        sorted_x = sorted(self.sentence_scores.items(), key=lambda kv: kv[1], reverse=True)

        counter = 0
        for i in range(len(sorted_x)):  # Loop over each of the sorted items.
            temp_summary.append(str(sorted_x[i][0]).capitalize())  # 15
            counter += 1
            if counter >= limit:
                break  # Break out of the loop if the counter exceeds the limit

        # self.summary = ' '.join(temp_summary)  # Return the list as a string by joining each element with a space.

        return ' '.join(temp_summary)


def summarize_content(section_content, potential_concepts, num_sentences):
    """ 3) Summarize Content using spaCy """
    spaCy_summarizer = SummarizeSpaCy()

    """ Get words & sentence strength """
    freq_words = spaCy_summarizer.tokenize_text(section_content)  #
    sent_strength = spaCy_summarizer.find_sent_strength(section_content)  #
    section_content = spaCy_summarizer.summarize(num_sentences)  # Summarize

    """ Select concepts that are not in candidates but are in summary """
    section_kw_candidates = tokenize_content(section_content)
    section_concepts = [val for val in potential_concepts if val not in section_kw_candidates and val in section_content]
    spaCy_kw = [key for key in sent_strength.word_frequencies if key not in section_kw_candidates and key in section_content]

    return section_content, section_kw_candidates, section_concepts, spaCy_kw


def tokenize_content(doc_content):
    """ Candidate Token Selection using spaCy """
    #  The first step to keyword extraction is producing a set of plausible keyword candidates.
    #  Normally, keywords are either single words or two words. Rarely do we see long keywords
    #  Using scikit-learn’s count vectorizer, we can specify the n-gram range parameter, then obtain the entire list of n-grams
    #  that fall within the specified range.
    n_gram_range = (1, 2)
    stop_words = "english"

    """ Extract candidate words/phrases using NLTK """
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words, preprocessor=custom_preprocessor).fit([doc_content])
    all_candidates = count.get_feature_names()

    # Most often or not, keywords are nouns or noun phrases.
    # "all_candidates" contains some verbs or verb phrases that we do not want included in the list.
    # To extract only candidates that are nouns or noun phrases. We can using spaCy, a powerful NLP library with POS-tagging features.
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(doc_content)
    noun_phrases = set(chunk.text.strip().lower() for chunk in doc.noun_chunks)  # Phrases containing Nouns

    # Select Nouns
    nouns = set()
    for token in doc:
        if token.pos_ == "NOUN":
            nouns.add(token.text)  # Create set of Nouns

    # Combine the set of Nouns with the set of noun phrases we obtained earlier to create a set of all nouns and noun phrases.
    all_nouns = nouns.union(noun_phrases)

    # Candidate selection process
    # Filter the earlier list of all candidates and including only those that are in the all nouns set
    candidates = list(filter(lambda candidate: candidate in all_nouns, all_candidates))

    return candidates


def custom_preprocessor(text):
    text = text.lower()  # lowering the text case
    text = re.sub("\\W", " ", text)  # remove special chars
    text = re.sub(r'\d+', ' ', text)  # Remove Numbers
    return text
