from nltk.corpus import wordnet as wn
from nltk.tokenize import RegexpTokenizer,sent_tokenize
import nltk
import re

def preprocess_text(input_text, contractions, discourse_markers):
    preprocessed_text = []
    sentences = sent_tokenize(input_text)
    tokens = set(nltk.word_tokenize(input_text.lower()))
    for sentence in sentences:
        # sentence = sentence.strip()
        # print("Original sentence:", sentence)
        sentence, contractions_applied = contraction_replacement(sentence, contractions)
        if not contractions_applied:
            # applying sentence simplification steps
            sentence = remove_parenthesis(sentence)
            sentence = remove_discourse_markers(sentence, discourse_markers)
            sentence = remove_appositions(sentence)
        sentence = synonym_substitution(sentence, tokens)
        # print("Obfuscated sentence:", sentence)
        preprocessed_text.append(sentence)
    preprocessed_text = " ".join(preprocessed_text)
    # print(preprocessed_text)
    return preprocessed_text

def get_synset_name(synset):
    synset = synset.split('-')
    offset = int(synset[0])
    pos = synset[1]
    return wn.synset_from_pos_and_offset(pos, offset)

def contraction_replacement(sentence, contractions):
    orig_sentence = sentence

    all_contractions = contractions.keys()
    all_expansions = contractions.values()
    contractions_count = 0
    expansions_count = 0

    for contraction in all_contractions:
        if contraction.lower() in sentence.lower():
            contractions_count += 1
    for expansion in all_expansions:
        if expansion.lower() in sentence.lower():
            expansions_count += 1

    if contractions_count > expansions_count:
        # Replace all contractions with their expansions
        temp_contractions = dict((k.lower(), v) for k, v in contractions.items())
        for contraction in all_contractions:
            if contraction.lower() in sentence.lower():
                case_insesitive = re.compile(re.escape(contraction.lower()), re.IGNORECASE)
                sentence = case_insesitive.sub(temp_contractions[contraction.lower()], sentence)
        contractions_applied = True
    elif expansions_count > contractions_count:
        inv_map = {v: k for k, v in contractions.items()}
        temp_contractions = dict((k.lower(), v) for k, v in inv_map.items())
        for expansion in all_expansions:
            if expansion.lower() in sentence.lower():
                case_insesitive = re.compile(re.escape(expansion.lower()), re.IGNORECASE)
                sentence = case_insesitive.sub(temp_contractions[expansion.lower()], sentence)
        contractions_applied = True
    else:
        contractions_applied = False

    # print("Original:::::", orig_sentence)
    # print("Total Contractions: ", contractions_count)
    # print("Total Expansions: ", expansions_count)
    # print("Obfuscated:::::", sentence)
    # print("============================================")

    return sentence, contractions_applied

def remove_parenthesis(sentence):
    sentence = re.sub(r" ?\([^)]+\)", "", sentence)
    return sentence

def remove_discourse_markers(sentence, discourse_markers):

    sent_tokens = sentence.lower().split()
    for marker in discourse_markers:
        if marker.lower() in sent_tokens:
            case_insesitive = re.compile(re.escape(marker.lower()), re.IGNORECASE)
            sentence = case_insesitive.sub('', sentence)

    return sentence

def remove_appositions(sentence):
    sentence = re.sub(r" ?\,[^)]+\,", "", sentence)
    return sentence

def untokenize(words):
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .', '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
        "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()

def synonym_substitution(sentence, all_words):
    new_tokens = []
    tokenizer = RegexpTokenizer(r'\w+')
    output = tokenizer.tokenize(sentence)
    for token in output:
        try:
            synset_name = get_synset_name(wn.synsets(token))
            synonyms = synset_name.lemma_names()
            # print(token, ":::::", synonyms)
            for synonym in synonyms:
                if synonym.lower() not in all_words:
                    token = synonym
                    break
        except Exception as e:
            # print(e)
            pass
        new_tokens.append(token)

    final = untokenize(new_tokens)
    final = final.capitalize()
    return final

def getContractions():
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "I'd": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it had",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there had",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'alls": "you alls",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had",
        "you'd've": "you would have",
        "you'll": "you you will",
        "you'll've": "you you will have",
        "you're": "you are",
        "you've": "you have"
    }
    return contractions

def getDiscourseMarkers():
    all_lines = open("discourse_markers.txt", "r").readlines()
    discourse_markers = []
    for line in all_lines:
        line = line.strip()
        discourse_markers.append(str(line))
    return discourse_markers