import pandas as pd
import emoji
import re
import string
import spacy
import os


# Loading Polish Model
nlp = spacy.load('pl_core_news_lg')
file_path = input('Full path to .csv file generated previously:')
df = pd.read_csv(file_path)

''' Functions for Dataframe preprocessing for NLP'''

print("Processing of file started -- This may take a little time depends of length of Your list .")
def emoji_counter(text):
    '''
    Adds number of emojis in text
    :param text
    :return: int
    '''
    return emoji.emoji_count(text)


def none_to_empty_string(text):
    ''' Get rid of Nones to avoid Error: Nonetype has no attribute len() also convert text to string
    :param text
    :return empty string if None apperance
    '''
    return '' if text is None else str(text)


def remove_hash(text):
    '''
    Remove hashtags from text
    :param text:
    :return: object
    '''
    cleaned_text = re.sub(r'#\w+', '', text)
    return cleaned_text.strip()

def remove_emojis(text):
    emoji_pattern = r'([^\w\s,!.%-?$])'
    cleaned_text = re.sub(emoji_pattern, '', text)
    return cleaned_text.strip()

def extract_all_emojis(text):
    '''
    Extracting all emojis apperance in text
    :param text:
    :return: str
    '''
    emoji_pattern = r'([^\w\s,!.%-?$])'
    emojis = re.findall(emoji_pattern, text)
    return ''.join(emojis)


def count_engagment(followers, likes, comments):
    return likes + comments / followers

def lenght_of_description(text):
    return len(text)




df['description'] = df['description'].apply(none_to_empty_string)
print('\nGetting rid of nones')
df['description'] = df['description'].apply(remove_hash)
print('\nRemoving hashtags from description')
df['no_emoticons'] = df['description'].apply(emoji_counter)
print('\nCounting emoticons')
df['emoticons'] = df['description'].apply(extract_all_emojis)
print('\nExtracting emoticons')
df['description'] = df['description'].apply(remove_emojis)
print('\nRemoving emoticons')

print('\nCounting Engagment')
df['engagment'] = (df['likes'] + df['comments']) / df['followers']


''' Processing for NLP:
    Removing punctuations like . , ! $( ) * % @ -- emoticons is not considered as punctuations. what about '?' and '!' marks
    Removing URLs -- create function with True or False if in description appears URL. Instead of URL check tag of profile.
    Removing Stop words -- download stop words for PL and delete them.
    Lower casing -- Case: Did uppercase words like 'PROMOCJA' has greater engagment
    Tokenization 
    Stemming 
    Lemmatization -- not every time good idea ()
    
                                                '''


def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree


#wystarczy zanlezc jeden
def check_tag(text):
    tag_pattern = re.compile(r'([@])\w+')
    tags = tag_pattern.findall(text)
    if tags:
        return True
    else:
        return False

#Używać compile w wielokrotnej
def remove_tag(text):
    tag_regex = r'([@])\w+'
    cleaned = re.sub(tag_regex, '', text)
    return cleaned.strip()


def remove_stop_words(text):
    doc = nlp(text)
    filtered_text = ' '.join([token.text for token in doc if not token.is_stop])
    return filtered_text


def tokenize(text):
    doc = nlp(text)
    tokenized = [token for token in doc]
    return tokenized


def lemmatize(text):
    doc = nlp(text)
    lemmatized = [token.lemma_ for token in doc]
    return lemmatized

def list_to_string(list):
    return ''.join(list)



print('\nChecking of difrent profile tag in description')
df["Tagged_profile"] = df['description'].apply(check_tag)
print('\nRemoving Tag')
df["description"] = df['description'].apply(remove_tag)
print('\nRemoving punctuations')
df['description'] = df['description'].apply(remove_punctuation)
print('\nRemoving Polish Stopwords')
df['description'] = df['description'].apply(remove_stop_words)
print('\nTokenizing and Lemmatizing text')
df['Tokenized'] = df['description'].apply(tokenize)
df['descriptionlenght'] = df['Tokenized'].apply(lenght_of_description)
df['Lemmatized'] = df['description'].apply(lemmatize)


print('-------------------------------------------------------------------------------------------------------------')
print("\nPreprocessing done")

file_name = 'IG_proceed.csv'
file_path = os.path.join(os.getcwd(), file_name)
df.to_csv(file_path)
print(f"Proceed .csv file in path {file_path} -- Good luck with further analysis. ")

