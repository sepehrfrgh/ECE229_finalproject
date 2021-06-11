import pandas as pd
import numpy as np
import urllib
import matplotlib.pyplot as plt
import io
import scipy.misc
import imageio
import nltk
from nltk.stem.porter import PorterStemmer
from scipy.spatial import distance
import re
import string
import spacy
import math
import plotly.graph_objects as go
from collections import Counter
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import warnings


books = pd.read_csv("filtered_books.csv")
input_string = "harry potterr"


def desc_contain_word(word, dataset =books):
    """
    This function returns all the books whose descriptions contain a specific substring
    Arg(s):
        :param1 value: an integer for the pages threshold
        :type value: int
        :param2 value: the input dataframe
        :type value: pd.core.frame.DataFrame
    returns:
        :return: dataframe containing all the books written with fewer pages than the threshold
        :rtype: pd.core.frame.DataFrame
    """
    assert isinstance(word, str)
    assert isinstance(dataset, pd.core.frame.DataFrame)
    books = books[books['book_desc'].notna()]  #future: this is the full description
    rows = []
    for i in range(len(books)):
        if word in books.iloc[i]['book_desc']:
            rows.append(i)
    filtered = books.iloc[list(rows)]
    #print(len(filtered), " books with a rating above ", rate_above)
    return filtered
#desc_contain_word("the big bang theory,")

def isEnglish(word):
    """
    This function returns if a word is in English language or not
    Arg(s):
        :param value: a word
        :type value: int
    returns:
        :return: boolean value demonstrating whether the given string is in English or not
        :rtype: bool
    """
    assert isinstance(word, str)
    try:
        word.encode(encoding="utf-8").decode("ascii")
    except UnicodeDecodeError:
        return False
    else:
        return True

def clean(text):
    """
    This function cleans the input text using simple nlp techniques
    Arg(s):
        :param value: a string that needs to be cleaned
        :type value: str
    returns:
        :return: cleaned text in terms of a list
        :rtype: list
    """
    assert isinstance(text, str)
    cleaned = []
    #checking for stopwords
    for w in text.split():
        w = w.lower()
        alphabet = re.compile('[^a-zA-Z]')
        w = alphabet.sub('', w)
        w = w.translate(str.maketrans('', '', punctuation))
        if w not in stopword_list and isEnglish(w) and len(w)>1:
            if len(w) < 40:
                cleaned.append(w)
    ##Stemming
    #porter = PorterStemmer()
    #cleaned = [porter.stem(w) for w in cleaned]
    cleaned = np.unique(cleaned)
    return cleaned


def return_isbn(rev):
    """
    Given the messy string of reiews, this function returns the isbn of a book
    Arg(s):
        :param value: a messy review containing tags and isbn
        :type value: str
    returns:
        :return: the isbn of a book
        :rtype: str
    """
    assert isinstance(rev,str)
    s = [m.start() for m in re.finditer('asin', rev)]
    e = [m.end() for m in re.finditer('"",  ""reviewerName"":',  rev)]
    if len(s)>0 and len(e)>0:
        isbn = rev[s[0]+10:e[0]-23]
    else:
        isbn = "no isbn"
    #assert len(isbn)==9
    return isbn

def return_info(i, authors_df = books):
    """
    This function returns the information of the authors of a book
    Arg(s):
        :param1 value: an integer inicating the index in the dataframe
        :type value: int
        :param2 value: a dataframe
        :type value: pd.core.frame.DataFrame
    returns:
        :return: information of the authors corresponding to a paper's index
        :rtype: pd.core.frame.DataFrame
    """
    assert isinstance(authors_df, pd.core.frame.DataFrame)
    assert isinstance(i, int)
    assert i>=0 and i < len(authors_df)
    website = authors_df['website'].iloc[i]
    twitter = authors_df['twitter'].iloc[i]
    country = authors_df['country'].iloc[i]
    gender = authors_df['gender'].iloc[i]
    works = authors_df['workcount'].iloc[i]
    average_rating = authors_df['average_rate'].iloc[i]
    info = ''
    if  str(website) != 'nan':
        info += "website: " + website + " | "
    if str(twitter) != 'nan':
        info += "twitter ID: " + twitter+ " | "
    if str(gender) != 'nan' and str(gender) != 'unknown' :
        info += "gender: " + gender+ " | "
    if str(country) != 'nan':
        info += "Country: " + country   + " | "
    if str(works)  != 'nan' and str(average_rating) != 'nan':
        info += "average ratings of {} works: {}".format(works, average_rating)
    #authors_df.head()
    return info

def title_contains_word(word):
    """
    This function gets a query word and returns all the books containing that specific word in their title
    Arg(s):
        :param value: desired word which will be searched in all books' titles
        :type value: str
    returns:
        :return: filtered dataset containing books which contain the querried word in their title
        :rtype: pd.core.frame.DataFrame
    """
    assert isinstance(word, str)
    rows = []
    for i in range(books.shape[0]):
        if word in books['book_title'].iloc[i]:
            rows.append(i)
    filtered = books.iloc[list(rows)]
    #print(len(filtered), " books with a rating above ", rate_above)
    return filtered
#title_contains_word("the great brain")



def book_authors(input_string):
    """
    This function gets the title of the book and returns list of authors
    Arg(s):
        :param value: title of the book
        :type value: str
    returns:
        :return: list of the authors of the book
        :rtype: list
    """
    if title_contains_word(input_string).shape[0]==1:
        title = title_contains_word(input_string).iloc[0]['book_title']
        #assert isinstance(title, str)
        authors = ""
        for i in range(books.shape[0]):
            if title in books['book_title'].iloc[i]:
                authors = books['book_authors'].iloc[i]
        #print(len(filtered), " books with a rating above ", rate_above)
    return [_.capitalize() for _ in authors.split("|")]
#book_authors("more adventures of the ")


def plot_image(input_string):
    """
    This function returns the image of the book (by title)
    Arg(s):
        :param value: title of the book
        :type value: str
    returns:
        :return: plotting the image
        :rtype: Nonetype
    """
    if title_contains_word(input_string).shape[0]==1:
        title = title_contains_word(input_string).iloc[0]['book_title']
        #print(title)
        url = books[books['book_title']==title]['image_url'].iloc[0]
        image = imageio.imread(url)
        plt.imshow(image)
        plt.axis("off")
        plt.show()
    else:
        return "Here the user can choose between the books which are shown or we can force them to be more specific" #future
#plot_image("more adventures of the ")


def book_genres(input_string):
    """
    This function returns the genres of a book (by title)
    Arg(s):
        :param value: title of the book
        :type value: str
    returns:
        :return: genres of the corresponding book
        :rtype: list
    """
    if title_contains_word(input_string).shape[0]==1:
        title = title_contains_word(input_string).iloc[0]['book_title']
        #print(title)
        genres = books[books['book_title']==title]['genres'].iloc[0].split("|")
    return genres
book_genres(input_string)


def book_desc(input_string, dataset=books):
    """
    This function returns the description of the book
    Arg(s):
        :param1 value: title of the book
        type value: str
        :param2 value: input dataset
        :type value:
    returns:
        :return: description for the querried title
        :rtype: pd.core.frame.DataFrame
    """
    assert isinstance(input_string, str)
    assert isinstance(dataset, pd.core.frame.DataFrame)
    if title_contains_word(input_string).shape[0]==1:
        title = title_contains_word(input_string).iloc[0]['book_title']
        #print(title)
        desc = books[books['book_title']==title]['book_desc'].iloc[0]
    return desc
#book_desc(input_string)


def book_pages(input_string):
    """
    This function returns the number of pages of a book
    Arg(s):
        :param value: title of the book
        :type value: str
    returns:
        :return: number of pages of a book
        :rtype: int
    """
    if title_contains_word(input_string).shape[0]==1:
        title = title_contains_word(input_string).iloc[0]['book_title']
        #print(title)
        pages = books[books['book_title']==title]['book_pages'].iloc[0]
    return int(pages)


def book_rating(input_string):
    """
    This function returns the rating of the querried book
    Arg(s):
        :param value: title of the book
        :type value: str
    returns:
        :return: rating of the book
        :rtype: float
    """
    if title_contains_word(input_string).shape[0]==1:
        title = title_contains_word(input_string).iloc[0]['book_title']
        #print(title)
        rating = books[books['book_title']==title]['book_rating'].iloc[0]
    return np.round(rating, 5)

def author_book(author):
    """
    This function returns the books of an author
    Arg(s):
        :param value: name of an author
        :type value: str
    returns:
        :return: list of books written by the author
        :rtype: list
    """
    assert isinstance(author, str)
    authors_books = []
    for i in range(books.shape[0]):
        if author in books['book_authors'].iloc[i]:
            authors_books.append(books["book_title"].iloc[i])
    #print(len(filtered), " books with a rating above ", rate_above)
    return authors_books
#author = books.iloc[500]['book_authors']
#print("author : ", author)
#author_book(books.iloc[500]['book_authors'])

def other_books_by_author(input_string, dataset = books):
    """
    This function returns other books by all author's of a given title.
    Arg(s):
        :param1 value: name of a book
        :type value: str
        :param2 value: an input dataframe
        :rtype: pd.core.frame.DataFrame
    returns:
        :output (list): list of books written by all the authors of the book
    """
    assert isinstance(input_string, str)
    assert isinstance(dataset, pd.core.frame.DataFrame)
    if title_contains_word(input_string).shape[0]==1:
        title = title_contains_word(input_string).iloc[0]['book_title']
        #print(title)
        book_list = []
        authors = books[books['book_title']==title]['book_authors'].iloc[0].split("|")
        for aut in authors:
            book_l = author_book(aut)
            for b in book_l:
                book_list.append(b)
    return book_list


def page_less_thresh(page_less):
    """
    This function returns all the books with pages less than a threshold
    Arg(s):
        :param value: an integer for the pages threshold
        :type value: int
    returns:
        :return: dataframe containing all the books written with fewer pages than the threshold
        :rtype: pd.core.frame.DataFrame
    """
    assert isinstance(page_less, int)
    assert page_less >=0
    filtered = books[books["book_pages"] >0.]
    filtered = filtered[filtered["book_pages"] <=page_less]
    #print(len(filtered), " books with a rating above ", rate_above)
    return filtered



def page_above_thresh(page_above):
    """
    This function returns all the books with pages more than a threshold
    Arg(s):
        :param value: an integer for the pages threshold
        :type value: int
    returns:
        :return: dataframe containing all the books written with more pages than the threshold
        :rtype: pd.core.frame.DataFrame
    """
    assert isinstance(page_above, int)
    assert page_above >=0
    filtered = books[books["book_pages"] >=page_above]
    #print(len(filtered), " books with a rating above ", rate_above)
    return filtered
#page_above_thresh(10000)
