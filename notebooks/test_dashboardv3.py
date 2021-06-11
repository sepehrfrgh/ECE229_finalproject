import pytest
from dashboardv3 import *

def test_recommend_desc():
    test = recommend_desc('Da Vinci Code')
    test = recommend_desc('To Kill a Mockingbird')

def test_book_engine():
    test = book_engine('To Kill a Mockingbird')

def test_get_jaccard_sim():
    test = get_jaccard_sim('To Kill a Mockingbird', 'Da Vinci Code')

