import pandas as pd
import numpy as np
import math
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import string
from collections import Counter
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import dash
import dash_core_components as dcc
import plotly.graph_objects as go
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
# %matplotlib inline
import seaborn as sns

df1 = pd.read_csv('dataset/books_cleaned.csv')
df2 = pd.read_csv('dataset/books_authors_final2.csv')
book_titles_dict1 = df1['book_title'].to_dict()
opt1 = []
for k,v in book_titles_dict1.items():
    temp_d = {}
    temp_d["label"] = v
    temp_d["value"] = v
    opt1.append(temp_d)
book_titles_dict2 = df2['title'].to_dict()
opt2 = []
for k,v in book_titles_dict2.items():
    temp_d = {}
    temp_d["label"] = v
    temp_d["value"] = v
    opt2.append(temp_d)
def book_engine(book):
    titles = list(df2['title'])
    response = requests.get(df2.iloc[0]['image_url'])
    img = Image.open(BytesIO(response.content))

    index = titles.index(book)
    authorsList = df2.iloc[index]['authors'].split("|")
    authorStr = ""
    for i in range(len(authorsList)):
        authorStr += authorsList[i]
        if i == len(authorsList) - 1:
            continue
        authorStr += " | "
    return [df2.iloc[index]['title'], authorStr,
            df2.iloc[index]['average rating'], df2.iloc[index]['genres'], df2.iloc[index]['number of pages'], df2.iloc[index]["reviews' keywords"], df2.iloc[index]['description'],
            df2.iloc[index]['more about author(s)'], df2.iloc[index]['image_url']]
def get_cosine_sim(*strs):
    #     print(strs)
    vectors = [t for t in get_vectors(*strs)]
    return cosine(*vectors)

def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def get_vectors(*strs):
    text = []
    for t in strs:
        t = t.translate(str.maketrans('', '', string.punctuation))
        stop_words = stopwords.words('english')
        stopwords_dict = Counter(stop_words)
        text.append(' '.join([word for word in t.split() if word not in stopwords_dict]))

    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()
def recommend_desc(book):

    if len(df1[df1['book_title'] == book.lower()]) > 0:
        desc = list(df1[df1['book_title'] == book.lower()]['book_desc'])[0]
        print('Found match: ', book, '\n')
        match = book
    else:
        index = np.argmax([fuzz.ratio(book.lower(), i) for i in list(df1['book_title']) if type(i)== str])
        desc = df1.iloc[index,:]['book_desc']
        print('Found closest match: ', df1.iloc[index,:]['book_title'], '\n')
        match = df1.iloc[index,:]['book_title']

    all_desc = list(df1['book_desc'])
    all_genres = list(df1['genres'])
    similarity_array = np.zeros([len(all_desc),])

    genre_input = df1.iloc[df1[df1['book_title'] == match].index[0],:]['genres'].split('|')

    for k, i in enumerate(all_desc):
        if len(list(set(genre_input) & set(all_genres[k].split('|')))) > 2:
            if type(i)==str:
                value = get_cosine_sim(i, desc)
                if value != math.nan:
                    similarity_array[k] = get_cosine_sim(desc, i)
    similarity = similarity_array.tolist()
    similarity.remove(max(similarity))
    final_index = np.nanargmax(similarity)

    response = requests.get(df1.iloc[final_index,:]['image_url'])

    img = Image.open(BytesIO(response.content))

    return [match, df1.iloc[final_index,:]['book_title'], df1.iloc[final_index,:]['book_desc'],
            df1.iloc[final_index,:]['book_rating'], df1.iloc[final_index,:]['book_pages'], df1.iloc[final_index,:]['book_authors'], df1.iloc[final_index,:]['image_url']]

app = dash.Dash()
mint = '#6dc9c0'
sky = '#8cd6e6'
tabs_styles = {
    'height': '60px'
}

tab_style = {
    'borderBottom': '1px solid #b5b3b3',
    'vertical-align': 'middle',
    'background-color':'#6AB187',
    'color':'white',#sky,
#     'padding': '0px',
    'fontSize': 20
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': 'white',
    'color': 'black',#'#57d2eb',
    'fontWeight': 'bold',
    'fontSize': 20,
    'vertical-align': 'middle'
}

body = {
    'height':'100%',
    'margin': '0',
#     'background':'#dcdedc',
    #'background-size': '500px auto'
}

h2 = {
    "color": '#de9714',
    'margin':'auto',
    'text-align':'center',
    'margin-bottom':'10px',
    'margin-top':'10px',
    'font-size':'20px'
}

h6={"color": 'black',
    "font-size":'16px',
    'padding':'1px',
    "margin-top": "1px",
    "margin-bottom": "1px"
#     "margin-left": "10px"
   }




app.layout = html.Div([
html.Header([
    html.H1('Book Hunt Dashboard',
            style={'color':'#191970',
                   'font-weight': 'bold',
                   'margin':'auto',
                   'margin-top':'10px',
                   'margin-bottom':'10px',
                   'text-align': 'center',
                   'font-size':'40px'
                  }
           ),
    html.H2('This is an interactive dashboard for users to explore, search and get recommendations from our book dataset'
           ,style={'font-weight': 'bold','color':'#191970','text-align':'center'})],
    style={"background-image": "url(https://media.istockphoto.com/vectors/books-seamless-pattern-vector-id470721440?k=6&m=470721440&s=612x612&w=0&h=OtHvCXQICZ0hXiJJT0tWYrwZznqYw_ncU307tQDrDUA=",
}),
    dcc.Tabs(id='tabs-example',
             value='Data Exploration',
             children=[
                dcc.Tab(label='Genres Distribution for Different Countries',
                        value='Genres Distribution for Different Countries',
                        style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='Data Exploration Based on Genres',
                        value='Data Exploration Based on Genres',
                        style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='Search Engine',
                        value='Search Engine',
                        style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='Make a Recommandation',
                        value='Make a Recommandation',
                        style=tab_style, selected_style=tab_selected_style)],
             style=tabs_styles),

    html.Div(id='tabs-example-content',style={'width':'80%','margin-top':'10px','margin':'auto'})
    ],style=body)



@app.callback(Output('tabs-example-content', 'children'),
              Input('tabs-example', 'value'))
def render_content(tab):
    """

    This function takes in the name of the tab and generate the content for different tabs.

    :param name: tab.
    :type name: str.
    :returns:  list -- A list of HTML object.

    """
    assert isinstance(tab,str)
    assert tab in ['Genres Distribution for Different Countries',
                    'Data Exploration Based on Genres',
                    'Search Engine',
                    'Make a Recommandation']
    if tab == 'Genres Distribution for Different Countries':
        return [html.P('Choose your desired country to see its distribution of frequent books genres',style={'background':'yellow','font-size':'22px','color':'black','margin':'auto','margin-top':'10px','text-align':'center'}),
            dcc.Graph(
                id='graph_Genres',
                style={'width': '59vh', 'height': '59vh', 'vertical-align': 'middle',
                                  "display": "block",
            "margin-left": "auto",
            "margin-right": "auto",}
            ),
            html.Div('Select a Country:', style={'text-align': 'center',
                                                    'margin':'auto',
                                                    'margin-top':'10px',
                                                    'margin-bottom':'10px',
                                                    'color': '#10b09b',
                                                    'fontSize': 25,
                                                    'font-weight': 'bold'}),
            dcc.Dropdown(
                id = 'countries',
                options=[
                    {"label": "Afghanistan", "value": "Afghanistan" },
                    {"label": "Albania", "value": "Albania" },
                    {"label": "Algeria", "value": "Algeria" },
                    {"label": "American Samoa", "value": "American Samoa" },
                    {"label": "Andorra", "value": "Andorra" },
                    {"label": "Angola", "value": "Angola" },
                    {"label": "Anguilla", "value": "Anguilla" },
                    {"label": "Antarctica", "value": "Antarctica" },
                    {"label": "Antigua and Barbuda", "value": "Antigua and Barbuda" },
                    {"label": "Argentina", "value": "Argentina" },
                    {"label": "Armenia", "value": "Armenia" },
                    {"label": "Australia", "value": "Australia" },
                    {"label": "Austria", "value": "Austria" },
                    {"label": "Azerbaijan", "value": "Azerbaijan" },
                    {"label": "Bahamas", "value": "Bahamas" },
                    {"label": "Bahrain", "value": "Bahrain" },
                    {"label": "Bangladesh", "value": "Bangladesh" },
                    {"label": "Barbados", "value": "Barbados" },
                    {"label": "Belarus", "value": "Belarus" },
                    {"label": "Belgium", "value": "Belgium" },
                    {"label": "Belize", "value": "Belize" },
                    {"label": "Bermuda", "value": "Bermuda" },
                    {"label": "Bhutan", "value": "Bhutan" },
                    {"label": "Bolivia", "value": "Bolivia" },
                    {"label": "Bosnia and Herzegovina", "value": "Bosnia and Herzegovina" },
                    {"label": "Botswana", "value": "Botswana" },
                    {"label": "Brazil", "value": "Brazil" },
                    {"label": "Bulgaria", "value": "Bulgaria" },
                    {"label": "Burkina Faso", "value": "Burkina Faso" },
                    {"label": "Burundi", "value": "Burundi" },
                    {"label": "Cabo Verde", "value": "Cabo Verde" },
                    {"label": "Cambodia", "value": "Cambodia" },
                    {"label": "Cameroon", "value": "Cameroon" },
                    {"label": "Canada", "value": "Canada" },
                    {"label": "Cayman Islands", "value": "Cayman Islands" },
                    {"label": "Central African Republic", "value": "Central African Republic" },
                    {"label": "Chad", "value": "Chad" },
                    {"label": "Chile", "value": "Chile" },
                    {"label": "China", "value": "China" },
                    {"label": "Colombia", "value": "Colombia" },
                    {"label": "Congo Republic", "value": "Congo Republic" },
                    {"label": "Costa Rica", "value": "Costa Rica" },
                    {"label": "Croatia", "value": "Croatia" },
                    {"label": "Cuba", "value": "Cuba" },
                    {"label": "Curaçao", "value": "Curaçao" },
                    {"label": "Cyprus", "value": "Cyprus" },
                    {"label": "Czechia", "value": "Czechia" },
                    {"label": "DR Congo", "value": "DR Congo" },
                    {"label": "Denmark", "value": "Denmark" },
                    {"label": "Djibouti", "value": "Djibouti" },
                    {"label": "Dominica", "value": "Dominica" },
                    {"label": "Dominican Republic", "value": "Dominican Republic" },
                    {"label": "Ecuador", "value": "Ecuador" },
                    {"label": "Egypt", "value": "Egypt" },
                    {"label": "El Salvador", "value": "El Salvador" },
                    {"label": "Eritrea", "value": "Eritrea" },
                    {"label": "Estonia", "value": "Estonia" },
                    {"label": "Eswatini", "value": "Eswatini" },
                    {"label": "Ethiopia", "value": "Ethiopia" },
                    {"label": "Falkland Islands", "value": "Falkland Islands" },
                    {"label": "Faroe Islands", "value": "Faroe Islands" },
                    {"label": "Fiji", "value": "Fiji" },
                    {"label": "Finland", "value": "Finland" },
                    {"label": "France", "value": "France" },
                    {"label": "French Guiana", "value": "French Guiana" },
                    {"label": "French Polynesia", "value": "French Polynesia" },
                    {"label": "French Southern Territories", "value": "French Southern Territories" },
                    {"label": "Gabon", "value": "Gabon" },
                    {"label": "Gambia", "value": "Gambia" },
                    {"label": "Georgia", "value": "Georgia" },
                    {"label": "Germany", "value": "Germany" },
                    {"label": "Ghana", "value": "Ghana" },
                    {"label": "Gibraltar", "value": "Gibraltar" },
                    {"label": "Greece", "value": "Greece" },
                    {"label": "Greenland", "value": "Greenland" },
                    {"label": "Grenada", "value": "Grenada" },
                    {"label": "Guadeloupe", "value": "Guadeloupe" },
                    {"label": "Guam", "value": "Guam" },
                    {"label": "Guatemala", "value": "Guatemala" },
                    {"label": "Guernsey", "value": "Guernsey" },
                    {"label": "Guinea", "value": "Guinea" },
                    {"label": "Guinea-Bissau", "value": "Guinea-Bissau" },
                    {"label": "Guyana", "value": "Guyana" },
                    {"label": "Haiti", "value": "Haiti" },
                    {"label": "Heard Island and McDonald Islands", "value": "Heard Island and McDonald Islands" },
                    {"label": "Honduras", "value": "Honduras" },
                    {"label": "Hong Kong", "value": "Hong Kong" },
                    {"label": "Hungary", "value": "Hungary" },
                    {"label": "Iceland", "value": "Iceland" },
                    {"label": "India", "value": "India" },
                    {"label": "Indonesia", "value": "Indonesia" },
                    {"label": "Iran", "value": "Iran" },
                    {"label": "Iraq", "value": "Iraq" },
                    {"label": "Ireland", "value": "Ireland" },
                    {"label": "Isle of Man", "value": "Isle of Man" },
                    {"label": "Israel", "value": "Israel" },
                    {"label": "Italy", "value": "Italy" },
                    {"label": "Ivory Coast", "value": "Ivory Coast" },
                    {"label": "Jamaica", "value": "Jamaica" },
                    {"label": "Japan", "value": "Japan" },
                    {"label": "Jersey", "value": "Jersey" },
                    {"label": "Jordan", "value": "Jordan" },
                    {"label": "Kazakhstan", "value": "Kazakhstan" },
                    {"label": "Kenya", "value": "Kenya" },
                    {"label": "Kosovo", "value": "Kosovo" },
                    {"label": "Kuwait", "value": "Kuwait" },
                    {"label": "Kyrgyzstan", "value": "Kyrgyzstan" },
                    {"label": "Latvia", "value": "Latvia" },
                    {"label": "Lebanon", "value": "Lebanon" },
                    {"label": "Lesotho", "value": "Lesotho" },
                    {"label": "Liberia", "value": "Liberia" },
                    {"label": "Libya", "value": "Libya" },
                    {"label": "Lithuania", "value": "Lithuania" },
                    {"label": "Luxembourg", "value": "Luxembourg" },
                    {"label": "Macao", "value": "Macao" },
                    {"label": "Madagascar", "value": "Madagascar" },
                    {"label": "Malawi", "value": "Malawi" },
                    {"label": "Malaysia", "value": "Malaysia" },
                    {"label": "Mali", "value": "Mali" },
                    {"label": "Malta", "value": "Malta" },
                    {"label": "Marshall Islands", "value": "Marshall Islands" },
                    {"label": "Martinique", "value": "Martinique" },
                    {"label": "Mauritania", "value": "Mauritania" },
                    {"label": "Mauritius", "value": "Mauritius" },
                    {"label": "Mexico", "value": "Mexico" },
                    {"label": "Micronesia", "value": "Micronesia" },
                    {"label": "Moldova", "value": "Moldova" },
                    {"label": "Mongolia", "value": "Mongolia" },
                    {"label": "Montenegro", "value": "Montenegro" },
                    {"label": "Morocco", "value": "Morocco" },
                    {"label": "Mozambique", "value": "Mozambique" },
                    {"label": "Myanmar", "value": "Myanmar" },
                    {"label": "Namibia", "value": "Namibia" },
                    {"label": "Nepal", "value": "Nepal" },
                    {"label": "Netherlands", "value": "Netherlands" },
                    {"label": "New Caledonia", "value": "New Caledonia" },
                    {"label": "New Zealand", "value": "New Zealand" },
                    {"label": "Nicaragua", "value": "Nicaragua" },
                    {"label": "Niger", "value": "Niger" },
                    {"label": "Nigeria", "value": "Nigeria" },
                    {"label": "Norfolk Island", "value": "Norfolk Island" },
                    {"label": "North Korea", "value": "North Korea" },
                    {"label": "North Macedonia", "value": "North Macedonia" },
                    {"label": "Northern Mariana Islands", "value": "Northern Mariana Islands" },
                    {"label": "Norway", "value": "Norway" },
                    {"label": "Oman", "value": "Oman" },
                    {"label": "Pakistan", "value": "Pakistan" },
                    {"label": "Palestine", "value": "Palestine" },
                    {"label": "Panama", "value": "Panama" },
                    {"label": "Papua New Guinea", "value": "Papua New Guinea" },
                    {"label": "Paraguay", "value": "Paraguay" },
                    {"label": "Peru", "value": "Peru" },
                    {"label": "Philippines", "value": "Philippines" },
                    {"label": "Poland", "value": "Poland" },
                    {"label": "Portugal", "value": "Portugal" },
                    {"label": "Puerto Rico", "value": "Puerto Rico" },
                    {"label": "Qatar", "value": "Qatar" },
                    {"label": "Romania", "value": "Romania" },
                    {"label": "Russia", "value": "Russia" },
                    {"label": "Rwanda", "value": "Rwanda" },
                    {"label": "Réunion", "value": "Réunion" },
                    {"label": "Saint Barthélemy", "value": "Saint Barthélemy" },
                    {"label": "Saint Helena", "value": "Saint Helena" },
                    {"label": "Saint Lucia", "value": "Saint Lucia" },
                    {"label": "Saint Martin", "value": "Saint Martin" },
                    {"label": "Samoa", "value": "Samoa" },
                    {"label": "San Marino", "value": "San Marino" },
                    {"label": "Saudi Arabia", "value": "Saudi Arabia" },
                    {"label": "Senegal", "value": "Senegal" },
                    {"label": "Serbia", "value": "Serbia" },
                    {"label": "Seychelles", "value": "Seychelles" },
                    {"label": "Sierra Leone", "value": "Sierra Leone" },
                    {"label": "Singapore", "value": "Singapore" },
                    {"label": "Slovakia", "value": "Slovakia" },
                    {"label": "Slovenia", "value": "Slovenia" },
                    {"label": "Solomon Islands", "value": "Solomon Islands" },
                    {"label": "Somalia", "value": "Somalia" },
                    {"label": "South Africa", "value": "South Africa" },
                    {"label": "South Georgia and South Sandwich Islands", "value": "South Georgia and South Sandwich Islands" },
                    {"label": "South Korea", "value": "South Korea" },
                    {"label": "South Sudan", "value": "South Sudan" },
                    {"label": "Spain", "value": "Spain" },
                    {"label": "Sri Lanka", "value": "Sri Lanka" },
                    {"label": "St Kitts and Nevis", "value": "St Kitts and Nevis" },
                    {"label": "Sudan", "value": "Sudan" },
                    {"label": "Suriname", "value": "Suriname" },
                    {"label": "Svalbard and Jan Mayen", "value": "Svalbard and Jan Mayen" },
                    {"label": "Sweden", "value": "Sweden" },
                    {"label": "Switzerland", "value": "Switzerland" },
                    {"label": "Syria", "value": "Syria" },
                    {"label": "Taiwan", "value": "Taiwan" },
                    {"label": "Tajikistan", "value": "Tajikistan" },
                    {"label": "Tanzania", "value": "Tanzania" },
                    {"label": "Thailand", "value": "Thailand" },
                    {"label": "Togo", "value": "Togo" },
                    {"label": "Tonga", "value": "Tonga" },
                    {"label": "Trinidad and Tobago", "value": "Trinidad and Tobago" },
                    {"label": "Tunisia", "value": "Tunisia" },
                    {"label": "Turkey", "value": "Turkey" },
                    {"label": "U.S. Virgin Islands", "value": "U.S. Virgin Islands" },
                    {"label": "Uganda", "value": "Uganda" },
                    {"label": "Ukraine", "value": "Ukraine" },
                    {"label": "United Arab Emirates", "value": "United Arab Emirates" },
                    {"label": "United Kingdom", "value": "United Kingdom" },
                    {"label": "United States", "value": "United States" },
                    {"label": "Uruguay", "value": "Uruguay" },
                    {"label": "Uzbekistan", "value": "Uzbekistan" },
                    {"label": "Vanuatu", "value": "Vanuatu" },
                    {"label": "Vatican City", "value": "Vatican City" },
                    {"label": "Venezuela", "value": "Venezuela" },
                    {"label": "Vietnam", "value": "Vietnam" },
                    {"label": "Yemen", "value": "Yemen" },
                    {"label": "Zambia", "value": "Zambia" },
                    {"label": "Zimbabwe", "value": "Zimbabwe" },
                    {"label": "Åland", "value": "Åland" },
                            ],
                value='India'
            )
        ]

    elif tab == 'Data Exploration Based on Genres':
        return [html.P('Choose the desired book attributes and genre to see the distribution of attribute values',style={'background':'yellow','font-size':'22px','color':'black','margin':'auto','margin-top':'10px','text-align':'center'}),
            html.H3('Histogram of Attributes',style={'font-size':'25px','color':'#10b09b','margin':'auto','margin-top':'10px','text-align':'center'}),
            dcc.Graph(
                id='graph_attributes'
            ),
            html.Div('Select a Attribute:', style={'text-align': 'center',
                                                    'margin':'auto',
                                                    'margin-top':'10px',
                                                    'margin-bottom':'10px',
                                                    'color': '#10b09b',
                                                    'fontSize': 25,
                                                    'font-weight': 'bold'}),
            dcc.Dropdown(
                id = 'column_seperated',
                options=[
                    {'label': 'Book rating', 'value': 'book_rating','color':'#57d2eb'},
                    {'label': 'Book rating count', 'value': 'book_rating_count'},
                    {'label': 'Book review count', 'value': 'book_review_count'},
                    {'label': 'Book pages', 'value': 'book_pages'},
                    {'label': 'Book format', 'value': 'book_format'},
                    {'label': 'Amazon recommanded', 'value': 'Amzn_rcmd'}
                            ],
                value='book_rating'
            ),
            html.Div('Select a Genre:', style={'text-align': 'center',
                                                    'margin':'auto',
                                                    'margin-top':'10px',
                                                    'margin-bottom':'10px',
                                                    'color': '#10b09b',
                                                    'fontSize': 25,
                                                    'font-weight': 'bold'}),
            dcc.Dropdown(
                id = 'Genres',
                options=[
                    {'label': 'All', 'value': 'All'},
                    {'label': 'Young Adult', 'value': 'Young Adult'},
                    {'label': 'Fiction', 'value': 'Fiction'},
                    {'label': 'Science', 'value': 'Science'},
                    {'label': 'Dystopia', 'value': 'Dystopia'},
                    {'label': 'Fantasy', 'value': 'Fantasy'},
                    {'label': 'Classics', 'value': 'Classics'},
                    {'label': 'Historical', 'value': 'Historical'},
                    {'label': 'Academic', 'value': 'Academic'},
                    {'label': 'Romance', 'value': 'Romance'},
                    {'label': 'Paranormal', 'value': 'Paranormal'},
                    {'label': 'Literature', 'value': 'Literature'},
                    {'label': 'Adventure', 'value': 'Adventure'},
                    {'label': 'Mythology', 'value': 'Mythology'},
                    {'label': 'Childrens', 'value': 'Childrens'},
                    {'label': 'Horror', 'value': 'Horror'},
                    {'label': 'Psychology', 'value': 'Psychology'},
                    {'label': 'Philosophy', 'value': 'Philosophy'}
                            ],
                value='All'
            ),
            html.Div('Percentage of data to display:', style={'text-align': 'center',
                                                    'margin':'auto',
                                                    'margin-top':'10px',
                                                    'margin-bottom':'10px',
                                                    'color': '#10b09b',
                                                    'fontSize': 25,
                                                    'font-weight': 'bold'}),
            dcc.RangeSlider(
                id='slider',
                min=0,
                max=100,
                step=1,
                value=[0, 100],
                marks={
                    0: {'label':'0'},
                    25: {'label':'25'},
                    50: {'label':'50'},
                    75: {'label':'75'},
                    100: {'label':'100'}
                }
            )
        ]

    elif tab == 'Search Engine':
        return [html.P('This is a comprehensive search engine for book dataset. Type your book title or select dropdown to see its detailed information',style={'background':'yellow','font-size':'22px','color':'black','margin':'auto','margin-top':'10px','text-align':'center'}),
                html.Br(),
                         dcc.Dropdown(
                             id='demo-dropdown2',
                             options= opt2,
                             value='To Kill a Mockingbird',
#                              style={"width":"1px"}
                         ), html.Br(),

                         html.Div(html.Center(id='url1')),

                         html.Div(
                             html.Table([
                                 html.Tr([html.Th(html.H6(' Title:',style={"color": 'black', "font-size":"16px",'padding':'1px',"margin": "1px", "margin-left": "10px"})), html.Td(id='Title')], style={"border": "2px solid gray", "font-size":"14px"}),
                                 html.Tr([html.Th(html.H6(' Author(s):',style={"color": 'black', "font-size":"16px", 'padding':'1px',"margin": "1px","margin-left": "10px"})), html.Td(id='Author')], style={"border": "2px solid grey", "font-size":"14px"}),
                                 html.Tr([html.Th(html.H6(' Average rating:',style={"color": 'black', "font-size":"16px",'padding':'1px',"margin": "1px", "margin-left": "10px"})), html.Td(id='Average_rating')], style={"border": "2px solid grey", "font-size":"14px"}),
                                 html.Tr([html.Th(html.H6(' Genres:',style={"color": 'black','font-size':"16px",'padding':'1px',"margin": "1px", "margin-left": "10px"})), html.Td(id='Genres')], style={"border": "2px solid grey", "font-size":"14px"}),
                                 html.Tr([html.Th(html.H6(' Number of Pages:',style={"color": 'black', "font-size":"16px",'padding':'1px',"margin": "1px", "margin-left": "10px"})), html.Td(id='Pages')], style={"border": "2px solid grey", "font-size":"14px"}),
                                 html.Tr([html.Th(html.H6(' Keywords from review(s):',style={"color": 'black', "font-size":"16px",'padding':'1px',"margin": "1px", "margin-left": "10px"})), html.Td(id='Keywords')], style={"border": "2px solid grey", "font-size":"14px"}),
                                 html.Tr([html.Th(html.H6(' More information about the Author(s):',style={"color": 'black', "font-size":"16px", 'padding':'1px',"margin": "1px","margin-left": "10px"})), html.Td(id='Info')], style={"border": "2px solid grey", "font-size":"14px"}),
                                 html.Tr([html.Th(html.H6(' Description:',style={"color": 'black', "font-size":"16px", 'padding':'1px',"margin": "1px","margin-left": "10px"})), html.Td(id='Desc')], style={"border": "2px solid grey", "font-size":"14px"}),
                             ], style={"border-spacing": "10px",
                                       "border": "2px #57d2eb",
                                       'float':'right',
                                       "margin":"auto",
                                       'width':'70%',
                                       'margin-top':'10px',
                                       "background-color": "Aliceblue",
                                       "opacity": "0.8"
#                                        "width":"1100px"
                                      })),html.Div(style={'height':"100%", 'background':"grey"
                                      })]

    elif tab == 'Make a Recommandation':
        return html.Div(children =[html.P('This is a recommendation system that suggests you closest book to your search. Type your book title or select dropdown to get our recommendation for the next book to read',style={'background':'yellow','font-size':'22px','color':'Black','margin':'auto','margin-top':'10px','text-align':'center'}),
                                   html.Br(),
                       dcc.Dropdown(
                           id='demo1-dropdown',
                           options= opt1,
                           value=''
                       ), html.Br(),html.Div( html.Center(id='url')),
                       html.Table([
                           html.Tr([html.Td(html.H6('Top Reccomendation',style=h6)), html.Td(id='top-rec')], style={"border": "2px solid grey", "font-size":"14px"}),
                           html.Tr([html.Td(html.H6('Author',style=h6)), html.Td(id='author')]),
                           html.Tr([html.Td(html.H6('Description',style=h6)), html.Td(id='desc')]),
                           html.Tr([html.Td(html.H6('Rating',style=h6)), html.Td(id='rating')]),
                           html.Tr([html.Td(html.H6('Length',style=h6)), html.Td(id='length')]),

                       ],style={'background-color':'#e4f2e9',
                               'opacity':'0.8'}),
            ])

@app.callback(
    Output('Title', 'children'),
    Output('Author', 'children'),
    Output('Average_rating', 'children'),
    Output('Genres', 'children'),
    Output('Pages', 'children'),
    Output('Keywords', 'children'),
    Output('Desc', 'children'),
    Output('Info', 'children'),
    Output('url1', 'children'),
    [dash.dependencies.Input('demo-dropdown2', 'value')])
def update_output1(value):
    x = book_engine(value)
    img = html.Img(src= x[8], style={'float':'left','width':'20%',"min-width": "300px", "height": "400px",'margin-bottom':'20px','margin':'auto'})
    return x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], img

@app.callback(
    Output('top-rec', 'children'),
    Output('author', 'children'),
    Output('desc', 'children'),
    Output('rating', 'children'),
    Output('length', 'children'),
    Output('url', 'children'),
    [dash.dependencies.Input('demo1-dropdown', 'value')])
def update_output2(value):
    x = recommend_desc(value)
    img = html.Img(src= x[6], style={'width':'10%',"min-width": "300px", "maxheight": "600px",'margin-bottom':'20px','margin':'auto'})
    l = '{} pages.'.format(str(int(x[4])))
    return x[1],  x[5],  x[2], x[3], l, img

@app.callback(
    Output("graph_attributes", "figure"),
    [Input("column_seperated", "value"),
     Input("Genres", "value"),
     Input('slider', 'value')])
def display_chart(attr, genere,slider):
    df=pd.read_csv('dataset/books_added_amzn.csv',error_bad_lines = False)
    if(genere != 'All'):
        df = df[df['genres'].astype(str).str.contains(genere)]
#         df = df[df['genres'].astype(str).str.contains('Fiction')]['book_rating'].to_frame()
    df[attr].to_frame()
    df.sort_values(by=[attr])
    upper_lim = int(df.shape[0]*slider[1]/100)
    lower_lim = int(df.shape[0]*slider[0]/100)
    df = df.loc[lower_lim:upper_lim]


    fig = px.histogram(df, x=attr,color_discrete_sequence=[sky])
    return fig


@app.callback(
    Output("graph_Genres", "figure"),
    Input("countries", "value")
)
def display_genres_from_country(attr):
    df=pd.read_csv('dataset/countries_genres_freq2csv.csv', error_bad_lines = False)
    df_genres_list = list(df['genres'])
    fig = go.Figure(go.Barpolar(r=list(df[attr]),
                                theta=df_genres_list,
                                marker_color='coral'))
    fig.update_layout(title={'text':'Distribution of Frequent genres for {}'.format(attr),'x':0.5,'y':0.9,'xanchor': 'center','yanchor': 'top'} )
    return fig


if __name__ =='__main__':
    app.run_server(host="0.0.0.0")
