import pandas as pd
import numpy as np
import math
import nltk
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

df1 = pd.read_csv('../book-reccomendation-engine/books_cleaned.csv')
df2 = pd.read_csv('../keyword_engine/books_authors_final.csv')
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
    """
    Model to recommend a book based on another book. Model uses book descriptions, authors, and genres to find the book with the closest closest similarity
    :param book: String for book to be looked up. Does not have to be exact match.
    :type book: String
    :return:
    [Corrected string of input,
    Recommended book Title,
    Recommended Book Description,
    Recommended Book Rating,
    Recommended Book Length,
    Recommended Book Author,
    Recommended Book URL]

    :rtype: List
    """

    assert isinstance(book, str)

    if len(df1[df1['book_title'] == book.lower()]) > 0:
        desc = list(df1[df1['book_title'] == book.lower()]['book_desc'])[0]
        # print('Found match: ', book, '\n')
        assert desc is not ''
        match = book
    else:
        index = np.argmax([fuzz.ratio(book.lower(), i) for i in list(df1['book_title']) if type(i)== str])
        desc = df1.iloc[index,:]['book_desc']
        print('Found closest match: ', df1.iloc[index,:]['book_title'], '\n')
        match = df1.iloc[index,:]['book_title']
        assert isinstance(match, str)

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

    assert response is not ''

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
    'borderBottom': '1px solid #d6d6d6',
    'vertical-align': 'middle',
    'color':'#ff5f03',#sky,
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

    html.H1('Book Hunt Dashboard',
            style={'color':'#e04a09',
                   'margin':'auto',
                   'margin-top':'10px',
                   'margin-bottom':'10px',
                   'text-align': 'center',
                   'font-size':'40px'
                  }
           ),
    html.Div('This ia an interactive dashboards for users to explore, search and get recommandation from our book dataset',
           style=h2),
    dcc.Tabs(id='tabs-example',
             value='Data Exploration',
             children=[
                dcc.Tab(label='Data Exploration based on Generes',
                        value='Data Exploration based on Generes',
                        style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='Search Engine',
                        value='Search Engine',
                        style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='Make a Recommandation',
                        value='Make a Recommandation',
                        style=tab_style, selected_style=tab_selected_style)],
             style=tabs_styles),

    html.Div(id='tabs-example-content',style={'width':'80%','margin-top':'10px','margin':'auto'})
    ])



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
                    
    if tab == 'Data Exploration based on Generes':
        return [
            html.H3('Histogram of Attributes',style={'font-size':'25px','color':'#10b09b','margin':'auto','margin-top':'10px','text-align':'center'}),
            dcc.Graph(
                id='graph_attributes'
            ),
            html.Div('Select an Attribute:', style={'text-align': 'center',
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
            html.Div('Select an Genere:', style={'text-align': 'center',
                                                    'margin':'auto',
                                                    'margin-top':'10px',
                                                    'margin-bottom':'10px',
                                                    'color': '#10b09b',
                                                    'fontSize': 25,
                                                    'font-weight': 'bold'}),
            dcc.Dropdown(
                id = 'generes',
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
            html.Div('Percent of Data to Display:', style={'text-align': 'center',
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
        return html.Div([html.Br(),
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
                                      }))])

    elif tab == 'Make a Recommandation':
        return html.Div(children =[html.Br(),
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
def update_output2_v3(value):
    """ The purpose of this function is to generate the output for the dash tab "Make a Recommendation". The function takes in the value of the user selected drop down to display the top recommendations.

        :param value: User selected book
        :type value: str

        :return: Returns values for different book attributes of the top recommended book
        :rtype: str
    """
    try:
        x = recommend_desc(value)
        img = html.Img(src= x[6], style={'width':'10%',"min-width": "300px", "maxheight": "600px",'margin-bottom':'20px','margin':'auto'})
        l = '{} pages.'.format(str(int(x[4])))
        return x[1],  x[5],  x[2], x[3], l, img
    except Exception as E:
        return str('Recommender did not load %s' %E)

@app.callback(
    Output("graph_attributes", "figure"),
    [Input("column_seperated", "value"),
     Input("generes", "value"),
     Input('slider', 'value')])
def display_chart(attr, genere,slider):
    df=pd.read_csv('../data/books_added_amzn.csv',error_bad_lines = False)
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

# @app.callback(
#     Output("graph", "figure"),
#     [Input("column", "value")])
# def display_hist(attr):
#     df=pd.read_csv('../data/books_added_amzn.csv',error_bad_lines = False)
#     fig = px.histogram(df, x=attr)
#     return fig


if __name__ =='__main__':
    app.run_server(port=4050)
