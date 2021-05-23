import dash
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State

df = pd.read_csv('../data/filtered_books.csv', error_bad_lines = False)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.config['suppress_callback_exceptions'] = True

tab_style = {
    'fontWeight': 'bold'
}



app.layout = html.Div(children = [
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Visualization', value='tab-1', style=tab_style),
        dcc.Tab(label='Recommended System', value='tab-2', style=tab_style),
    ]),
    html.Div(id='tabs-content')
])
layout_tab_2 = html.Div(children=[
    html.H6("Recommending you favorite book"),
    html.Div(["Search Your Book By Titles: ", dcc.Input(id='my-input', value='', type='text')]),
    html.Br(),
    html.Div(["Search Your Book By Authors: ", dcc.Input(id='my-input', value='', type='text')]),
    html.Br(),

    html.Div(id='my-output'),

])
test = html.Div(children=[dash_table.DataTable(
    id='table-filtering',
    columns=[
        {"name": i, "id": i} for i in sorted(df.columns)
    ],
    page_current=0,
    page_size=12,
    page_action='custom',

    filter_action='custom',
    filter_query=''
)])
@app.callback(
    Output(component_id='table-filtering', component_property='children'),
    Input(component_id='my-input', component_property='value'),
)
def update_output_div(input_value):
    return input_value

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value'),
)
def render_content(tab):
    if tab == 'tab-1':
        return layout_tab_1
    elif tab == 'tab-2':
        return layout_tab_2

if __name__ == '__main__':
    app.run_server(debug=True)

def title_contains_word(word):
    """
    input: desired word
    output: books containing that word in their title
    """
    assert isinstance(word, str)
    rows = []
    for i in range(books.shape[0]):
        if word in books['book_title'].iloc[i]:
            rows.append(i)
    filtered = books.iloc[list(rows)]

    return filtered

def book_authors(input_string):
    """
    input: title
    output: book authors
    """
    if title_contains_word(input_string).shape[0]==1:
        title = title_contains_word(input_string).iloc[0]['book_title']
        authors = ""
        for i in range(books.shape[0]):
            if title in books['book_title'].iloc[i]:
                authors = books['book_authors'].iloc[i]
    else:
        return "Here the user can choose between the books which are shown or we can force them to be more specific" #future
    return [_.capitalize() for _ in authors.split("|")]