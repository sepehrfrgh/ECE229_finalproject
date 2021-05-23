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

@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value'),
)
def update_output_div(input_value):
    sep = title_contains_word(input_value)
    return [
        dash_table.DataTable(
            data=sep.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in sep.columns],
            style_data={
                'width': '100px',
                'maxWidth': '100px',
                'minWidth': '100px',
            },
            style_cell_conditional=[
                {
                    'if': {'column_id': 'Region'},
                    'width': '250px'
                },
            ],
            style_table={
                'overflowX': 'auto'
            }
        )
    ]
def title_contains_word(word):
    """
    input: desired word
    output: books containing that word in their title
    """
    rows = []
    for i in range(df.shape[0]):
        if word in df['book_title'].iloc[i].decode('utf-8'):
            rows.append(i)
    filtered = df.iloc[list(rows)]
    return filtered

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


