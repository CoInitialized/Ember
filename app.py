import base64
import datetime
import io
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
from ember.preprocessing import Preprocessor
from ember.utils import DtypeSelector
from ember.optimize import GridSelector, BayesSelector
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
from ember.impute import GeneralImputer
from ember.preprocessing import GeneralEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ember.autolearn import Learner

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    dcc.Dropdown(
        id='demo-dropdown',
        options=[
            {'label': 'regression', 'value': 'regression'},
            {'label': 'classification', 'value': 'classification'},
            
        ],
        value='None'
    ),
    html.Button(id='submit-button', n_clicks=0, children='Submit'),
    html.Div(id='output-data-upload'),
])


def predict_data():
    pass


def parse_data(contents,filename,last_modified):
    df = None
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return None
    return df



@app.callback(Output('output-data-upload', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('upload-data', 'contents'),
               State('upload-data', 'filename'),
               State('upload-data', 'last_modified'),
               State('demo-dropdown', 'value'),])

def update_on_press(_,contents,filename,last_modified,mode):
    if contents is not None:
        data = parse_data(contents[0],filename[0],last_modified[0])
        if isinstance(data,pd.DataFrame):
            X, y = data.drop(columns = ['class']), data['class']
            lr = Learner(objective=mode,X=X,y=y)
            print("done")
            lr.fit(optimizer='bayes',cat=False)
            print(type(lr))
            return html.Div([
                "Score"
            ])
        else:
            return html.Div([
                'Error parsing the file'
            ])
    else:
        return html.Div([
            'You havent selected any data yet.'
        ])
    
    
# def update_output(list_of_contents, list_of_names, list_of_dates):
#     if list_of_contents is not None:
#         children = [
#             parse_contents(c, n, d) for c, n, d in
#             zip(list_of_contents, list_of_names, list_of_dates)]
#         return children



if __name__ == '__main__':
    app.run_server(debug=False)