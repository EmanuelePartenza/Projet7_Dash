import pandas as pd
import plotly.graph_objects as go # or plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash import Input, Output 


# X_test = pd.read_csv("./Dash_app/X_test_sample.csv",index_col='SK_ID_CURR')
prediction = pd.read_csv("./prediction_X_test_sample_df.csv",index_col='id')
id_list = prediction.index.tolist()
prediction = prediction.to_dict(orient='index')

app = dash.Dash()
app.layout = html.Div([
    dcc.Dropdown(id_list, 'None', id='Id_dropdown'),
    html.Div(id='dd-output-container'),
])

@app.callback(
    Output('dd-output-container', 'children'),
    Input('Id_dropdown', 'value')
)
def update_output(value):
    if value in id_list:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction[value]["y_proba_0"],
            title = {'text': "prediction"},
            # domain = {'x': [0, 1], 'y': [0, 1]}
        ))
    else :
        fig = go.Figure(go.Indicator(
            mode = "gauge",
            value = 0,
            title = {'text': "prediction"},
            domain = {'x': [0, 1], 'y': [0, 1]}
        ))

    return f'You have selected {value}', html.Div([dcc.Graph(figure=fig)])


if __name__ == '__main__':
    # dash_app.run_server(debug=True)
    app.run_server(debug=True, use_reloader=False) 
