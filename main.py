import pandas as pd
import plotly.graph_objects as go # or plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash import Input, Output 
import numpy as np

prediction = pd.read_csv("./prediction_X_test_sample_df.csv",index_col='id')
id_list = prediction.index.tolist()
prediction = prediction.to_dict(orient='index')

app = dash.Dash()
app.layout = html.Div([
    html.Label('Inserez le code du client'),
        dcc.Dropdown(id_list, 'None', id='input_dropdown_id'),
    html.Div(id='output_dropdown_id')
])

@app.callback(
    Output('output_dropdown_id', 'children'),
    Input('input_dropdown_id', 'value')
)

def update_output(value):
    if value in id_list:
        plot_bgcolor = "#def"
        quadrant_colors = [plot_bgcolor,"#2bad4e", "#85e043", "#eff229", "#f2a529","#f25829"] 
        quadrant_text = ["", "<b>Very high</b>", "<b>High</b>", "<b>Medium</b>", "<b>Low</b>", "<b>Very low</b>"]
        n_quadrants = len(quadrant_colors) - 1

        current_value = prediction[value]["y_proba_0"]
        min_value = 0
        max_value = 1
        hand_length = np.sqrt(2) / 4
        hand_angle = np.pi * (1 - (max(min_value, min(max_value, current_value)) - min_value) / (max_value - min_value))

        fig = go.Figure(
            data=[
                go.Pie(
                    values=[0.5] + (np.ones(n_quadrants) / 2 / n_quadrants).tolist(),
                    rotation=90,
                    hole=0.5,
                    marker_colors=quadrant_colors,
                    text=quadrant_text,
                    textinfo="text",
                    hoverinfo="skip",
                    ),
                ],
            layout=go.Layout(
                showlegend=False,
                margin=dict(b=0,t=10,l=10,r=10),
                width=450,
                height=450,
                paper_bgcolor=plot_bgcolor,
                annotations=[
                        go.layout.Annotation(
                        text=f"<b>Score:</b><br>{round(current_value,5)*100} %",
                        x=0.5, xanchor="center", xref="paper",
                        y=0.25, yanchor="bottom", yref="paper",
                        showarrow=False,
                    )
                ],
                shapes=[
                    go.layout.Shape(
                        type="circle",
                        x0=0.48, x1=0.52,
                        y0=0.48, y1=0.52,
                        fillcolor="#333",
                        line_color="#333",
                    ),
                    go.layout.Shape(
                        type="line",
                        x0=0.5, x1=0.5 + hand_length * np.cos(hand_angle),
                        y0=0.5, y1=0.5 + hand_length * np.sin(hand_angle),
                        line=dict(color="#333", width=4)
                    )
                ]
            )
        )
        # fig = go.Figure(go.Indicator(
        #     mode = "delta+gauge+number",
        #     gauge = {'axis': {'range': [0, 1]},
        #              'bar': {'color': '#333'},
        #             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.5}},
        #     delta={ "reference": 0.5},
        #     value = pred,
        #     title = {'text': "prediction"},
        #     domain = {'x': [0, 1], 'y': [0, 1]}
        # ))
        # fig.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})
        return f'Probabilité que le client {value} soit solvable :', html.Div([dcc.Graph(figure=fig)])
    return

if __name__ == '__main__':
    # dash_app.run_server(debug=True)
    app.run_server(debug=True, use_reloader=False)