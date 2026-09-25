import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Cargar y preparar los datos
url = 'https://raw.githubusercontent.com/krishnaik06/Simple-Linear-Regression/master/Salary_Data.csv'
df = pd.read_csv(url)

X = df[['YearsExperience']]
y = df['Salary']

# 2. Entrenar el modelo
modelo = LinearRegression()
modelo.fit(X, y)
df['Prediccion'] = modelo.predict(X)

# Métricas
r2 = r2_score(y, df['Prediccion'])
mae = mean_absolute_error(y, df['Prediccion'])

# 3. Inicializar Dash
app = dash.Dash(__name__)
server = app.server

# 4. Diseño del Dashboard
app.layout = html.Div(style={'fontFamily': 'Arial', 'padding': '20px'}, children=[
    html.H1("Dashboard: Predicción de Salarios vs Experiencia", style={'textAlign': 'center'}),
    
    html.Div([
        html.P("Este tablero resuelve el problema de la incertidumbre salarial, prediciendo el salario justo según años de experiencia.")
    ]),
    
    html.Div(style={'display': 'flex', 'justifyContent': 'space-around', 'backgroundColor': '#f2f2f2', 'padding': '20px'}, children=[
        html.Div([html.H3("Precisión (R²)"), html.H2(f"{r2*100:.2f}%", style={'color': 'green'})]),
        html.Div([html.H3("Margen de Error (MAE)"), html.H2(f"${mae:,.2f}", style={'color': 'red'})]),
        html.Div([html.H3("Aumento Anual"), html.H2(f"${modelo.coef_[0]:,.2f}", style={'color': 'blue'})])
    ]),
    
    dcc.Graph(id='grafico-regresion'),
    
    html.Div(style={'margin': '20px', 'padding': '20px', 'border': '2px solid black'}, children=[
        html.H3("Calculadora Interactiva de Salario"),
        html.Label("Mueve el deslizador para elegir los Años de Experiencia:"),
        dcc.Slider(id='slider-experiencia', min=0, max=20, step=0.5, value=5, 
                   marks={i: str(i) for i in range(0, 21, 2)}),
        html.H2(id='resultado-prediccion', style={'textAlign': 'center', 'color': 'green', 'marginTop': '20px'})
    ])
])

# 5. Interactividad
@app.callback(
    [Output('grafico-regresion', 'figure'),
     Output('resultado-prediccion', 'children')],
    [Input('slider-experiencia', 'value')]
)
def actualizar(experiencia):
    fig = px.scatter(df, x='YearsExperience', y='Salary', title='Línea de Tendencia Real')
    fig.add_trace(go.Scatter(x=df['YearsExperience'], y=df['Prediccion'], mode='lines', name='Modelo Predictivo'))
    
    salario_predicho = modelo.predict([[experiencia]])[0]
    fig.add_trace(go.Scatter(x=[experiencia], y=[salario_predicho], mode='markers', name='Tu Consulta', marker=dict(color='green', size=15)))
    
    texto = f"El salario sugerido para {experiencia} años de experiencia es: ${salario_predicho:,.2f}"
    return fig, texto

# 6. Ejecutar
if __name__ == '__main__':
    app.run (debug=True)