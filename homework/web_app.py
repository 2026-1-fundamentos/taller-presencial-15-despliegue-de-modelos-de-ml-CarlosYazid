import logging
import os

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import requests

# Configuración de Logging para el Cliente
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("House Price Web App")

# URL del endpoint de la API
API_URL = os.getenv('API_URL') or "http://127.0.0.1:8000/predict"

# Inicializar la app Dash usando un tema Bootstrap
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.FLATLY],
    title="House Price Predictor"
)

# Diseño de la Interfaz de Usuario (Layout)
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Predicción de Precios de Viviendas", className="text-center my-4 text-primary"), width=12)
    ]),
    
    dbc.Row([
        # Formulario de Características (Columna Izquierda)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Características de la Propiedad", className="mb-0")),
                dbc.CardBody([
                    # Habitaciones y Baños en una fila
                    dbc.Row([
                        dbc.Col([
                            html.Label("Habitaciones:"),
                            dcc.Input(id="input-bedrooms", type="number", value=3, step=1, min=0, className="form-control mb-3")
                        ], width=6),
                        dbc.Col([
                            html.Label("Baños:"),
                            dcc.Input(id="input-bathrooms", type="number", value=2, step=1, min=0, className="form-control mb-3")
                        ], width=6),
                    ]),
                    
                    # Pies cuadrados
                    html.Label("Pies Cuadrados Habitables:"),
                    dcc.Input(id="input-sqft-living", type="number", value=1800.0, step=50, className="form-control mb-3"),
                    
                    html.Label("Pies Cuadrados Habitables 15:"),
                    dcc.Input(id="input-sqft-living15", type="number", value=1750.0, step=50, className="form-control mb-3"),
                    
                    html.Label("Pies Cuadrados sobre el suelo:"),
                    dcc.Input(id="input-sqft-above", type="number", value=1500.0, step=50, className="form-control mb-3"),
                    
                    # Pisos y Grado
                    dbc.Row([
                        dbc.Col([
                            html.Label("Pisos:"),
                            dcc.Input(id="input-floors", type="number", value=2, step=1, min=1, className="form-control mb-3")
                        ], width=6),
                        dbc.Col([
                            html.Label("Grado de diseño:"),
                            dcc.Input(id="input-grade", type="number", value=7, step=1, min=1, max=13, className="form-control mb-3")
                        ], width=6),
                    ]),
                    
                    # Latitud y Vista al mar
                    dbc.Row([
                        dbc.Col([
                            html.Label("Latitud:"),
                            dcc.Input(id="input-lat", type="number", value=47.5112, step=0.0001, className="form-control mb-3")
                        ], width=6),
                        dbc.Col([
                            html.Label("Vista al Mar:"),
                            dcc.Dropdown(
                                id="input-waterfront",
                                options=[{"label": "No", "value": 0}, {"label": "Sí", "value": 1}],
                                value=0,
                                clearable=False,
                                className="mb-3"
                            )
                        ], width=6),
                    ]),
                    
                    # Botón de envío
                    dbc.Button("Calcular Precio Estimado", id="btn-predict", color="primary", size="lg", className="w-100 mt-2")
                ])
            ], className="shadow")
        ], md=6, sm=12),
        
        # Resultados de la Inferencia (Columna Derecha)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Resultado del Modelo", className="mb-0")),
                dbc.CardBody([
                    html.Div(id="result-container", children=[
                        html.P("Rellena los datos de la izquierda y haz clic en calcular.", className="text-muted text-center py-5")
                    ])
                ])
            ], className="shadow h-100")
        ], md=6, sm=12)
    ], className="g-4")
], fluid=True, className="px-5")

# Callback para realizar la Request

@app.callback(
    Output("result-container", "children"),
    Input("btn-predict", "n_clicks"),
    State("input-bedrooms", "value"),
    State("input-bathrooms", "value"),
    State("input-sqft-living", "value"),
    State("input-sqft-living15", "value"),
    State("input-sqft-above", "value"),
    State("input-floors", "value"),
    State("input-grade", "value"),
    State("input-lat", "value"),
    State("input-waterfront", "value"),
    prevent_initial_call=True
)
def estimate_house_price(
    n_clicks,
    bedrooms,
    bathrooms,
    sqft_living,
    sqft_living15,
    sqft_above,
    floors,
    grade,
    lat,
    waterfront):
    
    # Payload que machea exactamente con el HouseFeaturesRequest de Pydantic
    payload = {
        "bedrooms": float(bedrooms),
        "bathrooms": float(bathrooms),
        "sqft_living": float(sqft_living),
        "sqft_living15": float(sqft_living15),
        "sqft_above": float(sqft_above),
        "floors": float(floors),
        "waterfront": int(waterfront),
        "grade": int(grade),
        "lat": float(lat)
    }
    
    logger.info(f"Enviando petición sincrónica a la API. Payload: {payload}")
    
    try:
        
        # Petición
        response = requests.post(API_URL, json=payload, timeout=5.0)
        
        if response.status_code == 200:
            
            data = response.json()
            predicted_price = data.get("price", 0.0)
            
            logger.info(f"Respuesta exitosa recibida: {predicted_price}")
            
            # Renderizamos una tarjeta visualmente atractiva con el precio
            return html.Div([
                html.H3("Precio Estimado", className="text-center text-success mt-4"),
                html.H1(f"${predicted_price:,.2f}", className="text-center display-4 font-weight-bold my-4 text-dark"),
                html.Hr(),
                html.P("Inferencia completada exitosamente.", className="text-muted text-center small")
            ])
        else:
            logger.error(f"La API respondió con error {response.status_code}: {response.text}")
            return dbc.Alert(f"Error del Servidor ({response.status_code}): {response.text}", color="danger", className="mt-4")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"No se pudo conectar con el servidor de la API: {str(e)}")
        return dbc.Alert(
            "Error de Conexión: Asegúrate de que el servidor de la API esté levantado y corriendo.", 
            color="warning", 
            className="mt-4"
        )

# Ejecutar Servidor Dash
if __name__ == "__main__":
    app.run(debug=True, port=8050)