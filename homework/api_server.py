import logging
import os
from contextlib import asynccontextmanager

import numpy as np
from onnxruntime import SessionOptions, InferenceSession 
from fastapi import FastAPI, Request, Depends, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv('MODEL_PATH') or 'house_price_model.onnx'

# Configuración del Logging Estándar
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()  # Envía los logs a la consola/terminal
    ]
)
logger = logging.getLogger("House Price Model API")

# Modelos de Pydantic (Request / Response)

class HouseFeaturesRequest(BaseModel):
    
    bedrooms: float = Field(..., description="Número de habitaciones")
    bathrooms: float = Field(..., description="Número de baños")
    sqft_living: float = Field(..., description="Pies cuadrados del espacio habitable")
    sqft_living15: float = Field(..., description="Pies cuadrados del espacio habitable en 2015")
    sqft_above: float = Field(..., description="Pies cuadrados del espacio sobre el suelo")
    floors: float = Field(..., description="Número de pisos")
    waterfront: int = Field(..., description="Vista al mar (0 o 1)")
    grade: int = Field(..., description="Grado de construcción y diseño")
    lat: float = Field(..., description="Latitud")

    class Config:
        json_schema_extra = {
            "example": {
                "bedrooms": 3.0,
                "bathrooms": 2.5,
                "sqft_living": 1800.0,
                "sqft_living15": 1750.0,
                "sqft_above": 1500.0,
                "floors": 2.0,
                "waterfront": 0,
                "grade": 7,
                "lat": 47.5112
            }
        }

class PriceResponse(BaseModel):
    price: float = Field(..., description="Precio estimado de la propiedad")


# Ciclo de Vida de la App (Lifespan)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Maneja la inicialización y el apagado seguro de la aplicación.
    Carga el modelo ONNX configurando un mínimo de 2 hilos.
    """
    
    logger.info("Iniciando el servidor FastAPI...")
    
    try:
        
        logger.info(f"Configurando SessionOptions para ONNX Runtime")
        
        opts = SessionOptions()
        # Se configuran 2 hilos para paraleleismo interno e inter-operador
        opts.intra_op_num_threads = int(os.getenv('NUM_INFERENCE_THREADS') or 2)
        opts.inter_op_num_threads = int(os.getenv('NUM_INFERENCE_THREADS') or 2)
        
        logger.info(f"Cargando el modelo ONNX desde: {MODEL_PATH}")
        
        ort_session = InferenceSession(MODEL_PATH, sess_options=opts)
        
        # Guardamos nombres de entradas y salidas para evitar recalcularlos en cada request
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        logger.info(f"Modelo cargado con éxito. Input: '{input_name}', Output esperado: '{output_name}'")
        
        # El diccionario yield expone estos recursos a través de request.state
        yield {
            "onnx_session": ort_session,
            "input_name": input_name,
            "output_name": output_name
        }
        
    except Exception as e:
        logger.critical(f"Error crítico al inicializar el modelo ONNX: {str(e)}", exc_info=True)
        raise e
        
    finally:
        logger.info("Apagando el servidor. Limpiando recursos de la sesión ONNX.")

# Inicializamos FastAPI con el ciclo de vida definido

app = FastAPI(
    title="House Model Inference API",
    description="API para predicción de precios optimizada con multi-threading y logs.",
    version="1.1.0",
    lifespan=lifespan
)

# Inyección de Dependencias
def get_onnx_resources(request: Request):
    """Retorna los recursos cargados en el estado de la aplicación."""
    return {
        "session": request.state.onnx_session,
        "input_name": request.state.input_name,
        "output_name": request.state.output_name
    }

# Endpoint Sincrónico de Inferencia

@app.post("/predict", response_model=PriceResponse)
def predict_price(
    request_data: HouseFeaturesRequest, 
    resources: dict = Depends(get_onnx_resources)
    ):
    
    """
    Endpoint sincrónico que recibe datos de la casa, aplica inferencia bajo ONNX
    utilizando hilos paralelos y devuelve el precio.
    """
    logger.info("Recibida una nueva solicitud de predicción.")
    
    try:
        
        # Convertir a estructura requerida
        input_array = np.array(
            [list(request_data.model_dump().values())],
            dtype=np.float32
            )
        
        # Ejecutar la inferencia usando los recursos cacheados
        session = resources["session"]
        input_name = resources["input_name"]
        output_name = resources["output_name"]
        
        logger.debug(f"Ejecutando inferencia ONNX para el input shape: {input_array.shape}")
        
        onnx_outputs = session.run([output_name], {input_name: input_array})
        
        predicted_price = float(onnx_outputs[0][0][0])
        
        logger.info(f"Predicción exitosa. Precio calculado: {predicted_price}")
        
        return PriceResponse(price=predicted_price)

    except Exception as e:
        logger.error(f"Error procesando la inferencia: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Ocurrió un error interno al procesar la predicción en el modelo."
        )