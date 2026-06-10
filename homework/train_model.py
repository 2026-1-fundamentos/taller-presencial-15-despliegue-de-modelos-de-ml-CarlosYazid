import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

df = pd.read_csv("../files/input/house_data.csv")

features = df[
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_living15",
        "sqft_above",
        "floors",
        "waterfront",
        "grade",
        "lat"
    ]
]

target = df[["price"]]

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, _, y_train, _ = train_test_split(features, target, test_size=0.2, random_state=42)

estimator = LinearRegression()
estimator.fit(X_train, y_train)

pipeline = Pipeline([('scaler', scaler), ('estimator', estimator)])

onnx_model = convert_sklearn(
    pipeline,
    initial_types=[
        ("features", FloatTensorType([None, 9]))
    ],
    final_types=[
        ("price", FloatTensorType([None, 1]))
    ],
    target_opset=17
)

with open("house_price_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())