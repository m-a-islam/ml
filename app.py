from fastapi import FastAPI
import uvicorn

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel

#create fastapip
app = FastAPI()


class request_model(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


iris = load_iris()
X = iris.data
y = iris.target

model = GaussianNB()
model.fit(X, y)


@app.post("/predict")
def predict(data: request_model):
    test_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = model.predict(test_data)[0]
    return {"class": iris.target_names[prediction]}


@app.get("/")
def read_root():
    return {"Hello": "World"}