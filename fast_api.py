from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

model = load('penguin_model')

app = FastAPI()

class my_input(BaseModel):
  
    culmen_length_mm: float
    culmen_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: int


@app.get('/')
async def welcome():
    return {'message':'Hii..there'}

@app.post('/predict/')
async def main(input: my_input):
    data = input.dict()
    data_ =  [[data['culmen_length_mm'], data['culmen_depth_mm'], data['flipper_length_mm'], data['body_mass_g'], data['sex']]]

    species = model.predict(data_)[0]

    probability = model.predict_proba(data_).max()
    if species == 0:
        species_name = 'Adelie'
    elif species == 1:
        species_name = 'Chinstrap'
    else:
        species_name = 'Gentoo'

    return {
'prediction': species_name,
'probability': float(probability)
}