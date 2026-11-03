import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

datos=pd.read_parquet("../data/processed/spain-used-cars-clean.parquet") #leemos el parquet

columnas_numericas = [
    "reg_year",
    "mileage_km",
    "power_hp",
]

columnas_categoricas = [
    "make",
    "model",
    "fuel",
    "transmission",
    "body_type",
]

columna_objetivo = "price_eur"

x=datos[columnas_numericas+columnas_categoricas]
y=datos[columna_objetivo]

x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(
    x, 
    y,
    test_size=0.2,
    random_state=42,
)

preprocesador=ColumnTransformer(
    transformers=[(
        "categorias",
        OneHotEncoder(handle_unknown="ignore"),
        columnas_categoricas,
    )],
    remainder="passthrough", #las columnas que no se transforman, se dejan tal cual, sin cambios.
)

modelo=Pipeline(# creamos el modelo que primero preprocesa los datos y luego aplica la regresión
    [
        ("preprocesador", preprocesador),
        ("regresor", RandomForestRegressor(
            n_estimators=50,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )),
    ]
)

modelo.fit(x_entrenamiento, y_entrenamiento) #entrenamos el modelo con los datos de entrenamiento

# Predicciones sobre coches que el modelo nunca vio
predicciones = modelo.predict(x_prueba)

comparacion=pd.DataFrame({
    "Real": y_prueba.to_numpy(),
    "Predicción": predicciones
})

# Predicciones sobre coches que ya vio
predicciones_entrenamiento = modelo.predict( #Le pedimos al modelo que prediga los precios de los coches con los que fue entrenado.
    x_entrenamiento
)

mae_entrenamiento = mean_absolute_error( #Así obtenemos cuánto se equivoca, de media, con los datos que ya conoce.
    y_entrenamiento,
    predicciones_entrenamiento,
)

r2_entrenamiento = r2_score( #Así obtenemos el R², que nos dice qué porcentaje de la variabilidad de los precios de los coches es capaz de explicar el modelo con los datos que ya conoce.
    y_entrenamiento,
    predicciones_entrenamiento,
)


print("Prueba:")
mae=mean_absolute_error(y_prueba, predicciones)
mse=mean_squared_error(y_prueba, predicciones)
rmse=np.sqrt(mse)
r2=r2_score(y_prueba, predicciones)

print(f"MAE entrenamiento: {mae_entrenamiento:,.2f} €")
print(f"MAE prueba:        {mae:,.2f} €")
print(f"R² entrenamiento: {r2_entrenamiento:.4f}")
print(f"R² prueba:        {r2:.4f}")

print(comparacion.head(10).round(2))
