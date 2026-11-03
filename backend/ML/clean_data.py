from pathlib import Path #sirve para manejar rutas de archivos y directorios de manera más conveniente y legible.
import pandas as pd


input_file = Path("data/raw/european-used-cars.parquet")
output_file = Path("data/processed/spain-used-cars-clean.parquet")


df = pd.read_parquet(input_file) #lee el archivo parquet y lo carga en un DataFrame de pandas.

print(f"Registros europeos: {len(df):,}")

df_spain = df[df["seller_country"] == "ES"].copy() #filtra el DataFrame original para obtener solo los registros donde la columna "seller_country" sea igual a "ES" (España) y crea una copia de ese subconjunto.

print(f"Registros españoles: {len(df_spain):,}")




precio_valido = df_spain["price_eur"].between( # con esto hacemos un filtro para quedarnos con los precios que estén entre 3.000 y 250.000 euros.
    3_000,
    250_000,
)

df_precio_limpio = df_spain[precio_valido].copy()

print(f"Antes de limpiar precios: {len(df_spain):,}")
print(f"Después de limpiar precios: {len(df_precio_limpio):,}")
print(
    "Precios excluidos:",
    f"{len(df_spain) - len(df_precio_limpio):,}",
)


kilometraje_valido = df_precio_limpio["mileage_km"].between(
    0,
    500_000,
)




df_kilometraje_limpio = df_precio_limpio[
    kilometraje_valido
].copy()

print(
    f"Antes de limpiar kilometraje: "
    f"{len(df_precio_limpio):,}"
)

print(
    f"Después de limpiar kilometraje: "
    f"{len(df_kilometraje_limpio):,}"
)

print(
    "Kilometrajes excluidos:",
    f"{len(df_precio_limpio) - len(df_kilometraje_limpio):,}",
)




potencia_valida = df_kilometraje_limpio["power_hp"].between(
    30,
    700,
)

df_potencia_limpia = df_kilometraje_limpio[
    potencia_valida
].copy()

print(
    f"Antes de limpiar potencia: "
    f"{len(df_kilometraje_limpio):,}"
)

print(
    f"Después de limpiar potencia: "
    f"{len(df_potencia_limpia):,}"
)

print(
    "Potencias excluidas:",
    f"{len(df_kilometraje_limpio) - len(df_potencia_limpia):,}",
)



anio_valido = df_potencia_limpia["reg_year"].between(
    2016,
    2026,
)

df_limpio = df_potencia_limpia[
    anio_valido
].copy()

print(
    "Años excluidos:",
    f"{len(df_potencia_limpia) - len(df_limpio):,}",
)




filas_antes = len(df_limpio)

df_limpio = df_limpio.drop_duplicates(
    subset=["source_site", "id"],
    keep="last",
)

print(
    "Duplicados eliminados:",
    f"{filas_antes - len(df_limpio):,}",
)



output_file.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df_limpio.to_parquet(
    output_file,
    index=False,
)

print(f"Registros finales: {len(df_limpio):,}")
print(f"Archivo guardado en: {output_file}")