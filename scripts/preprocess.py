import pandas as pd
import numpy as np

def clean_data(input_path, output_path):
    print(f"Cargando datos desde {input_path}...")
    # Leer con separador ;
    df = pd.read_csv(input_path, sep=';')
    
    # 1. Reemplazar guiones por NaN
    df = df.replace('-', np.nan)
    
    # 2. Función para extraer solo el código (CIE-10 o Procedimiento)
    def extract_code(val):
        if pd.isna(val): return val
        return str(val).split(' - ')[0].strip()

    diag_cols = [c for c in df.columns if 'Diag' in c]
    proc_cols = [c for c in df.columns if 'Proced' in c]
    
    for col in diag_cols + proc_cols:
        df[col] = df[col].apply(extract_code)
    
    # 3. Feature Engineering: Contar diagnósticos y procedimientos
    df['n_diagnosticos'] = df[diag_cols].notna().sum(axis=1)
    df['n_procedimientos'] = df[proc_cols].notna().sum(axis=1)
    
    # 4. Codificación de Sexo
    df['sexo_bin'] = df['Sexo (Desc)'].map({'Hombre': 0, 'Mujer': 1})
    
    # 5. Limpieza de GRD (Target)
    df['grd_target'] = df['GRD'].str.split(' - ', n=1, expand=True)[0]
    
    # Seleccionar columnas finales para la matriz
    final_df = df[['Edad en años', 'sexo_bin', 'n_diagnosticos', 'n_procedimientos', 'grd_target']]
    
    # Guardar versión limpia
    final_df.to_csv(output_path, index=False)
    print(f"Archivo procesado guardado en {output_path}")

if __name__ == "__main__":
    clean_data('dataset_elpino.csv', 'data_cleaned.csv')