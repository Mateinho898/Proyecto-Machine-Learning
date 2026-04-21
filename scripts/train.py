import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import joblib

def train():
    # Cargar datos limpios
    df = pd.read_csv('data_cleaned.csv')
    
    # Filtrar clases con muy pocos datos para mejorar el entrenamiento
    top_classes = df['grd_target'].value_counts()
    valid_classes = top_classes[top_classes > 10].index
    df = df[df['grd_target'].isin(valid_classes)]

    X = df.drop('grd_target', axis=1)
    y = df['grd_target']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Entrenando Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Evaluación
    preds = model.predict(X_test)
    score = f1_score(y_test, preds, average='weighted')
    
    print(f"\nWeighted F1-Score: {score:.4f}")
    
    # Guardar el modelo
    joblib.dump(model, 'grd_model.pkl')
    print("Modelo guardado como 'grd_model.pkl'")

if __name__ == "__main__":
    train()
