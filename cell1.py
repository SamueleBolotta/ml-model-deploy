# =============================================================================
# WORKSHOP: DEPLOY DI MODELLI AI
# Serializzazione con pickle e joblib
# =============================================================================

import pickle
import joblib
import numpy as np
import pandas as pd
import time
import os
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

def main():
    """
    Funzione principale del workshop
    """
    # Creazione cartella per i modelli
    models_dir = Path("saved_models")
    models_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("WORKSHOP: DEPLOY DI MODELLI AI - VERSIONE LOCALE")
    print("Serializzazione con pickle e joblib")
    print("="*80)

    # Creazione dataset sintetico per dimostrazioni
    print("\n1. CREAZIONE DATASET DI ESEMPIO")
    print("-" * 50)

    # Dataset con 10.000 campioni, 20 features, problema binario
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=1,
        random_state=42
    )

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Dataset creato: {X.shape[0]} campioni, {X.shape[1]} features")
    print(f"Training set: {X_train.shape[0]} campioni")
    print(f"Test set: {X_test.shape[0]} campioni")
    print(f"Distribuzione classi: {np.bincount(y)}")

    # =============================================================================
    # PARTE 1: SERIALIZZAZIONE CON PICKLE
    # =============================================================================

    print("\n\n2. SERIALIZZAZIONE CON PICKLE")
    print("-" * 50)

    # Training di un modello semplice
    print("Training modello LogisticRegression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_accuracy = lr_model.score(X_test, y_test)
    print(f"Accuracy: {lr_accuracy:.4f}")

    # Salvataggio con pickle
    pickle_path = models_dir / 'model_pickle.pkl'
    print(f"\nSalvataggio con pickle in: {pickle_path}")
    start_time = time.time()
    with open(pickle_path, 'wb') as f:
        pickle.dump(lr_model, f)
    pickle_save_time = time.time() - start_time
    pickle_size = pickle_path.stat().st_size
    print(f"Tempo salvataggio pickle: {pickle_save_time:.4f} secondi")
    print(f"Dimensione file pickle: {pickle_size / 1024:.2f} KB")

    # Caricamento con pickle
    print("\nCaricamento con pickle...")
    start_time = time.time()
    with open(pickle_path, 'rb') as f:
        lr_loaded = pickle.load(f)
    pickle_load_time = time.time() - start_time
    print(f"Tempo caricamento pickle: {pickle_load_time:.4f} secondi")

    # Sostituire con:
    # TODO: Implementare verifica che il modello caricato funzioni correttamente
    # Calcolare accuracy e confrontare con il modello originale
    raise NotImplementedError("Implementare verifica funzionalità modello caricato")

    # =============================================================================
    # PARTE 2: SERIALIZZAZIONE CON JOBLIB
    # =============================================================================

    print("\n\n3. SERIALIZZAZIONE CON JOBLIB")
    print("-" * 50)

    # Training di un modello più complesso
    print("Training modello RandomForest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_accuracy = rf_model.score(X_test, y_test)
    print(f"Accuracy RandomForest: {rf_accuracy:.4f}")

    # Salvataggio con joblib (senza compressione)
    joblib_path = models_dir / 'model_joblib.pkl'
    print(f"\nSalvataggio con joblib in: {joblib_path}")
    start_time = time.time()
    joblib.dump(rf_model, joblib_path)
    joblib_save_time = time.time() - start_time
    joblib_size = joblib_path.stat().st_size
    print(f"Tempo salvataggio joblib: {joblib_save_time:.4f} secondi")
    print(f"Dimensione file joblib: {joblib_size / 1024:.2f} KB")

    # Salvataggio con compressione
    joblib_comp_path = models_dir / 'model_joblib_compressed.pkl'
    print(f"\nSalvataggio con joblib (compressione livello 3) in: {joblib_comp_path}")
    start_time = time.time()
    joblib.dump(rf_model, joblib_comp_path, compress=3)
    # TODO: Misurare le performance della compressione e calcolare la riduzione percentuale
    raise NotImplementedError("Implementare misurazione performance compressione")
    # Caricamento con joblib
    print("\nCaricamento con joblib...")
    start_time = time.time()
    rf_loaded = joblib.load(joblib_comp_path)
    joblib_load_time = time.time() - start_time
    print(f"Tempo caricamento joblib: {joblib_load_time:.4f} secondi")

    # Verifica funzionalità
    loaded_rf_accuracy = rf_loaded.score(X_test, y_test)
    print(f"Accuracy modello caricato: {loaded_rf_accuracy:.4f}")
    print(f"Modelli identici: {rf_accuracy == loaded_rf_accuracy}")

    # =============================================================================
    # PARTE 3: CONFRONTO PERFORMANCE PICKLE VS JOBLIB
    # =============================================================================

    print("\n\n4. CONFRONTO PERFORMANCE CON GRANDI ARRAY")
    print("-" * 50)

    # Creazione di un grande array NumPy per test
    large_array = np.random.randn(1000, 1000).astype(np.float32)
    print(f"Array di test: {large_array.shape}, {large_array.dtype}")
    print(f"Dimensione in memoria: {large_array.nbytes / 1024 / 1024:.2f} MB")

    # Test pickle con array grande
    large_pickle_path = models_dir / 'large_array_pickle.pkl'
    print(f"\nTest pickle con array grande in: {large_pickle_path}")
    start_time = time.time()
    with open(large_pickle_path, 'wb') as f:
        pickle.dump(large_array, f)
    pickle_large_save_time = time.time() - start_time
    pickle_large_size = large_pickle_path.stat().st_size

    start_time = time.time()
    with open(large_pickle_path, 'rb') as f:
        array_pickle_loaded = pickle.load(f)
    pickle_large_load_time = time.time() - start_time

    # TODO: Confrontare le performance tra pickle e joblib per grandi array
    # Analizzare tempi di salvataggio/caricamento e dimensioni dei file
    raise NotImplementedError("Implementare confronto performance pickle vs joblib")
    start_time = time.time()
    joblib.dump(large_array, large_joblib_path)
    joblib_large_save_time = time.time() - start_time
    joblib_large_size = large_joblib_path.stat().st_size

    start_time = time.time()
    array_joblib_loaded = joblib.load(large_joblib_path)
    joblib_large_load_time = time.time() - start_time

    print(f"Joblib - Salvataggio: {joblib_large_save_time:.4f}s, Caricamento: {joblib_large_load_time:.4f}s")
    print(f"Joblib - Dimensione: {joblib_large_size / 1024 / 1024:.2f} MB")

    # Test joblib con compressione
    large_joblib_comp_path = models_dir / 'large_array_joblib_comp.pkl'
    start_time = time.time()
    joblib.dump(large_array, large_joblib_comp_path, compress=3)
    joblib_comp_save_time = time.time() - start_time
    joblib_comp_size = large_joblib_comp_path.stat().st_size

    print(f"Joblib compresso - Salvataggio: {joblib_comp_save_time:.4f}s")
    print(f"Joblib compresso - Dimensione: {joblib_comp_size / 1024 / 1024:.2f} MB")

    # Verifica integrità
    arrays_equal = np.array_equal(large_array, array_joblib_loaded)
    print(f"\nArray identici dopo serializzazione: {arrays_equal}")

    # =============================================================================
    # PARTE 4: PIPELINE COMPLESSE E METADATA
    # =============================================================================

    print("\n\n5. SERIALIZZAZIONE DI PIPELINE COMPLESSE")
    print("-" * 50)

    # Creazione pipeline complessa
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
    ])

    print("Training pipeline complessa...")
    pipeline.fit(X_train, y_train)
    pipeline_accuracy = pipeline.score(X_test, y_test)
    print(f"Accuracy pipeline: {pipeline_accuracy:.4f}")

    # TODO: Creare un dizionario completo di metadata per tracciare il modello
    # Includere versioni software, parametri, metriche e timestamp
    raise NotImplementedError("Implementare creazione metadata completi del modello")
        'hyperparameters': {
            'n_estimators': 50,
            'random_state': 42,
            'scaler': 'StandardScaler'
        }
    }

    # Salvataggio pipeline + metadata
    model_package = {
        'model': pipeline,
        'metadata': model_metadata,
        'feature_names': [f'feature_{i}' for i in range(X_train.shape[1])]
    }

    package_path = models_dir / 'complete_model_package.pkl'
    print(f"\nSalvataggio package completo in: {package_path}")
    joblib.dump(model_package, package_path, compress=3)
    package_size = package_path.stat().st_size
    print(f"Dimensione package completo: {package_size / 1024:.2f} KB")

    # Caricamento e verifica package
    print("\nCaricamento package completo...")
    loaded_package = joblib.load(package_path)
    loaded_model = loaded_package['model']
    loaded_metadata = loaded_package['metadata']

    print("\nMetadata del modello caricato:")
    for key, value in loaded_metadata.items():
        if key != 'hyperparameters':
            print(f"  {key}: {value}")

    print("\nIperparametri:")
    for key, value in loaded_metadata['hyperparameters'].items():
        print(f"  {key}: {value}")

    # Test predizione
    sample_prediction = loaded_model.predict(X_test[:5])
    sample_proba = loaded_model.predict_proba(X_test[:5])
    print(f"\nPredizioni campione: {sample_prediction}")
    print(f"Probabilità campione: {sample_proba[:, 1]}")

    # =============================================================================
    # PARTE 5: GESTIONE ERRORI E COMPATIBILITÀ
    # =============================================================================

    print("\n\n6. GESTIONE ERRORI E BEST PRACTICES")
    print("-" * 50)

    def safe_model_save(model, filepath, method='joblib', compress=3):
        """
        Salva un modello con gestione degli errori
        """
        try:
            filepath = Path(filepath)
            
            if method == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
            elif method == 'joblib':
                joblib.dump(model, filepath, compress=compress)

            # Verifica che il file sia stato creato
            if filepath.exists():
                size = filepath.stat().st_size
                print(f"Modello salvato con successo: {filepath}")
                print(f"Dimensione: {size / 1024:.2f} KB")
                return True
            else:
                print(f"Errore: file {filepath} non creato")
                return False

        except Exception as e:
            print(f"Errore durante il salvataggio: {str(e)}")
            return False

    def safe_model_load(filepath, method='joblib'):
        """
        Carica un modello con gestione degli errori
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                raise FileNotFoundError(f"File {filepath} non trovato")

            if method == 'pickle':
                with open(filepath, 'rb') as f:
                    model = pickle.load(f)
            elif method == 'joblib':
                model = joblib.load(filepath)

            print(f"Modello caricato con successo da: {filepath}")
            return model, True

        except Exception as e:
            print(f"Errore durante il caricamento: {str(e)}")
            return None, False

    # Test funzioni sicure
    safe_model_path = models_dir / 'safe_model.pkl'
    print(f"Test salvataggio sicuro in: {safe_model_path}")
    success = safe_model_save(pipeline, safe_model_path, method='joblib', compress=3)

    if success:
        print("\nTest caricamento sicuro...")
        loaded_model, load_success = safe_model_load(safe_model_path, method='joblib')

        # TODO: Verificare compatibilità delle versioni software nel modello caricato
        # Confrontare metadata salvati con ambiente corrente
        raise NotImplementedError("Implementare verifica compatibilità versioni")

    # Test caricamento file inesistente
    print("\nTest caricamento file inesistente...")
    fake_model, fake_success = safe_model_load(models_dir / 'nonexistent_model.pkl')

    # =============================================================================
    # PARTE 6: RIEPILOGO FILES CREATI
    # =============================================================================

    print("\n\n7. RIEPILOGO FILES CREATI")
    print("-" * 50)
    
    print(f"\nCartella modelli: {models_dir.absolute()}")
    print("Files creati:")
    
    for file_path in sorted(models_dir.glob("*.pkl")):
        size_kb = file_path.stat().st_size / 1024
        print(f"  - {file_path.name}: {size_kb:.2f} KB")
    
    print(f"\nTotale files: {len(list(models_dir.glob('*.pkl')))}")
    
    # Calcolo spazio totale occupato
    total_size = sum(f.stat().st_size for f in models_dir.glob("*.pkl"))
    print(f"Spazio totale occupato: {total_size / 1024:.2f} KB")
    
    print("\n" + "="*80)
    print("WORKSHOP COMPLETATO!")
    print("Tutti i modelli sono stati salvati nella cartella 'saved_models'")
    print("="*80)

if __name__ == "__main__":
    main()
