"""
Pruebas unitarias para el módulo preprocessing.py.
"""

import pytest
import math
from src import preprocessing as pp

# --- 1. Fixture Requerida ---

@pytest.fixture
def sample_numbers() -> list[float]:
    """
    Fixture que provee una lista de números simple para varios tests.
    """
    return [1.0, 2.0, 3.0, 4.0, 5.0]


# --- 2. Pruebas para Funciones de Limpieza ---

@pytest.mark.parametrize("input_list, expected_list", [
    ([1, None, 2, ""], [1, 2]),                  # Caso básico
    ([1, 2, 3], [1, 2, 3]),                      # Sin nulos
    ([], []),                                    # Lista vacía
    ([None, "", math.nan], []),                  # Solo nulos
    (["hola", None, 1.5], ["hola", 1.5])         # Mixto
])
def test_remove_missing(input_list, expected_list):
    """Prueba la función remove_missing con varios casos."""
    assert pp.remove_missing(input_list) == expected_list

@pytest.mark.parametrize("input_list, fill_value, expected_list", [
    ([1, None, 3], 0, [1, 0, 3]),                # Relleno con default (0)
    ([1, None, 3], 99, [1, 99, 3]),              # Relleno con int
    ([1, "", 3], "NA", [1, "NA", 3]),            # Relleno con string
    ([1, 2, 3], 0, [1, 2, 3]),                   # Sin nulos
    ([math.nan, 2, 3], 0, [0, 2, 3])             # Relleno de 'nan'
])
def test_fill_missing(input_list, fill_value, expected_list):
    """Prueba la función fill_missing con diferentes valores de relleno."""
    assert pp.fill_missing(input_list, fill_value=fill_value) == expected_list

def test_remove_duplicates():
    """Prueba la eliminación de duplicados conservando el orden."""
    assert pp.remove_duplicates([1, 2, 1, 3, 2]) == [1, 2, 3]
    assert pp.remove_duplicates(["a", "b", "a"]) == ["a", "b"]
    assert pp.remove_duplicates(["c", "a", "b", "a"]) == ["c", "a", "b"]
    assert pp.remove_duplicates([]) == []


# --- 3. Pruebas para Funciones Numéricas ---

@pytest.mark.parametrize("new_min, new_max, expected", [
    (0.0, 1.0, [0.0, 0.25, 0.5, 0.75, 1.0]),     # Default (0 a 1)
    (-1.0, 1.0, [-1.0, -0.5, 0.0, 0.5, 1.0])    # -1 a 1
])
def test_normalize_min_max(sample_numbers, new_min, new_max, expected):
    """Prueba la normalización min-max usando la fixture."""
    result = pp.normalize_min_max(sample_numbers, new_min, new_max)
    assert result == pytest.approx(expected)

def test_normalize_min_max_single_value():
    """Prueba el caso borde de normalizar una lista con un solo valor."""
    assert pp.normalize_min_max([5], 0.0, 1.0) == [0.0]

def test_standardize_z_score(sample_numbers):
    """Prueba la estandarización z-score usando la fixture."""
    result = pp.standardize_z_score(sample_numbers)
    # mean=3, std_dev=sqrt(2) approx 1.4142
    expected = [
        (1-3)/math.sqrt(2),
        (2-3)/math.sqrt(2),
        (3-3)/math.sqrt(2),
        (4-3)/math.sqrt(2),
        (5-3)/math.sqrt(2)
    ]
    assert result == pytest.approx(expected)

def test_standardize_z_score_single_value():
    """Prueba el caso borde de estandarizar una lista con valores idénticos."""
    assert pp.standardize_z_score([5, 5, 5]) == [0.0, 0.0, 0.0]

def test_clip_values():
    """Prueba el recorte de valores a un rango."""
    values = [0, 5, 10, 15]
    min_val, max_val = 2, 8
    expected = [2, 5, 8, 8]
    assert pp.clip_values(values, min_val, max_val) == expected

def test_convert_to_int():
    """Prueba la conversión de strings a enteros."""
    values = ["1", "2.5", "3.9", "hola", "4.0", "-1.8"]
    expected = [1, 2, 3, 4, -1]
    assert pp.convert_to_int(values) == expected

def test_transform_log():
    """Prueba la transformación logarítmica (solo positivos)."""
    values = [1, 10, -5, 0, math.e]
    expected = [math.log(1), math.log(10), math.log(math.e)] # 0, 2.302..., 1
    assert pp.transform_log(values) == pytest.approx(expected)


# --- 4. Pruebas para Funciones de Texto ---

def test_tokenize():
    """Prueba la tokenización de texto."""
    text = "Hola, esto es 1 prueba!"
    expected = ["hola", "esto", "es", "1", "prueba"]
    assert pp.tokenize(text) == expected

def test_select_alphanumeric_spaces():
    """Prueba la eliminación de puntuación."""
    text = "Hola!! ¿Qué tal?"
    expected = "Hola Qu tal" # 'é' y '¿' se eliminan
    assert pp.select_alphanumeric_spaces(text) == expected

def test_remove_stopwords():
    """Prueba la eliminación de stop-words (ignorando mayúsculas)."""
    text = "El Perro de mi amigo"
    stopwords = ["el", "de", "mi"]
    expected = "perro amigo"
    assert pp.remove_stopwords(text, stopwords) == expected


# --- 5. Pruebas para Funciones de Estructura ---

def test_flatten_list():
    """Prueba el aplanado de una lista de listas."""
    nested = [[1, 2], [3, 4], [], [5]]
    expected = [1, 2, 3, 4, 5]
    assert pp.flatten_list(nested) == expected

def test_shuffle_list():
    """
    Prueba que el shuffle sea reproducible con semilla
    y que no modifique la lista original.
    """
    original_list = [1, 2, 3, 4, 5, 6, 7]
    original_copy = original_list[:] # Copia explícita
    
    shuffled_1 = pp.shuffle_list(original_list, seed=42)
    shuffled_2 = pp.shuffle_list(original_list, seed=42)
    
    # Comprueba que la lista original no ha sido modificada
    assert original_list == original_copy
    
    # Comprueba que la semilla produce resultados idénticos
    assert shuffled_1 == shuffled_2
    
    # Comprueba que la lista ha sido, de hecho, mezclada
    assert shuffled_1 != original_list
