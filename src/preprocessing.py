"""
Módulo de pre-procesamiento de datos.

Contiene 13 funciones para la limpieza y transformación de datos,
cubriendo valores faltantes, duplicados, normalización,
transformaciones de texto y de estructuras.
"""

import math
import random
import re
from typing import List, Any, Optional, Union


# --- Funciones de Limpieza ---


def remove_missing(values: List[Any]) -> List[Any]:
    """
    Elimina valores faltantes (None, "" y nan).
    Input: Lista de valores.
    Output: Lista sin valores faltantes.
    """
    result = []
    for v in values:
        is_nan = False
        try:
            # math.isnan solo funciona con números
            is_nan = math.isnan(float(v))
        except (TypeError, ValueError):
            pass  # El valor no es un número, por lo que no puede ser 'nan'

        if v is not None and v != "" and not is_nan:
            result.append(v)
    return result


def fill_missing(values: List[Any], fill_value: Any = 0) -> List[Any]:
    """
    Rellena valores faltantes (None, "", nan) con un valor.
    Input: Lista de valores, valor de relleno (default 0).
    Output: Lista con valores faltantes reemplazados.
    """
    result = []
    for v in values:
        is_nan = False
        try:
            is_nan = math.isnan(float(v))
        except (TypeError, ValueError):
            pass

        if v is None or v == "" or is_nan:
            result.append(fill_value)
        else:
            result.append(v)
    return result


def remove_duplicates(values: List[Any]) -> List[Any]:
    """
    Elimina valores duplicados de una lista, conservando el orden.
    Input: Lista de valores.
    Output: Lista con valores únicos.
    """
    # dict.fromkeys preserva el orden de inserción en Python 3.7+
    try:
        return list(dict.fromkeys(values))
    except TypeError:
        # Para tipos no "hasheables" como listas anidadas
        seen = []
        for item in values:
            if item not in seen:
                seen.append(item)
        return seen


# --- Funciones Numéricas ---


def normalize_min_max(
    values: List[float], new_min: float = 0.0, new_max: float = 1.0
) -> List[float]:
    """
    Normaliza valores numéricos usando la escala min-max.
    Input: Lista de números, nuevo min (default 0.0), nuevo max (default 1.0).
    Output: Lista de números normalizados.
    """
    if not values:
        return []

    old_min = min(values)
    old_max = max(values)
    old_range = old_max - old_min
    new_range = new_max - new_min

    if old_range == 0:
        # Todos los valores son iguales
        return [new_min] * len(values)

    return [(((v - old_min) * new_range) / old_range) + new_min for v in values]


def standardize_z_score(values: List[float]) -> List[float]:
    """
    Estandariza valores numéricos usando z-score.
    Input: Lista de números.
    Output: Lista de números estandarizados.
    """
    n = len(values)
    if n == 0:
        return []

    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std_dev = math.sqrt(variance)

    if std_dev == 0:
        # Todos los valores son iguales
        return [0.0] * n

    return [(v - mean) / std_dev for v in values]


def clip_values(values: List[float], min_val: float, max_val: float) -> List[float]:
    """
    Recorta valores numéricos a un rango [min_val, max_val].
    Input: Lista de números, valor mínimo, valor máximo.
    Output: Lista de números recortados.
    """
    return [max(min_val, min(v, max_val)) for v in values]


def convert_to_int(values: List[str]) -> List[int]:
    """
    Convierte una lista de strings a enteros, excluyendo no numéricos.
    Input: Lista de strings.
    Output: Lista de enteros.
    """
    result = []
    for v in values:
        try:
            # Convertir a float primero maneja strings como "3.14" o "2.0"
            result.append(int(float(v)))
        except (ValueError, TypeError):
            # No es un string numérico, se excluye
            pass
    return result


def transform_log(values: List[Union[int, float]]) -> List[float]:
    """
    Aplica una transformación logarítmica (log natural) a valores positivos.
    Input: Lista de números.
    Output: Lista de números transformados (solo positivos).
    """
    result = []
    for v in values:
        if v > 0:
            result.append(math.log(v))
    return result


# --- Funciones de Texto ---


def tokenize(text: str) -> List[str]:
    """
    Tokeniza texto en palabras, seleccionando solo alfanuméricos y minúsculas.
    Input: Texto a procesar.
    Output: Lista de palabras (tokens).
    """
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def select_alphanumeric_spaces(text: str) -> str:
    """
    Selecciona solo caracteres alfanuméricos y espacios.
    Input: Texto a procesar.
    Output: Texto procesado.
    """
    # Esto es equivalente a "eliminar puntuación"
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)


def remove_stopwords(text: str, stopwords: List[str]) -> str:
    """
    Elimina stop-words de un texto. El texto se procesa en minúsculas.
    Input: Texto a procesar, lista de stop-words.
    Output: Texto procesado sin stop-words.
    """
    words = text.lower().split()
    stopwords_set = set(stopwords)
    filtered_words = [word for word in words if word not in stopwords_set]
    return " ".join(filtered_words)


# --- Funciones de Estructura ---


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """
    Aplana una lista de listas.
    Input: Lista de listas.
    Output: Lista aplanada.
    """
    return [item for sublist in nested_list for item in sublist]


def shuffle_list(values: List[Any], seed: Optional[int] = None) -> List[Any]:
    """
    Mezcla aleatoriamente una lista de valores.
    Input: Lista de valores, semilla (opcional, default None).
    Output: Lista mezclada (es una nueva copia).
    """
    if seed is not None:
        random.seed(seed)

    # Crear una copia para no modificar la lista original
    shuffled_values = values[:]
    random.shuffle(shuffled_values)
    return shuffled_values
