"""
Interfaz de Línea de Comandos (CLI) para el proyecto de pre-procesamiento.

Utiliza 'click' para crear un grupo principal 'cli' y cuatro subgrupos:
- clean: Funciones de limpieza de datos.
- numeric: Funciones para atributos numéricos.
- text: Funciones para procesar texto.
- struct: Funciones para estructuras de datos.
"""

import click
import ast  # Para convertir strings '[[1,2]]' a listas de listas
from typing import List, Any
import src.preprocessing as pp


# --- Ayudantes de conversión de tipos ---


def str_to_list(ctx: Any, param: Any, value: str) -> List[str]:
    """Callback de Click para convertir un string "a,b,c" a una lista [a, b, c]."""
    if not value:
        return []
    # Maneja valores con comas dentro, asumiendo que el usuario no los pone
    return value.split(",")


def str_to_float_list(ctx: Any, param: Any, value: str) -> List[float]:
    """Callback de Click para convertir "1,2,3.5" a [1.0, 2.0, 3.5]."""
    if not value:
        return []
    try:
        return [float(v) for v in value.split(",")]
    except ValueError as e:
        raise click.BadParameter(f"La lista contiene valores no numéricos: {e}")


def str_to_nested_list(ctx: Any, param: Any, value: str) -> List[List[Any]]:
    """Callback para convertir un string '[[1,2],[3,4]]' a una lista."""
    if not value:
        return []
    try:
        result = ast.literal_eval(value)
        if not isinstance(result, list) or not all(isinstance(i, list) for i in result):
            raise ValueError
        return result
    except (ValueError, SyntaxError):
        raise click.BadParameter(
            "Formato de lista de listas no válido. "
            "Use comillas y formato Python: '[[1,2],[3,4]]'"
        )


def str_to_str_list(ctx: Any, param: Any, value: str) -> List[str]:
    """Callback para la lista de stopwords "a,an,the"."""
    return str_to_list(ctx, param, value)


# --- Grupo Principal ---


@click.group()
def cli():
    """
    Herramienta CLI para aplicar varias técnicas de pre-procesamiento de datos.
    Los comandos están organizados en 4 grupos: clean, numeric, text, y struct.
    """
    pass


# --- 1. Grupo 'clean' ---


@click.group(help="Funciones relacionadas con la limpieza de datos.")
def clean():
    pass


@clean.command(help='Elimina valores faltantes (None, "", nan).')
@click.argument("values", callback=str_to_list)
def remove_missing(values):
    """
    Elimina valores faltantes de una lista de strings.
    Ejemplo: python -m src.cli clean remove-missing "1,2,,3,None,nan"
    """
    result = pp.remove_missing(values)
    click.echo(result)


@clean.command(help="Rellena valores faltantes.")
@click.argument("values", callback=str_to_list)
@click.option(
    "--fill",
    "fill_value",
    default=0,
    type=str,
    help="Valor con el que rellenar (default: 0).",
)
def fill_missing(values, fill_value):
    """
    Rellena valores faltantes con un valor (default 0).
    Ejemplo: python -m src.cli clean fill-missing "1,2,,3" --fill 99
    """
    # Intenta convertir el fill_value a número si es posible
    try:
        fill_value = float(fill_value)
        if fill_value.is_integer():
            fill_value = int(fill_value)
    except ValueError:
        pass  # Dejarlo como string

    result = pp.fill_missing(values, fill_value=fill_value)
    click.echo(result)


# --- 2. Grupo 'numeric' ---


@click.group(help="Funciones para procesar atributos numéricos.")
def numeric():
    pass


@numeric.command(help="Normaliza valores con min-max.")
@click.argument("values", callback=str_to_float_list)
@click.option("--min", "new_min", default="0.0", type=float, help="Nuevo mínimo (default 0.0).")
@click.option("--max", "new_max", default="1.0", type=float, help="Nuevo máximo (default 1.0).")
def normalize(values, new_min, new_max):
    """
    Normaliza una lista de números al rango [min, max].
    Ejemplo: python -m src.cli numeric normalize "1,2,3,4,5" --min -1 --max 1
    """
    result = pp.normalize_min_max(values, new_min=new_min, new_max=new_max)
    click.echo(result)


@numeric.command(help="Estandariza valores con z-score.")
@click.argument("values", callback=str_to_float_list)
def standardize(values):
    """
    Estandariza una lista de números usando z-score.
    Ejemplo: python -m src.cli numeric standardize "1,2,3,4,5"
    """
    result = pp.standardize_z_score(values)
    click.echo(result)


@numeric.command(help="Recorta valores numéricos a un rango.")
@click.argument("values", callback=str_to_float_list)
@click.option("--min", "min_val", default="0.0", type=float, help="Valor mínimo (default 0.0).")
@click.option("--max", "max_val", default="1.0", type=float, help="Valor máximo (default 1.0).")
def clip(values, min_val, max_val):
    """
    Recorta números a un rango [min, max].
    Ejemplo: python -m src.cli numeric clip "1,5,11,0" --min 1 --max 10
    """
    result = pp.clip_values(values, min_val=min_val, max_val=max_val)
    click.echo(result)
    
@numeric.command(help="Convierte lista de strings a enteros.")
@click.argument("values", callback=str_to_list)
def to_int(values):
    """
    Convierte lista de strings a enteros (ignora no numéricos).
    Ejemplo: python -m src.cli numeric to-int "1.1,2.9,hola,3"
    """
    result = pp.convert_to_int(values)
    click.echo(result)


@numeric.command(help="Transforma a escala logarítmica (solo positivos).")
@click.argument("values", callback=str_to_float_list)
def log_transform(values):
    """
    Aplica logaritmo natural a una lista (solo valores > 0).
    Ejemplo: python -m src.cli numeric log-transform "1,10,100,-5"
    """
    result = pp.transform_log(values)
    click.echo(result)


# --- 3. Grupo 'text' ---


@click.group(help="Funciones para procesar información textual.")
def text():
    pass


@text.command(help="Tokeniza texto (alfanuméricos y minúsculas).")
@click.argument("input_text", type=str)
def tokenize(input_text):
    """
    Tokeniza un string, quedándose con alfanuméricos en minúsculas.
    Ejemplo: python -m src.cli text tokenize "Hola, esto es 1 prueba!"
    """
    result = pp.tokenize(input_text)
    click.echo(result)


@text.command(help="Elimina puntuación (deja alfanuméricos y espacios).")
@click.argument("input_text", type=str)
def remove_punctuation(input_text):
    """
    Elimina puntuación de un string.
    Ejemplo: python -m src.cli text remove-punctuation "Hola!! ¿Qué tal?"
    """
    result = pp.select_alphanumeric_spaces(input_text)
    click.echo(result)


@text.command(help="Elimina stop-words de un texto.")
@click.argument("input_text", type=str)
@click.option(
    "--stopwords",
    "stop_words",
    required=True,
    callback=str_to_str_list,
    help="Lista de stop-words separadas por comas. Ej: 'el,la,los'",
)
def remove_stopwords(input_text, stop_words):
    """
    Elimina una lista de stop-words de un string.
    Ejemplo: python -m src.cli text remove-stopwords "el perro de mi amigo" --stopwords "el,de,mi"
    """
    result = pp.remove_stopwords(input_text, stopwords=stop_words)
    click.echo(result)


# --- 4. Grupo 'struct' ---


@click.group(help="Funciones para cambiar la estructura de los datos.")
def struct():
    pass


@struct.command(help="Mezcla una lista aleatoriamente.")
@click.argument("values", callback=str_to_list)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Semilla para reproducibilidad (default: None).",
)
def shuffle(values, seed):
    """
    Mezcla una lista. Usa --seed para un resultado reproducible.
    Ejemplo: python -m src.cli struct shuffle "a,b,c,d" --seed 42
    """
    result = pp.shuffle_list(values, seed=seed)
    click.echo(result)


@struct.command(help="Aplana una lista de listas.")
@click.argument("values", callback=str_to_nested_list)
def flatten(values):
    """
    Aplana una lista de listas.
    Ejemplo: python -m src.cli struct flatten "[[1,2],[3,4]]"
    (¡Recuerda usar comillas dobles y simples como en el ejemplo!)
    """
    result = pp.flatten_list(values)
    click.echo(result)


@struct.command(help="Obtiene valores únicos de una lista (conserva orden).")
@click.argument("values", callback=str_to_list)
def unique(values):
    """
    Devuelve los valores únicos de una lista.
    Ejemplo: python -m src.cli struct unique "1,b,1,c,b,2"
    """
    # Intentamos convertir a números para que '1' y '1.0' sean iguales
    processed_values = []
    for v in values:
        try:
            fv = float(v)
            processed_values.append(int(fv) if fv.is_integer() else fv)
        except ValueError:
            processed_values.append(v)

    result = pp.remove_duplicates(processed_values)
    click.echo(result)


# --- Añadir grupos al CLI principal ---
cli.add_command(clean)
cli.add_command(numeric)
cli.add_command(text)
cli.add_command(struct)


if __name__ == "__main__":
    cli()
