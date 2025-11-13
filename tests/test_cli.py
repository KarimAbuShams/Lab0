"""
Pruebas de integración para la CLI.

Verifica que los comandos de la CLI (src/cli.py) se ejecuten
y llamen a la lógica de pre-procesamiento (src/preprocessing.py)
correctamente, comprobando la salida en la terminal.
"""

import pytest
from click.testing import CliRunner
from src.cli import cli # Importa el grupo principal de comandos

# --- 1. Fixture Requerida ---

@pytest.fixture
def runner():
    """
    Fixture para instanciar un CliRunner que será usado
    por todos los tests de integración.
    """
    return CliRunner()


# --- 2. Pruebas del Grupo 'clean' ---

def test_cli_remove_missing(runner):
    """
    Prueba el comando 'clean remove-missing'.
    Comprueba que el comando se ejecuta (exit_code == 0)
    y que la salida impresa (output) es la lista procesada.
    """
    # Simulamos: python -m src.cli clean remove-missing "1,2,,None,nan"
    result = runner.invoke(cli, ['clean', 'remove-missing', '1,2,,None,nan'])

    assert result.exit_code == 0
    # click.echo() añade un salto de línea al final, por eso el '\n'
    assert result.output == "['1', '2']\n"

def test_cli_fill_missing_with_option(runner):
    """
    Prueba el comando 'clean fill-missing' usando una opción.
    """
    # Simulamos: python -m src.cli clean fill-missing "1,,3" --fill 99
    result = runner.invoke(cli, [
        'clean',
        'fill-missing',
        '1,,3',
        '--fill',
        '99'
    ])

    assert result.exit_code == 0
    assert result.output == "['1', 99, '3']\n" # '99' se convierte a float/int


# --- 3. Pruebas del Grupo 'numeric' ---

def test_cli_normalize(runner):
    """Prueba el comando 'numeric normalize' con opciones."""
    # Simulamos: python -m src.cli numeric normalize "1,5,10" --min -1 --max 1
    result = runner.invoke(cli, [
        'numeric',
        'normalize',
        '1,5,10',
        '--min', '-1.0',
        '--max', '1.0'
    ])

    assert result.exit_code == 0
    # 1 -> -1.0
    # 5 -> 0.0
    # 10 -> 1.0
    assert result.output == "[-1.0, -0.11111111111111116, 1.0]\n"

def test_cli_clip(runner):
    """Prueba el comando 'numeric clip' con opciones default."""
    # Simulamos: python -m src.cli numeric clip "-1,0.5,2"
    # (usa defaults min=0.0, max=1.0)
    result = runner.invoke(cli, ['numeric', 'clip', '--', '-1,0.5,2'])
    
    assert result.exit_code == 0
    assert result.output == "[0.0, 0.5, 1.0]\n"


# --- 4. Pruebas del Grupo 'text' ---

def test_cli_tokenize(runner):
    """Prueba el comando 'text tokenize'."""
    result = runner.invoke(cli, ['text', 'tokenize', 'Hola, es 1 prueba!'])
    
    assert result.exit_code == 0
    assert result.output == "['hola', 'es', '1', 'prueba']\n"

def test_cli_remove_stopwords(runner):
    """Prueba el comando 'text remove-stopwords' con su opción."""
    result = runner.invoke(cli, [
        'text',
        'remove-stopwords',
        'el perro de mi amigo',
        '--stopwords', 'el,de,mi'
    ])
    
    assert result.exit_code == 0
    assert result.output == "perro amigo\n"


# --- 5. Pruebas del Grupo 'struct' ---

def test_cli_flatten(runner):
    """Prueba el comando 'struct flatten'."""
    # Recuerda que para ast.literal_eval necesitamos comillas
    # dobles y simples.
    arg = "[[1,2], [3,4]]"
    result = runner.invoke(cli, ['struct', 'flatten', arg])
    
    assert result.exit_code == 0
    assert result.output == "[1, 2, 3, 4]\n"

def test_cli_shuffle_seed(runner):
    """Prueba 'struct shuffle' con una semilla para reproducibilidad."""
    result = runner.invoke(cli, [
        'struct',
        'shuffle',
        'a,b,c,d,e',
        '--seed', '42'
    ])
    
    assert result.exit_code == 0
    # La semilla 42 siempre dará este orden
    assert result.output == "['d', 'b', 'c', 'e', 'a']\n"
