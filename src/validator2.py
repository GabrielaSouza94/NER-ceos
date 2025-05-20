import pandas as pd
import re
from collections import defaultdict
import os


def carregar_csv(caminho, encoding='utf-8-sig', is_gold=False):
    """Carrega um arquivo CSV e retorna um DataFrame pandas."""
    try:
        # Detecta o separador: vírgula ou ponto e vírgula
        with open (caminho, 'r', encoding=encoding) as f:
            primeira_linha = f.readline ()
            separador = ';' if ';' in primeira_linha else ','

        # Para o arquivo de respostas, precisamos tratar aspas inconsistentes

            # Processa normalmente o arquivo gold
            df = pd.read_csv (caminho, encoding=encoding, sep=separador)

        # Se for o arquivo gold, remove a extensão .pdf dos nomes dos arquivos para facilitar a comparação
        if is_gold:
            if 'arquivo' in df.columns:
                df['arquivo'] = df['arquivo'].astype (str)
        else:
            # Se for o arquivo de respostas, remove a extensão .pdf se presente
            if 'arquivo' in df.columns:
                df['arquivo'] = df['arquivo'].astype (str).str.replace ('.pdf', '')

        return df
    except UnicodeDecodeError:
        # Tenta com encoding alternativo
        if encoding != 'latin1':
            return carregar_csv (caminho, encoding='latin1', is_gold=is_gold)
        else:
            print (f"Erro de codificação no arquivo {caminho}")
            return None
    except Exception as e:
        print (f"Erro ao carregar o arquivo {caminho}: {e}")
        return None


def normalizar_texto(texto):
    """Normaliza o texto removendo acentos, convertendo para minúsculas e removendo caracteres especiais."""
    if pd.isna (texto):
        return ""

    # Converte para string caso não seja
    texto = str (texto).lower ().strip ()

    # Remove caracteres especiais, mas mantém espaços
    texto = re.sub (r'[^\w\s]', '', texto)

    # Remove espaços extras
    texto = re.sub (r'\s+', ' ', texto)

    return texto


def normalizar_identificador(identificador):
    """Normaliza identificadores como CPF/CNPJ removendo formatação."""
    if pd.isna (identificador):
        return ""

    # Converte para string e remove texto como "CPF:" ou "RG:"
    texto = str (identificador).lower ().strip ()
    texto = re.sub (r'(cpf|cnpj|rg|sspsc|sspssc)(\s*:|\s+)', '', texto)

    # Remove caracteres não numéricos
    texto = re.sub (r'[^\d]', '', texto)

    return texto


def entidade_esta_presente(entidade_gold, identificador_gold, auxiliar_gold, df_respostas, arquivo):
    """
    Verifica se uma entidade do arquivo gold está presente no arquivo de respostas.
    A entidade é considerada presente se (nome + identificador) ou (nome + auxiliar) estiver nas respostas.

    Args:
        entidade_gold: Nome da entidade no arquivo gold
        identificador_gold: Identificador da entidade no arquivo gold
        auxiliar_gold: Auxiliar da entidade no arquivo gold
        df_respostas: DataFrame com as respostas
        arquivo: Nome do arquivo a filtrar no DataFrame de respostas

    Returns:
        (bool, str): Tupla com status (encontrado ou não) e tipo de match (nome+id ou nome+auxiliar)
    """
    nome_normalizado = normalizar_texto (entidade_gold)
    id_normalizado = normalizar_identificador (identificador_gold)
    auxiliar_normalizado = normalizar_identificador (auxiliar_gold)

    # Filtra respostas pelo mesmo arquivo
    respostas_mesmo_arquivo = df_respostas[df_respostas['arquivo'] == arquivo]

    for _, resposta_row in respostas_mesmo_arquivo.iterrows ():
        nome_resp = normalizar_texto (resposta_row.get ('nome', ''))
        id_resp = normalizar_identificador (resposta_row.get ('identificador', ''))
        aux_resp = normalizar_identificador (resposta_row.get ('aux', ''))

        # Verifica se os nomes são similares (considerando possíveis variações de digitação)
        nomes_similares = nome_normalizado in nome_resp or nome_resp in nome_normalizado

        if nomes_similares:
            # Verifica nome+identificador
            if id_normalizado and id_resp and (id_normalizado in id_resp or id_resp in id_normalizado):
                return True, "nome+identificador"

            # Verifica nome+auxiliar
            if auxiliar_normalizado and aux_resp and (
                    auxiliar_normalizado in aux_resp or aux_resp in auxiliar_normalizado):
                return True, "nome+auxiliar"

    return False, None


def validar_entidades(df_gold, df_respostas):
    """
    Valida se as entidades do arquivo gold estão presentes no arquivo de respostas.

    Args:
        df_gold: DataFrame com as entidades originais (gold)
        df_respostas: DataFrame com as respostas a serem validadas

    Returns:
        Estatísticas e lista de entidades não encontradas
    """
    total_entidades = len (df_gold)
    entidades_encontradas = 0
    encontradas_nome_id = 0
    encontradas_nome_auxiliar = 0
    entidades_nao_encontradas = []

    # Estatísticas por tipo de entidade
    stats_por_tipo = defaultdict (lambda: {
        'total': 0,
        'encontradas': 0,
        'nome_id': 0,
        'nome_auxiliar': 0
    })

    # Estatísticas por arquivo
    stats_por_arquivo = defaultdict (lambda: {
        'total': 0,
        'encontradas': 0,
        'pessoas': 0,
        'empresas': 0,
        'cpfs': 0,
        'cnpjs': 0
    })

    # Para cada entidade no arquivo gold
    for i, row in df_gold.iterrows ():
        # Extrai os campos relevantes
        arquivo = row.get ('arquivo', '')
        tipo_entidade = row.get ('tipo_entidade', '').lower ()
        entidade = row.get ('nome', '')
        identificador = row.get ('identificador', '')
        auxiliar = row.get ('aux', '')

        # Incrementa contadores por tipo
        stats_por_tipo[tipo_entidade]['total'] += 1

        # Incrementa contadores por arquivo
        stats_por_arquivo[arquivo]['total'] += 1

        # Verifica se a entidade está presente nas respostas
        encontrada, tipo_encontrada = entidade_esta_presente (
            entidade,
            identificador,
            auxiliar,
            df_respostas,
            arquivo
        )

        if encontrada:
            entidades_encontradas += 1
            stats_por_tipo[tipo_entidade]['encontradas'] += 1
            stats_por_arquivo[arquivo]['encontradas'] += 1

            # Incrementa contadores específicos por arquivo
            if tipo_entidade.lower () == 'pessoa':
                stats_por_arquivo[arquivo]['pessoas'] += 1
            elif tipo_entidade.lower () == 'empresa':
                stats_por_arquivo[arquivo]['empresas'] += 1

            # Verifica se o identificador é CPF ou CNPJ
            id_limpo = normalizar_identificador (identificador)
            if id_limpo:
                if len (id_limpo) == 11:  # Tamanho de um CPF
                    stats_por_arquivo[arquivo]['cpfs'] += 1
                elif len (id_limpo) == 14:  # Tamanho de um CNPJ
                    stats_por_arquivo[arquivo]['cnpjs'] += 1

            if tipo_encontrada == "nome+identificador":
                encontradas_nome_id += 1
                stats_por_tipo[tipo_entidade]['nome_id'] += 1
            elif tipo_encontrada == "nome+auxiliar":
                encontradas_nome_auxiliar += 1
                stats_por_tipo[tipo_entidade]['nome_auxiliar'] += 1
        else:
            entidades_nao_encontradas.append ({
                'arquivo': arquivo,
                'tipo_entidade': tipo_entidade,
                'entidade': entidade,
                'identificador': identificador,
                'auxiliar': auxiliar
            })

    # Compilar estatísticas
    estatisticas = {
        'total_entidades': total_entidades,
        'entidades_encontradas': entidades_encontradas,
        'taxa_acerto': entidades_encontradas / total_entidades if total_entidades > 0 else 0,
        'encontradas_nome_id': encontradas_nome_id,
        'encontradas_nome_auxiliar': encontradas_nome_auxiliar,
        'entidades_nao_encontradas': len (entidades_nao_encontradas),
        'por_tipo': stats_por_tipo,
        'por_arquivo': stats_por_arquivo
    }

    return estatisticas, entidades_nao_encontradas


def imprimir_estatisticas(estatisticas):
    """
    Imprime as estatísticas de validação de forma estruturada.

    Args:
        estatisticas: Dicionário com as estatísticas compiladas
    """
    print ("\n====== ESTATÍSTICAS DE VALIDAÇÃO ======")
    print (f"Total de entidades a validar: {estatisticas['total_entidades']}")
    print (f"Entidades encontradas: {estatisticas['entidades_encontradas']} ({estatisticas['taxa_acerto']:.2%})")
    print (f"   - Via nome+identificador: {estatisticas['encontradas_nome_id']}")
    print (f"   - Via nome+auxiliar: {estatisticas['encontradas_nome_auxiliar']}")
    print (f"Entidades não encontradas: {estatisticas['entidades_nao_encontradas']}")

    # Estatísticas por tipo de entidade
    print ("\n====== ESTATÍSTICAS POR TIPO DE ENTIDADE ======")
    for tipo, stats in estatisticas['por_tipo'].items ():
        print (f"Tipo: {tipo.upper ()}")
        print (f"  Total: {stats['total']}")
        taxa = stats['encontradas'] / stats['total'] if stats['total'] > 0 else 0
        print (f"  Encontradas: {stats['encontradas']} ({taxa:.2%})")
        print (f"    - Via nome+identificador: {stats['nome_id']}")
        print (f"    - Via nome+auxiliar: {stats['nome_auxiliar']}")

    # Estatísticas por arquivo
    print ("\n====== ESTATÍSTICAS POR ARQUIVO ======")
    for arquivo, stats in estatisticas['por_arquivo'].items ():
        print (f"Arquivo: {arquivo}")
        print (f"  Total de entidades: {stats['total']}")
        taxa = stats['encontradas'] / stats['total'] if stats['total'] > 0 else 0
        print (f"  Entidades encontradas: {stats['encontradas']} ({taxa:.2%})")
        print (f"    - Pessoas identificadas: {stats['pessoas']}")
        print (f"    - Empresas identificadas: {stats['empresas']}")
        print (f"    - CPFs identificados: {stats['cpfs']}")
        print (f"    - CNPJs identificados: {stats['cnpjs']}")


def salvar_resultados(estatisticas, nao_encontradas, output_file):
    """
    Salva os resultados em um arquivo de texto e um CSV.

    Args:
        estatisticas: Dicionário com as estatísticas compiladas
        nao_encontradas: Lista de entidades não encontradas
        output_file: Caminho do arquivo para salvar os resultados
    """
    # Criar DataFrame com entidades não encontradas
    df_nao_encontradas = pd.DataFrame (nao_encontradas)

    # Salvar as estatísticas e entidades não encontradas em um arquivo de texto
    with open (output_file, 'w', encoding='utf-8') as f:
        f.write ("====== ESTATÍSTICAS DE VALIDAÇÃO ======\n")
        f.write (f"Total de entidades a validar: {estatisticas['total_entidades']}\n")
        f.write (
            f"Entidades encontradas: {estatisticas['entidades_encontradas']} ({estatisticas['taxa_acerto']:.2%})\n")
        f.write (f"   - Via nome+identificador: {estatisticas['encontradas_nome_id']}\n")
        f.write (f"   - Via nome+auxiliar: {estatisticas['encontradas_nome_auxiliar']}\n")
        f.write (f"Entidades não encontradas: {estatisticas['entidades_nao_encontradas']}\n")

        f.write ("\n====== ESTATÍSTICAS POR TIPO DE ENTIDADE ======\n")
        for tipo, stats in estatisticas['por_tipo'].items ():
            f.write (f"Tipo: {tipo.upper ()}\n")
            f.write (f"  Total: {stats['total']}\n")
            taxa = stats['encontradas'] / stats['total'] if stats['total'] > 0 else 0
            f.write (f"  Encontradas: {stats['encontradas']} ({taxa:.2%})\n")
            f.write (f"    - Via nome+identificador: {stats['nome_id']}\n")
            f.write (f"    - Via nome+auxiliar: {stats['nome_auxiliar']}\n")

        f.write ("\n====== ESTATÍSTICAS POR ARQUIVO ======\n")
        for arquivo, stats in estatisticas['por_arquivo'].items ():
            f.write (f"Arquivo: {arquivo}\n")
            f.write (f"  Total de entidades: {stats['total']}\n")
            taxa = stats['encontradas'] / stats['total'] if stats['total'] > 0 else 0
            f.write (f"  Entidades encontradas: {stats['encontradas']} ({taxa:.2%})\n")
            f.write (f"    - Pessoas identificadas: {stats['pessoas']}\n")
            f.write (f"    - Empresas identificadas: {stats['empresas']}\n")
            f.write (f"    - CPFs identificados: {stats['cpfs']}\n")
            f.write (f"    - CNPJs identificados: {stats['cnpjs']}\n")

        f.write ("\n====== ENTIDADES NÃO ENCONTRADAS ======\n")

    # Salvar DataFrame de entidades não encontradas como CSV
    df_nao_encontradas.to_csv (f"{output_file}.csv", index=False, encoding='utf-8-sig')
    print (f"Resultados salvos em {output_file} e {output_file}.csv")

    # Salvar estatísticas detalhadas por arquivo
    arquivo_stats_por_arquivo = f"{output_file}_por_arquivo.csv"
    lista_stats_arquivo = []
    for arquivo, stats in estatisticas['por_arquivo'].items ():
        taxa = stats['encontradas'] / stats['total'] if stats['total'] > 0 else 0
        lista_stats_arquivo.append ({
            'arquivo': arquivo,
            'total': stats['total'],
            'encontradas': stats['encontradas'],
            'taxa_acerto': taxa,
            'pessoas': stats['pessoas'],
            'empresas': stats['empresas'],
            'cpfs': stats['cpfs'],
            'cnpjs': stats['cnpjs']
        })

    df_stats_arquivo = pd.DataFrame (lista_stats_arquivo)
    df_stats_arquivo.to_csv (arquivo_stats_por_arquivo, index=False, encoding='utf-8-sig')
    print (f"Estatísticas por arquivo salvas em {arquivo_stats_por_arquivo}")


def executar_validacao(arquivo_gold, arquivo_respostas, output_file=None, silent=False):
    """
    Função principal que executa todo o processo de validação.

    Args:
        arquivo_gold: Caminho para o arquivo CSV original (gold)
        arquivo_respostas: Caminho para o arquivo CSV com as respostas
        output_file: Caminho para salvar os resultados (opcional)
        silent: Se True, não imprime estatísticas na tela (apenas salva se output_file for fornecido)

    Returns:
        Tupla (estatisticas, nao_encontradas) ou None se ocorrer erro
    """
    # Carregar arquivos
    if not silent:
        print (f"Carregando arquivo gold: {arquivo_gold}")
    df_gold = carregar_csv (arquivo_gold, is_gold=True)
    if df_gold is None:
        return None

    if not silent:
        print (f"Carregando arquivo de respostas: {arquivo_respostas}")
    df_respostas = carregar_csv (arquivo_respostas, is_gold=False)
    if df_respostas is None:
        return None

    # Exibir informações dos arquivos
    if not silent:
        print (f"\nArquivo gold contém {len (df_gold)} linhas e {len (df_gold.columns)} colunas")
        print (f"Colunas do arquivo gold: {', '.join (df_gold.columns.tolist ())}")
        print (f"\nArquivo de respostas contém {len (df_respostas)} linhas e {len (df_respostas.columns)} colunas")
        print (f"Colunas do arquivo de respostas: {', '.join (df_respostas.columns.tolist ())}")
        print ("\nValidando entidades...")

    # Validar entidades
    resultados = validar_entidades(df_gold, df_respostas)

    if resultados is None:
        return None

    estatisticas, nao_encontradas = resultados

    # Imprimir estatísticas se não for modo silencioso
    if not silent:
        imprimir_estatisticas (estatisticas)

    # Salvar resultados, se solicitado
    if output_file:
        salvar_resultados (estatisticas, nao_encontradas, output_file)

    return estatisticas, nao_encontradas


# Funções para usar como API a partir do programa principal
def validar_entidades_simples(arquivo_gold, arquivo_respostas, output_file=None, imprimir=True):
    """
    Função simplificada para ser chamada a partir do programa principal.

    Args:
        arquivo_gold: Caminho para o arquivo CSV gold
        arquivo_respostas: Caminho para o arquivo CSV de respostas
        output_file: Caminho para salvar os resultados (opcional)
        imprimir: Se True, imprime estatísticas na tela

    Returns:
        dict: Estatísticas compiladas ou None se ocorrer erro
    """
    resultados = executar_validacao (arquivo_gold, arquivo_respostas, output_file, silent=not imprimir)

    if resultados is None:
        return None

    return resultados[0]  # Retorna apenas as estatísticas


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser (description='Validador de entidades em arquivos CSV')
    parser.add_argument ('arquivo_gold', help='Caminho para o arquivo CSV original (gold)')
    parser.add_argument ('arquivo_respostas', help='Caminho para o arquivo CSV com as respostas')
    parser.add_argument ('--output', help='Arquivo para salvar os resultados (opcional)')
    parser.add_argument ('--silent', action='store_true', help='Executa sem imprimir informações na tela')

    args = parser.parse_args ()

    # Executar validação com os parâmetros recebidos
    executar_validacao (args.arquivo_gold, args.arquivo_respostas, args.output, silent=args.silent)