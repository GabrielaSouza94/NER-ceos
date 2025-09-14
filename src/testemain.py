import sys
import os
from validador_estat import validar_entidades_simples


def main():
    arquivo_gold = "../out_files/gold_teste.csv"
    arquivo_respostas = "../out_files/respostas_llm.csv"


    # Opcionalmente, definir um arquivo de saída para os resultados
    arquivo_saida = "../out_files/resultados_validacao.txt"

    # Chamar o validador
    print ("Iniciando validação de entidades...")
    estatisticas = validar_entidades_simples (
        arquivo_gold=arquivo_gold,
        arquivo_respostas=arquivo_respostas,
        output_file=arquivo_saida,
        imprimir=True  # Se quiser imprimir estatísticas na tela
    )

    if estatisticas:
        # Aqui você pode fazer qualquer processamento adicional com as estatísticas
        print (f"Taxa de acerto: {estatisticas['taxa_acerto']:.2%}")

        # Exemplo: Verificar estatísticas específicas por arquivo
        print ("\nResumo de estatísticas por arquivo:")
        for arquivo, stats in estatisticas['por_arquivo'].items ():
            taxa = stats['encontradas'] / stats['total'] if stats['total'] > 0 else 0
            print (f"{arquivo}: {taxa:.2%} de acerto ({stats['encontradas']}/{stats['total']})")

            # Verificações específicas
            if stats['pessoas'] == 0 and stats['total'] > 0:
                print (f"  ATENÇÃO: Nenhuma pessoa identificada no arquivo {arquivo}")

            if stats['empresas'] == 0 and stats['total'] > 0:
                print (f"  ATENÇÃO: Nenhuma empresa identificada no arquivo {arquivo}")
    else:
        print ("Erro ao executar a validação.")


if __name__ == "__main__":
    main ()