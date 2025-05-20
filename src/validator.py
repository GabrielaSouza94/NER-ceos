import pandas as pd
import csv

#gold_path="../out_files/gold_teste.csv"
#respostas_path="../out_files/respostas_llm.csv"
def carregar_respostas(respostas_path):
    respostas_lidas = []
    with open(respostas_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if len(row) >= 4:
                arquivo = row[0].replace(".pdf", "").strip()
                tipo_entidade = row[1].strip().lower()
                nome = row[2].strip()
                identificador = row[3].strip()
                aux = row[4].strip() if len(row) > 4 else ""
                respostas_lidas.append({
                    "arquivo": arquivo,
                    "tipo_entidade": tipo_entidade,
                    "nome": nome,
                    "identificador": identificador,
                    "aux": aux
                })
    return pd.DataFrame(respostas_lidas)

def avaliar_extracao(gold_path, respostas_path):
    gold_df = pd.read_csv(gold_path, sep=";", encoding="utf-8-sig")
    respostas_df = carregar_respostas(respostas_path)

    # Normalizar colunas
    gold_df.columns = [col.strip ().lower ().replace (" ", "_") for col in gold_df.columns]
    respostas_df.columns = [col.strip ().lower ().replace (" ", "_") for col in respostas_df.columns]

    # Converter colunas para string para evitar erros de dtype
    for col in ["arquivo", "tipo_entidade", "nome", "identificador", "aux"]:
        if col in gold_df.columns:
            gold_df[col] = gold_df[col].astype (str)
        if col in respostas_df.columns:
            respostas_df[col] = respostas_df[col].astype (str)

    # Preencher nulos com string vazia
    gold_df.fillna ("", inplace=True)
    respostas_df.fillna ("", inplace=True)

    # Chaves alternativas para match
    gold_df["chave_id"] = gold_df["arquivo"].astype(str) + "|" + gold_df["tipo_entidade"] + "|" + gold_df["nome"].str.strip() + "|" + gold_df["identificador"].str.strip()
    gold_df["chave_rg"] = gold_df["arquivo"].astype(str) + "|" + gold_df["tipo_entidade"] + "|" + gold_df["nome"].str.strip() + "|" + gold_df["aux"].str.strip()

    respostas_df["chave_id"] = respostas_df["arquivo"].astype(str) + "|" + respostas_df["tipo_entidade"] + "|" + respostas_df["nome"].str.strip() + "|" + respostas_df["identificador"].str.strip()
    respostas_df["chave_rg"] = respostas_df["arquivo"].astype(str) + "|" + respostas_df["tipo_entidade"] + "|" + respostas_df["nome"].str.strip() + "|" + respostas_df["aux"].str.strip()

    # Verificar matches por qualquer chave
    gold_keys_id = set(gold_df["chave_id"])
    gold_keys_rg = set(gold_df["chave_rg"])
    respostas_keys_id = set(respostas_df["chave_id"])
    respostas_keys_rg = set(respostas_df["chave_rg"])

    # Contar acertos por correspondência de qualquer chave
    acertos = sum((key in gold_keys_id or key in gold_keys_rg) for key in respostas_keys_id.union(respostas_keys_rg))

    total_gold = len(gold_df)
    total_respostas = len(respostas_df)

    estatisticas = {
        "Total entidades esperadas (gold)": total_gold,
        "Total respostas LLM": total_respostas,
        "Corretamente identificadas (nome + identificador ou nome + rg)": acertos,
        "Precisão (%)": round(acertos / total_respostas * 100, 2) if total_respostas else 0.0,
        "Revocação (%)": round(acertos / total_gold * 100, 2) if total_gold else 0.0
    }

    return estatisticas
