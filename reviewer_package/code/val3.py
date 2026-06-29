#!/usr/bin/env python3
"""
Validador de Entidades Baseado em ID - 
Compara arquivo gabarito com arquivo de respostas
Estrutura CSV: Id, tipo_entidade, nome, identificador, aux

Regras de validação alinhadas a validador_json3 (CSV flat: Id, tipo_entidade, nome, identificador, aux):
- Normalização de texto (maiúsculas, sem pontuação), sufixos societários removidos em empresas
- CPF/CNPJ: só dígitos; RG: alfanumérico; gold vazio → resposta deve estar vazia
- Empresa correta = nome (similaridade >= tolerância) + identificador exato
- Sócio completo = nome + identificador/aux conforme gabarito
- Métrica global "TODAS EMPRESAS + TODOS SÓCIOS" + média por documento
"""

import csv
import re
import json
from collections import defaultdict
from difflib import SequenceMatcher
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, Any


@dataclass
class MetricasEntidade:
    """Classe para armazenar métricas de validação de entidades"""
    
    # ===== EMPRESAS =====
    total_empresas_gabarito: int = 0
    total_empresas_resposta: int = 0
    empresas_corretas: int = 0
    empresas_nomes_corretos: int = 0
    empresas_cnpjs_corretos: int = 0
    empresas_extras: int = 0
    empresas_faltantes: int = 0
    
    # ===== PESSOAS (SÓCIOS) =====
    total_pessoas_gabarito: int = 0
    total_pessoas_resposta: int = 0
    
    # Combinações de validação de pessoas (CPF/RG)
    pessoas_nome_cpf_corretas: int = 0
    pessoas_nome_rg_corretas: int = 0  # ← MÉTRICA PRINCIPAL DE SÓCIOS
    pessoas_cpf_rg_corretas: int = 0
    pessoas_completas: int = 0  # Nome + identificador/aux conforme gold
    
    # Campos individuais (identificador/aux vazio no gold → vazio na resposta = correto)
    pessoas_nome_correto: int = 0
    pessoas_cpf_correto: int = 0
    pessoas_rg_correto: int = 0
    
    pessoas_extras: int = 0
    pessoas_faltantes: int = 0
    
    # ===== NOVAS MÉTRICAS COMBINADAS EMPRESA+SÓCIO =====
    empresa_socio_nome_cpf: int = 0                # Empresa (Nome+CNPJ) + Sócio (Nome+CPF)
    empresa_socio_nome_rg: int = 0                 # Empresa (Nome+CNPJ) + Sócio (Nome+RG)
    empresa_socio_nome_cpf_rg: int = 0             # Empresa correta + sócio 100% conforme gabarito
    
    socios_corretos: int = 0  # Alias para pessoas_nome_rg_corretas
    
    # ===== MÉTRICA GLOBAL - VALIDAÇÃO COMPLETA =====
    documentos_completos: int = 0      # Empresa + TODOS sócios conforme gabarito
    documentos_parciais: int = 0       # Empresa correta + ALGUNS sócios corretos
    
    empresas_com_socios_completos: int = 0
    empresas_com_socios_parciais: int = 0
    
    # ===== NOVA: Por Documento - Validação Rigorosa =====
    documento_todas_empresas_socios: bool = False  # True se TUDO passou


def normalizar_texto(texto: str) -> str:
    """Normaliza texto removendo acentos, pontuação e espaços extras (maiúsculas)."""
    if not texto:
        return ""

    texto = ''.join(
        c for c in unicodedata.normalize('NFD', str(texto))
        if unicodedata.category(c) != 'Mn'
    )
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = texto.upper()
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def limpar_extensoes_empresa(nome: str) -> str:
    """Remove sufixos societários irrelevantes de nomes de empresas."""
    if not nome:
        return ""

    extensoes_irrelevantes = [
        "LTDA", "L T D A", "LTDA ME",
        "EIRELI", "E I R E L I",
        "ME", "M E", "EPP", "E P P",
        "SA", "S A", "SOCIEDADE LTDA",
        "EMPRESA INDIVIDUAL", "MICROEMPRESA",
    ]

    nome_limpo = nome.upper()
    for termo in extensoes_irrelevantes:
        nome_limpo = nome_limpo.replace(termo, "")

    return ' '.join(nome_limpo.split()).strip()


_VALORES_INVALIDOS = frozenset({
    'não informado', 'nao informado', 'n/a', 'na', '-', '--', '',
})


def _valor_vazio(valor: str) -> bool:
    if not valor:
        return True
    return str(valor).strip().lower() in _VALORES_INVALIDOS


def normalizar_id(id_valor: str) -> str:
    """Normaliza ID removendo extensões (.pdf, .jpg, etc) e convertendo para número"""
    if not id_valor or id_valor == "":
        return ""
    
    # Remove extensões de arquivo
    id_str = str(id_valor).strip()
    id_str = re.sub(r'\.(pdf|jpg|png|docx|doc|txt)$', '', id_str, flags=re.IGNORECASE)
    
    # Mantém apenas dígitos
    id_normalizado = re.sub(r'\D', '', id_str)
    
    return id_normalizado


def normalizar_identificador(identificador: str) -> str:
    """Normaliza CPF/CNPJ removendo caracteres não numéricos e valores inválidos."""
    if _valor_vazio(identificador):
        return ""

    id_str = str(identificador).strip()

    if 'e' in id_str.lower():
        return ""

    return re.sub(r'\D', '', id_str)


def normalizar_rg(rg: str) -> str:
    """Normaliza RG removendo caracteres especiais mas mantendo letras."""
    if _valor_vazio(rg):
        return ""
    rg_limpo = re.sub(r"[^\w]", "", str(rg).strip())
    return rg_limpo.upper()


def similaridade_texto(texto1: str, texto2: str) -> float:
    """Calcula similaridade entre dois textos já normalizados ou brutos (0 a 1)."""
    a = normalizar_texto(texto1)
    b = normalizar_texto(texto2)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def carregar_entidades_csv(arquivo_csv: str) -> Dict[str, List[Dict]]:
    """Carrega entidades do arquivo CSV com tratamento robusto"""
    entidades_por_id = defaultdict(list)
    
    with open(arquivo_csv, 'r', encoding='utf-8-sig') as f:
        # Detecta o delimitador (vírgula, ponto-e-vírgula ou tab) pela 1ª linha.
        primeira_linha = f.readline()
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(primeira_linha, delimiters=",;\t")
            delimitador = dialect.delimiter
        except csv.Error:
            # Fallback: escolhe o separador mais frequente no cabeçalho
            delimitador = max(",;\t", key=primeira_linha.count) if primeira_linha else ","

        reader = csv.DictReader(f, delimiter=delimitador)

        if reader.fieldnames is None:
            raise ValueError(f"Arquivo CSV vazio ou mal formado: {arquivo_csv}")

        fieldnames = reader.fieldnames
        fieldnames_lower = {k.lower(): k for k in fieldnames}
        
        id_key = fieldnames_lower.get('arquivo') or fieldnames_lower.get('id') or fieldnames_lower.get('document_id')
        tipo_key = fieldnames_lower.get('tipo_entidade') or fieldnames_lower.get('tipo') or fieldnames_lower.get('type')
        nome_key = fieldnames_lower.get('nome') or fieldnames_lower.get('name') or fieldnames_lower.get('entity_name')
        identificador_key = fieldnames_lower.get('identificador') or fieldnames_lower.get('cpf') or fieldnames_lower.get('cnpj')
        aux_key = fieldnames_lower.get('aux') or fieldnames_lower.get('rg') or fieldnames_lower.get('auxiliary')
        
        for row_num, row in enumerate(reader, start=2):
            try:
                id_valor = row.get(id_key, "").strip() if id_key else ""
                tipo_valor = row.get(tipo_key, "").strip() if tipo_key else ""
                nome_valor = row.get(nome_key, "").strip() if nome_key else ""
                identificador_valor = row.get(identificador_key, "").strip() if identificador_key else ""
                aux_valor = row.get(aux_key, "").strip() if aux_key else ""
                
                id_normalizado = normalizar_id(id_valor)
                
                if not id_normalizado:
                    continue
                
                if not nome_valor and not identificador_valor:
                    continue
                
                entidade = {
                    'id': id_normalizado,
                    'tipo': tipo_valor,
                    'nome': nome_valor,
                    'identificador': normalizar_identificador(identificador_valor),
                    'aux': normalizar_rg(aux_valor),
                }
                
                entidades_por_id[id_normalizado].append(entidade)
            
            except Exception as e:
                print(f"⚠️  Aviso: Erro processando linha {row_num}: {e}")
                continue
    
    return entidades_por_id


def comparar_empresa(entidade_gabarito: Dict, entidade_resposta: Dict,
                     tolerancia_nome: float) -> Tuple[bool, bool, bool]:
    """
    Compara empresa gabarito vs resposta.
    Returns: (empresa_correta, nome_correto, identificador_correto)
    """
    nome_gab = limpar_extensoes_empresa(normalizar_texto(entidade_gabarito['nome']))
    nome_resp = limpar_extensoes_empresa(normalizar_texto(entidade_resposta['nome']))

    id_gab = normalizar_identificador(entidade_gabarito['identificador'])
    id_resp = normalizar_identificador(entidade_resposta['identificador'])

    nome_correto = similaridade_texto(nome_gab, nome_resp) >= tolerancia_nome
    identificador_correto = id_gab == id_resp
    empresa_correta = nome_correto and identificador_correto

    return empresa_correta, nome_correto, identificador_correto


def comparar_pessoa(entidade_gabarito: Dict, entidade_resposta: Dict,
                    tolerancia_nome: float) -> Dict[str, bool]:
    """
    Compara pessoa/sócio gabarito vs resposta.
    identificador/aux: se o gold está vazio, a resposta deve estar vazia;
    se o gold tem valor, deve coincidir após normalização.
    """
    nome_resp_raw = normalizar_texto(entidade_resposta.get('nome', ''))
    if "não é possível fornecer" in nome_resp_raw or "pois o texto não contém" in nome_resp_raw:
        return {
            'nome_correto': False,
            'cpf_correto': False,
            'rg_correto': False,
            'nome_cpf': False,
            'nome_rg': False,
            'cpf_rg': False,
            'completo': False,
        }

    nome_gab = normalizar_texto(entidade_gabarito['nome'])
    cpf_gab = normalizar_identificador(entidade_gabarito['identificador'])
    cpf_resp = normalizar_identificador(entidade_resposta['identificador'])
    rg_gab = normalizar_rg(entidade_gabarito['aux'])
    rg_resp = normalizar_rg(entidade_resposta['aux'])

    nome_correto = (
        bool(nome_gab)
        and similaridade_texto(nome_gab, nome_resp_raw) >= tolerancia_nome
    )

    if cpf_gab == "":
        cpf_correto = cpf_resp == ""
    else:
        cpf_correto = cpf_gab == cpf_resp

    if rg_gab == "":
        rg_correto = rg_resp == ""
    else:
        rg_correto = rg_gab == rg_resp

    return {
        'nome_correto': nome_correto,
        'cpf_correto': cpf_correto,
        'rg_correto': rg_correto,
        'nome_cpf': nome_correto and cpf_correto,
        'nome_rg': nome_correto and rg_correto,
        'cpf_rg': cpf_correto and rg_correto,
        'completo': nome_correto and cpf_correto and rg_correto,
    }


class ValidadorEntidadesID:
    """Classe principal para validar entidades baseado em ID - VERSÃO 2 COM NOVA MÉTRICA GLOBAL"""
    
    def __init__(self, arquivo_gabarito: str, arquivo_respostas: str,
                 limiar_similaridade: float = 0.85, config_params: Dict = None,
                 tolerancia_nome: float = None, tolerancia_nome_socio: float = None):
        self.arquivo_gabarito = arquivo_gabarito
        self.arquivo_respostas = arquivo_respostas
        self.limiar_similaridade = limiar_similaridade
        self.tolerancia_nome = tolerancia_nome if tolerancia_nome is not None else limiar_similaridade
        self.tolerancia_nome_socio = (
            tolerancia_nome_socio if tolerancia_nome_socio is not None else limiar_similaridade
        )
        self.config_params = config_params or {}
        
        self.gabarito = {}
        self.respostas = {}
        self.metricas_por_id = {}
        self.metricas_globais = MetricasEntidade()
        
        # ===== NOVO: Rastreamento para métrica global rigorosa =====
        self.documentos_validacao_completa = []  # Lista de docs que passaram
        self.metricas_validacao_completa_por_doc = {}  # Métricas por doc
    
    def carregar_arquivos(self) -> bool:
        """Carrega os arquivos CSV"""
        try:
            print(f"📂 Carregando gabarito: {self.arquivo_gabarito}")
            self.gabarito = carregar_entidades_csv(self.arquivo_gabarito)
            print(f"✅ Gabarito carregado: {len(self.gabarito)} documentos, "
                  f"{sum(len(e) for e in self.gabarito.values())} entidades")
            
            print(f"\n📂 Carregando respostas: {self.arquivo_respostas}")
            self.respostas = carregar_entidades_csv(self.arquivo_respostas)
            print(f"✅ Respostas carregadas: {len(self.respostas)} documentos, "
                  f"{sum(len(e) for e in self.respostas.values())} entidades")
            
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar arquivos: {e}")
            return False
    
    def validar_documento(self, doc_id: str) -> MetricasEntidade:
        """Valida um documento específico com rastreamento de validação completa"""
        metricas = MetricasEntidade()
        
        entidades_gab = self.gabarito.get(doc_id, [])
        entidades_resp = self.respostas.get(doc_id, [])
        
        # Separar por tipo
        empresas_gab = [e for e in entidades_gab if 'empresa' in normalizar_texto(e['tipo']).lower()]
        pessoas_gab = [e for e in entidades_gab if 'pessoa' in normalizar_texto(e['tipo']).lower()]
        
        empresas_resp = [e for e in entidades_resp if 'empresa' in normalizar_texto(e['tipo']).lower()]
        pessoas_resp = [e for e in entidades_resp if 'pessoa' in normalizar_texto(e['tipo']).lower()]
        
        # ===== VALIDAÇÃO DE EMPRESAS =====
        metricas.total_empresas_gabarito = len(empresas_gab)
        metricas.total_empresas_resposta = len(empresas_resp)
        
        empresas_resp_processadas = set()
        empresas_corretas_indices = []
        
        for emp_gab in empresas_gab:
            encontrada = False

            for i, emp_resp in enumerate(empresas_resp):
                if i in empresas_resp_processadas:
                    continue

                empresa_correta, nome_correto, cnpj_correto = comparar_empresa(
                    emp_gab, emp_resp, self.tolerancia_nome
                )

                if nome_correto or cnpj_correto:
                    empresas_resp_processadas.add(i)
                    encontrada = True

                    if empresa_correta:
                        metricas.empresas_corretas += 1
                        empresas_corretas_indices.append(i)
                    if nome_correto:
                        metricas.empresas_nomes_corretos += 1
                    if cnpj_correto:
                        metricas.empresas_cnpjs_corretos += 1

                    break

            if not encontrada:
                metricas.empresas_faltantes += 1
        
        metricas.empresas_extras = len(empresas_resp) - len(empresas_resp_processadas)
        
        # ===== VALIDAÇÃO DE PESSOAS (SÓCIOS) =====
        metricas.total_pessoas_gabarito = len(pessoas_gab)
        metricas.total_pessoas_resposta = len(pessoas_resp)
        
        pessoas_resp_processadas = set()
        pessoas_corretas_count = 0
        
        for pes_gab in pessoas_gab:
            encontrada = False
            melhor_match = None
            melhor_score = 0
            melhor_index = -1

            for i, pes_resp in enumerate(pessoas_resp):
                if i in pessoas_resp_processadas:
                    continue

                comparacao = comparar_pessoa(pes_gab, pes_resp, self.tolerancia_nome_socio)
                score = sum([
                    comparacao['nome_correto'] * 3,
                    comparacao['cpf_correto'] * 2,
                    comparacao['rg_correto'] * 1,
                ])

                if score > melhor_score:
                    melhor_score = score
                    melhor_match = comparacao
                    melhor_index = i

            if melhor_match and melhor_score > 0:
                pessoas_resp_processadas.add(melhor_index)
                encontrada = True

                if melhor_match['nome_correto']:
                    metricas.pessoas_nome_correto += 1
                if melhor_match['cpf_correto']:
                    metricas.pessoas_cpf_correto += 1
                if melhor_match['rg_correto']:
                    metricas.pessoas_rg_correto += 1
                if melhor_match['nome_cpf']:
                    metricas.pessoas_nome_cpf_corretas += 1
                if melhor_match['nome_rg']:
                    metricas.pessoas_nome_rg_corretas += 1
                    metricas.socios_corretos += 1
                if melhor_match['cpf_rg']:
                    metricas.pessoas_cpf_rg_corretas += 1
                if melhor_match['completo']:
                    metricas.pessoas_completas += 1
                    pessoas_corretas_count += 1

            if not encontrada:
                metricas.pessoas_faltantes += 1
        
        metricas.pessoas_extras = len(pessoas_resp) - len(pessoas_resp_processadas)
        
        # ===== NOVAS MÉTRICAS COMBINADAS EMPRESA+SÓCIO =====
        if len(empresas_corretas_indices) > 0:
            if metricas.pessoas_nome_cpf_corretas > 0:
                metricas.empresa_socio_nome_cpf = metricas.pessoas_nome_cpf_corretas
            if metricas.pessoas_nome_rg_corretas > 0:
                metricas.empresa_socio_nome_rg = metricas.pessoas_nome_rg_corretas
            if metricas.pessoas_completas > 0:
                metricas.empresa_socio_nome_cpf_rg = metricas.pessoas_completas
        
        # ===== MÉTRICA GLOBAL - VALIDAÇÃO COMPLETA (ENCONTROU TUDO) =====
        if (metricas.empresas_corretas > 0 and 
            metricas.pessoas_completas == metricas.total_pessoas_gabarito and
            metricas.pessoas_faltantes == 0 and
            metricas.pessoas_extras == 0):
            metricas.documentos_completos = 1
            metricas.empresas_com_socios_completos = metricas.empresas_corretas
        elif metricas.empresas_corretas > 0 and metricas.pessoas_completas > 0:
            metricas.documentos_parciais = 1
            metricas.empresas_com_socios_parciais = 1
        
        # ===== NOVA LÓGICA: VALIDAÇÃO RIGOROSA POR DOCUMENTO =====
        # O documento passou se:
        # 1. TODAS as empresas foram acertadas
        # 2. Não há empresas extras/faltantes
        # 3. TODOS os sócios conforme gabarito (nome + identificador/aux esperados)
        # 4. Não há sócios extras/faltantes
        
        documento_passou = (
            metricas.total_empresas_gabarito > 0 and
            metricas.total_empresas_gabarito == metricas.empresas_corretas and
            metricas.empresas_extras == 0 and
            metricas.empresas_faltantes == 0 and
            metricas.total_pessoas_gabarito > 0 and
            metricas.total_pessoas_gabarito == metricas.pessoas_completas and
            metricas.pessoas_extras == 0 and
            metricas.pessoas_faltantes == 0
        )
        
        metricas.documento_todas_empresas_socios = documento_passou
        
        return metricas
    
    def validar_todos_documentos(self):
        """Valida todos os documentos"""
        todos_ids = set(self.gabarito.keys()) | set(self.respostas.keys())
        
        for doc_id in sorted(todos_ids):
            metricas_doc = self.validar_documento(doc_id)
            self.metricas_por_id[doc_id] = metricas_doc
            
            # Acumula globalmente
            self.metricas_globais.total_empresas_gabarito += metricas_doc.total_empresas_gabarito
            self.metricas_globais.total_empresas_resposta += metricas_doc.total_empresas_resposta
            self.metricas_globais.empresas_corretas += metricas_doc.empresas_corretas
            self.metricas_globais.empresas_nomes_corretos += metricas_doc.empresas_nomes_corretos
            self.metricas_globais.empresas_cnpjs_corretos += metricas_doc.empresas_cnpjs_corretos
            self.metricas_globais.empresas_extras += metricas_doc.empresas_extras
            self.metricas_globais.empresas_faltantes += metricas_doc.empresas_faltantes
            
            self.metricas_globais.total_pessoas_gabarito += metricas_doc.total_pessoas_gabarito
            self.metricas_globais.total_pessoas_resposta += metricas_doc.total_pessoas_resposta
            self.metricas_globais.pessoas_nome_cpf_corretas += metricas_doc.pessoas_nome_cpf_corretas
            self.metricas_globais.pessoas_nome_rg_corretas += metricas_doc.pessoas_nome_rg_corretas
            self.metricas_globais.pessoas_cpf_rg_corretas += metricas_doc.pessoas_cpf_rg_corretas
            self.metricas_globais.pessoas_completas += metricas_doc.pessoas_completas
            self.metricas_globais.pessoas_nome_correto += metricas_doc.pessoas_nome_correto
            self.metricas_globais.pessoas_cpf_correto += metricas_doc.pessoas_cpf_correto
            self.metricas_globais.pessoas_rg_correto += metricas_doc.pessoas_rg_correto
            self.metricas_globais.pessoas_extras += metricas_doc.pessoas_extras
            self.metricas_globais.pessoas_faltantes += metricas_doc.pessoas_faltantes
            
            # Novas métricas combinadas
            self.metricas_globais.socios_corretos += metricas_doc.socios_corretos
            self.metricas_globais.empresa_socio_nome_cpf += metricas_doc.empresa_socio_nome_cpf
            self.metricas_globais.empresa_socio_nome_rg += metricas_doc.empresa_socio_nome_rg
            self.metricas_globais.empresa_socio_nome_cpf_rg += metricas_doc.empresa_socio_nome_cpf_rg
            
            # Métricas globais
            self.metricas_globais.documentos_completos += metricas_doc.documentos_completos
            self.metricas_globais.documentos_parciais += metricas_doc.documentos_parciais
            self.metricas_globais.empresas_com_socios_completos += metricas_doc.empresas_com_socios_completos
            self.metricas_globais.empresas_com_socios_parciais += metricas_doc.empresas_com_socios_parciais
            
            # ===== RASTREAMENTO PARA NOVA MÉTRICA =====
            if metricas_doc.documento_todas_empresas_socios:
                self.documentos_validacao_completa.append(doc_id)
            
            self.metricas_validacao_completa_por_doc[doc_id] = {
                'passou': metricas_doc.documento_todas_empresas_socios,
                'empresas_gab': metricas_doc.total_empresas_gabarito,
                'empresas_resp': metricas_doc.total_empresas_resposta,
                'empresas_corretas': metricas_doc.empresas_corretas,
                'pessoas_gab': metricas_doc.total_pessoas_gabarito,
                'pessoas_resp': metricas_doc.total_pessoas_resposta,
                'pessoas_completas': metricas_doc.pessoas_completas
            }
    
    def calcular_metricas(self, metricas: MetricasEntidade) -> Dict[str, float]:
        """Calcula Precisão, Recall, F1-Score e Acurácia"""
        resultado = {}
        
        # ===== EMPRESAS =====
        if metricas.total_empresas_resposta > 0:
            resultado['precisao_empresas'] = metricas.empresas_corretas / metricas.total_empresas_resposta
            resultado['precisao_empresas_nomes'] = metricas.empresas_nomes_corretos / metricas.total_empresas_resposta
            resultado['precisao_empresas_cnpjs'] = metricas.empresas_cnpjs_corretos / metricas.total_empresas_resposta
        else:
            resultado['precisao_empresas'] = 0.0
            resultado['precisao_empresas_nomes'] = 0.0
            resultado['precisao_empresas_cnpjs'] = 0.0
        
        if metricas.total_empresas_gabarito > 0:
            resultado['recall_empresas'] = metricas.empresas_corretas / metricas.total_empresas_gabarito
            resultado['recall_empresas_nomes'] = metricas.empresas_nomes_corretos / metricas.total_empresas_gabarito
            resultado['recall_empresas_cnpjs'] = metricas.empresas_cnpjs_corretos / metricas.total_empresas_gabarito
        else:
            resultado['recall_empresas'] = 0.0
            resultado['recall_empresas_nomes'] = 0.0
            resultado['recall_empresas_cnpjs'] = 0.0
        
        if resultado['precisao_empresas'] + resultado['recall_empresas'] > 0:
            resultado['f1_empresas'] = 2 * (resultado['precisao_empresas'] * resultado['recall_empresas']) / \
                                      (resultado['precisao_empresas'] + resultado['recall_empresas'])
        else:
            resultado['f1_empresas'] = 0.0
        
        total_max = max(metricas.total_empresas_gabarito, metricas.total_empresas_resposta)
        resultado['acuracia_empresas'] = metricas.empresas_corretas / total_max if total_max > 0 else 0.0
        
        # ===== PESSOAS =====
        if metricas.total_pessoas_resposta > 0:
            resultado['precisao_pessoas_nome_cpf'] = metricas.pessoas_nome_cpf_corretas / metricas.total_pessoas_resposta
            resultado['precisao_pessoas_nome_rg'] = metricas.pessoas_nome_rg_corretas / metricas.total_pessoas_resposta
            resultado['precisao_pessoas_completa'] = metricas.pessoas_completas / metricas.total_pessoas_resposta
            resultado['precisao_pessoas_nome'] = metricas.pessoas_nome_correto / metricas.total_pessoas_resposta
            resultado['precisao_pessoas_cpf'] = metricas.pessoas_cpf_correto / metricas.total_pessoas_resposta
            resultado['precisao_pessoas_rg'] = metricas.pessoas_rg_correto / metricas.total_pessoas_resposta
        else:
            resultado['precisao_pessoas_nome_cpf'] = 0.0
            resultado['precisao_pessoas_nome_rg'] = 0.0
            resultado['precisao_pessoas_completa'] = 0.0
            resultado['precisao_pessoas_nome'] = 0.0
            resultado['precisao_pessoas_cpf'] = 0.0
            resultado['precisao_pessoas_rg'] = 0.0
        
        if metricas.total_pessoas_gabarito > 0:
            resultado['recall_pessoas_nome_cpf'] = metricas.pessoas_nome_cpf_corretas / metricas.total_pessoas_gabarito
            resultado['recall_pessoas_nome_rg'] = metricas.pessoas_nome_rg_corretas / metricas.total_pessoas_gabarito
            resultado['recall_pessoas_completa'] = metricas.pessoas_completas / metricas.total_pessoas_gabarito
            resultado['recall_pessoas_nome'] = metricas.pessoas_nome_correto / metricas.total_pessoas_gabarito
            resultado['recall_pessoas_cpf'] = metricas.pessoas_cpf_correto / metricas.total_pessoas_gabarito
            resultado['recall_pessoas_rg'] = metricas.pessoas_rg_correto / metricas.total_pessoas_gabarito
        else:
            resultado['recall_pessoas_nome_cpf'] = 0.0
            resultado['recall_pessoas_nome_rg'] = 0.0
            resultado['recall_pessoas_completa'] = 0.0
            resultado['recall_pessoas_nome'] = 0.0
            resultado['recall_pessoas_cpf'] = 0.0
            resultado['recall_pessoas_rg'] = 0.0
        
        # F1-Scores
        for combo in ['nome_cpf', 'nome_rg', 'completa']:
            prec_key = f'precisao_pessoas_{combo}'
            rec_key = f'recall_pessoas_{combo}'
            if resultado.get(prec_key, 0) + resultado.get(rec_key, 0) > 0:
                resultado[f'f1_pessoas_{combo}'] = 2 * (resultado[prec_key] * resultado[rec_key]) / \
                                                   (resultado[prec_key] + resultado[rec_key])
            else:
                resultado[f'f1_pessoas_{combo}'] = 0.0
        
        # Acurácia Pessoas
        total_pessoas_max = max(metricas.total_pessoas_gabarito, metricas.total_pessoas_resposta)
        resultado['acuracia_pessoas_nome_cpf'] = metricas.pessoas_nome_cpf_corretas / total_pessoas_max if total_pessoas_max > 0 else 0.0
        resultado['acuracia_pessoas_nome_rg'] = metricas.pessoas_nome_rg_corretas / total_pessoas_max if total_pessoas_max > 0 else 0.0
        resultado['acuracia_pessoas_completa'] = metricas.pessoas_completas / total_pessoas_max if total_pessoas_max > 0 else 0.0
        
        # ===== NOVAS MÉTRICAS COMBINADAS EMPRESA+SÓCIO =====
        if metricas.total_pessoas_resposta > 0:
            resultado['precisao_empresa_socio_nome_cpf'] = metricas.empresa_socio_nome_cpf / metricas.total_pessoas_resposta
            resultado['precisao_empresa_socio_nome_rg'] = metricas.empresa_socio_nome_rg / metricas.total_pessoas_resposta
            resultado['precisao_empresa_socio_completo'] = metricas.empresa_socio_nome_cpf_rg / metricas.total_pessoas_resposta
        else:
            resultado['precisao_empresa_socio_nome_cpf'] = 0.0
            resultado['precisao_empresa_socio_nome_rg'] = 0.0
            resultado['precisao_empresa_socio_completo'] = 0.0
        
        if metricas.total_pessoas_gabarito > 0:
            resultado['recall_empresa_socio_nome_cpf'] = metricas.empresa_socio_nome_cpf / metricas.total_pessoas_gabarito
            resultado['recall_empresa_socio_nome_rg'] = metricas.empresa_socio_nome_rg / metricas.total_pessoas_gabarito
            resultado['recall_empresa_socio_completo'] = metricas.empresa_socio_nome_cpf_rg / metricas.total_pessoas_gabarito
        else:
            resultado['recall_empresa_socio_nome_cpf'] = 0.0
            resultado['recall_empresa_socio_nome_rg'] = 0.0
            resultado['recall_empresa_socio_completo'] = 0.0
        
        # F1-Scores Combinadas
        for combo in ['nome_cpf', 'nome_rg', 'completo']:
            prec_key = f'precisao_empresa_socio_{combo}'
            rec_key = f'recall_empresa_socio_{combo}'
            if resultado[prec_key] + resultado[rec_key] > 0:
                resultado[f'f1_empresa_socio_{combo}'] = 2 * (resultado[prec_key] * resultado[rec_key]) / \
                                                         (resultado[prec_key] + resultado[rec_key])
            else:
                resultado[f'f1_empresa_socio_{combo}'] = 0.0
        
        # Acurácia Combinadas
        resultado['acuracia_empresa_socio_nome_cpf'] = metricas.empresa_socio_nome_cpf / total_pessoas_max if total_pessoas_max > 0 else 0.0
        resultado['acuracia_empresa_socio_nome_rg'] = metricas.empresa_socio_nome_rg / total_pessoas_max if total_pessoas_max > 0 else 0.0
        resultado['acuracia_empresa_socio_completo'] = metricas.empresa_socio_nome_cpf_rg / total_pessoas_max if total_pessoas_max > 0 else 0.0
        
        # ===== MÉTRICAS GLOBAIS COMBINADAS =====
        total_entidades_gab = metricas.total_empresas_gabarito + metricas.total_pessoas_gabarito
        total_entidades_resp = metricas.total_empresas_resposta + metricas.total_pessoas_resposta
        total_corretas = metricas.empresas_corretas + metricas.pessoas_completas
        
        if total_entidades_resp > 0:
            resultado['precisao_global'] = total_corretas / total_entidades_resp
        else:
            resultado['precisao_global'] = 0.0
        
        if total_entidades_gab > 0:
            resultado['recall_global'] = total_corretas / total_entidades_gab
        else:
            resultado['recall_global'] = 0.0
        
        if resultado['precisao_global'] + resultado['recall_global'] > 0:
            resultado['f1_global'] = 2 * (resultado['precisao_global'] * resultado['recall_global']) / \
                                    (resultado['precisao_global'] + resultado['recall_global'])
        else:
            resultado['f1_global'] = 0.0
        
        total_max_global = max(total_entidades_gab, total_entidades_resp)
        resultado['acuracia_global'] = total_corretas / total_max_global if total_max_global > 0 else 0.0
        
        # ===== MÉTRICA GLOBAL COMPLETA (ENCONTROU TUDO) =====
        if metricas.total_empresas_gabarito > 0:
            resultado['precisao_global_completa'] = \
                metricas.empresas_com_socios_completos / max(metricas.total_empresas_resposta, 1)
            
            resultado['recall_global_completa'] = \
                metricas.empresas_com_socios_completos / metricas.total_empresas_gabarito
            
            resultado['taxa_documentos_completos'] = \
                (metricas.empresas_com_socios_completos / metricas.total_empresas_gabarito) * 100
            
            resultado['taxa_documentos_parciais'] = \
                (metricas.empresas_com_socios_parciais / metricas.total_empresas_gabarito) * 100
        else:
            resultado['precisao_global_completa'] = 0.0
            resultado['recall_global_completa'] = 0.0
            resultado['taxa_documentos_completos'] = 0.0
            resultado['taxa_documentos_parciais'] = 0.0
        
        # F1-Score Global Completa
        if resultado['precisao_global_completa'] + resultado['recall_global_completa'] > 0:
            resultado['f1_global_completa'] = \
                2 * (resultado['precisao_global_completa'] * resultado['recall_global_completa']) / \
                (resultado['precisao_global_completa'] + resultado['recall_global_completa'])
        else:
            resultado['f1_global_completa'] = 0.0
        
        # Acurácia Global Completa
        total_possivel = max(metricas.total_empresas_gabarito, metricas.total_empresas_resposta)
        resultado['acuracia_global_completa'] = \
            metricas.empresas_com_socios_completos / total_possivel if total_possivel > 0 else 0.0
        
        return resultado
    
    def calcular_media_validacao_completa(self) -> Dict[str, float]:
        """
        Calcula a média de precisão, recall, F1-score para a métrica 
        'Todas Empresas + Todos Sócios' considerando todos os documentos
        """
        if not self.metricas_validacao_completa_por_doc:
            return {
                'documentos_processados': 0,
                'documentos_passaram_validacao_completa': 0,
                'taxa_documentos_completos': 0.0,
                'precisao_media': 0.0,
                'recall_media': 0.0,
                'f1_media': 0.0,
                'acuracia_media': 0.0,
                'documentos_lista': []
            }
        
        total_documentos = len(self.metricas_validacao_completa_por_doc)
        documentos_passaram = len(self.documentos_validacao_completa)
        
        # Cálculo de precisão, recall e F1 por documento
        precisoes = []
        recalls = []
        f1_scores = []
        acuracias = []
        
        for doc_id, metricas_doc in self.metricas_validacao_completa_por_doc.items():
            # Por documento: 1.0 se passou, 0.0 se não
            if metricas_doc['empresas_resp'] > 0:
                precisao_doc = 1.0 if metricas_doc['passou'] else 0.0
                precisoes.append(precisao_doc)
            
            if metricas_doc['empresas_gab'] > 0:
                recall_doc = 1.0 if metricas_doc['passou'] else 0.0
                recalls.append(recall_doc)
            
            f1_doc = 1.0 if metricas_doc['passou'] else 0.0
            f1_scores.append(f1_doc)
            
            acuracia_doc = 1.0 if metricas_doc['passou'] else 0.0
            acuracias.append(acuracia_doc)
        
        # Calcular médias
        precisao_media = sum(precisoes) / len(precisoes) if precisoes else 0.0
        recall_media = sum(recalls) / len(recalls) if recalls else 0.0
        f1_media = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        acuracia_media = sum(acuracias) / len(acuracias) if acuracias else 0.0
        
        # F1-score global baseado nas médias
        if precisao_media + recall_media > 0:
            f1_media_calculado = 2 * (precisao_media * recall_media) / (precisao_media + recall_media)
        else:
            f1_media_calculado = 0.0
        
        return {
            'documentos_processados': total_documentos,
            'documentos_passaram_validacao_completa': documentos_passaram,
            'taxa_documentos_completos': documentos_passaram / total_documentos if total_documentos > 0 else 0.0,
            'precisao_media': precisao_media,
            'recall_media': recall_media,
            'f1_media': f1_media_calculado,
            'acuracia_media': acuracia_media,
            'documentos_lista': self.documentos_validacao_completa
        }
    
    def gerar_relatorio_txt(self, arquivo_saida: str = None) -> str:
        """Gera relatório completo em TXT com métricas combinadas e globais"""
        linhas = []
        
        linhas.append("=" * 100)
        linhas.append("RELATÓRIO DE VALIDAÇÃO DE ENTIDADES POR ID - VERSÃO 2 COM NOVA MÉTRICA GLOBAL".center(100))
        linhas.append("=" * 100)
        linhas.append("")
        
        linhas.append(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        linhas.append("")
        
        linhas.append("ARQUIVOS ANALISADOS")
        linhas.append("-" * 50)
        linhas.append(f"Gabarito: {self.arquivo_gabarito}")
        linhas.append(f"Respostas: {self.arquivo_respostas}")
        linhas.append("")
        
        if self.config_params:
            linhas.append("PARÂMETROS DE CONFIGURAÇÃO")
            linhas.append("-" * 50)
            for param, valor in self.config_params.items():
                linhas.append(f"  {param}: {valor}")
            linhas.append("")

        linhas.append("PARÂMETROS DE VALIDAÇÃO")
        linhas.append("-" * 50)
        linhas.append(f"Tolerância Nome Empresa: {self.tolerancia_nome:.0%}")
        linhas.append(f"Tolerância Nome Sócio: {self.tolerancia_nome_socio:.0%}")
        linhas.append("")
        
        # ===== NOVA SEÇÃO: MÉTRICA GLOBAL "TODAS EMPRESAS + TODOS SÓCIOS" =====
        metricas_validacao_completa = self.calcular_media_validacao_completa()
        
        linhas.append("=" * 100)
        linhas.append("NOVA MÉTRICA GLOBAL: TODAS EMPRESAS + TODOS SÓCIOS (MÉDIA POR DOCUMENTO)".center(100))
        linhas.append("=" * 100)
        linhas.append("")
        
        linhas.append("DESCRIÇÃO:")
        linhas.append("  Esta métrica valida se TODOS os documentos foram completamente validados, ou seja:")
        linhas.append("  - TODAS as empresas do documento foram acertadas (Nome + Identificador/CNPJ)")
        linhas.append("  - TODOS os sócios batem com o gabarito: nome; identificador/aux iguais ao gold se preenchidos,")
        linhas.append("    ou vazios na resposta quando vazios no gold (não deve haver dado inventado)")
        linhas.append("  - Não há empresas ou sócios extras ou faltantes")
        linhas.append("")
        
        linhas.append("RESUMO:")
        linhas.append("-" * 50)
        linhas.append(f"Total de documentos processados: {metricas_validacao_completa['documentos_processados']}")
        linhas.append(f"Documentos que passaram na validação completa: {metricas_validacao_completa['documentos_passaram_validacao_completa']}")
        linhas.append(f"Taxa de documentos completamente validados: {metricas_validacao_completa['taxa_documentos_completos']:.2%}")
        linhas.append("")
        
        linhas.append("MÉTRICAS DE DESEMPENHO (MÉDIA ENTRE TODOS OS DOCUMENTOS):")
        linhas.append("-" * 50)
        linhas.append(f"Precisão Média: {metricas_validacao_completa['precisao_media']:.2%}")
        linhas.append(f"Recall Média: {metricas_validacao_completa['recall_media']:.2%}")
        linhas.append(f"F1-Score Médio: {metricas_validacao_completa['f1_media']:.2%}")
        linhas.append(f"Acurácia Média: {metricas_validacao_completa['acuracia_media']:.2%}")
        linhas.append("")
        
        if metricas_validacao_completa['documentos_lista']:
            linhas.append("DOCUMENTOS QUE PASSARAM NA VALIDAÇÃO COMPLETA:")
            linhas.append("-" * 50)
            for doc_id in metricas_validacao_completa['documentos_lista']:
                linhas.append(f"  ✓ {doc_id}")
            linhas.append("")
        
        # Métricas Globais
        linhas.append("=" * 100)
        linhas.append("MÉTRICAS GLOBAIS".center(100))
        linhas.append("=" * 100)
        linhas.append("")
        
        linhas.append("EMPRESAS")
        linhas.append("-" * 50)
        linhas.append(f"Total gabarito: {self.metricas_globais.total_empresas_gabarito} | "
                     f"Total resposta: {self.metricas_globais.total_empresas_resposta} | "
                     f"Corretas: {self.metricas_globais.empresas_corretas}")
        linhas.append("")
        
        linhas.append("PESSOAS (SÓCIOS)")
        linhas.append("-" * 50)
        linhas.append(f"Total gabarito: {self.metricas_globais.total_pessoas_gabarito} | "
                     f"Total resposta: {self.metricas_globais.total_pessoas_resposta}")
        linhas.append("")
        linhas.append("Combinações:")
        linhas.append(f"  • Nome + Identificador: {self.metricas_globais.pessoas_nome_cpf_corretas}")
        linhas.append(f"  • Nome + Aux: {self.metricas_globais.pessoas_nome_rg_corretas}")
        linhas.append(f"  • Conforme gabarito (nome + ID + aux esperados): {self.metricas_globais.pessoas_completas}")
        linhas.append("")
        
        linhas.append("=" * 100)
        linhas.append("MÉTRICAS COMBINADAS EMPRESA + SÓCIO (NOVO)".center(100))
        linhas.append("=" * 100)
        linhas.append("")
        
        linhas.append("Pares validados:")
        linhas.append(f"  • Empresa + Sócio (Nome+ID): {self.metricas_globais.empresa_socio_nome_cpf}")
        linhas.append(f"  • Empresa + Sócio (Nome+Aux): {self.metricas_globais.empresa_socio_nome_rg}")
        linhas.append(f"  • Empresa + Sócio (conforme gabarito): {self.metricas_globais.empresa_socio_nome_cpf_rg}")
        linhas.append("")
        
        metricas_calc = self.calcular_metricas(self.metricas_globais)
        
        linhas.append("=" * 100)
        linhas.append("MÉTRICAS DE DESEMPENHO".center(100))
        linhas.append("=" * 100)
        linhas.append("")
        
        linhas.append("EMPRESAS")
        linhas.append("-" * 50)
        linhas.append(f"Precisão: {metricas_calc['precisao_empresas']:.2%} | "
                     f"Recall: {metricas_calc['recall_empresas']:.2%} | "
                     f"F1: {metricas_calc['f1_empresas']:.2%} | "
                     f"Acurácia: {metricas_calc['acuracia_empresas']:.2%}")
        linhas.append("")
        
        linhas.append("PESSOAS")
        linhas.append("-" * 50)
        linhas.append("Nome + Identificador:")
        linhas.append(f"  Precisão: {metricas_calc['precisao_pessoas_nome_cpf']:.2%} | "
                     f"Recall: {metricas_calc['recall_pessoas_nome_cpf']:.2%} | "
                     f"F1: {metricas_calc['f1_pessoas_nome_cpf']:.2%}")
        linhas.append("Nome + Aux:")
        linhas.append(f"  Precisão: {metricas_calc['precisao_pessoas_nome_rg']:.2%} | "
                     f"Recall: {metricas_calc['recall_pessoas_nome_rg']:.2%} | "
                     f"F1: {metricas_calc['f1_pessoas_nome_rg']:.2%}")
        
        linhas.append("Conforme gabarito (Nome + ID + Aux):")
        linhas.append(f"  Precisão: {metricas_calc['precisao_pessoas_completa']:.2%} | "
                     f"Recall: {metricas_calc['recall_pessoas_completa']:.2%} | "
                     f"F1: {metricas_calc['f1_pessoas_completa']:.2%}")
        linhas.append("")
        
        linhas.append("=" * 100)
        linhas.append("MÉTRICAS COMBINADAS EMPRESA + SÓCIO".center(100))
        linhas.append("=" * 100)
        linhas.append("")
        
        linhas.append("Empresa (Nome+ID) + Sócio (Nome+ID):")
        linhas.append(f"  Precisão: {metricas_calc['precisao_empresa_socio_nome_cpf']:.2%} | "
                     f"Recall: {metricas_calc['recall_empresa_socio_nome_cpf']:.2%} | "
                     f"F1: {metricas_calc['f1_empresa_socio_nome_cpf']:.2%}")
        
        linhas.append("Empresa (Nome+ID) + Sócio (Nome+Aux):")
        linhas.append(f"  Precisão: {metricas_calc['precisao_empresa_socio_nome_rg']:.2%} | "
                     f"Recall: {metricas_calc['recall_empresa_socio_nome_rg']:.2%} | "
                     f"F1: {metricas_calc['f1_empresa_socio_nome_rg']:.2%}")
        
        linhas.append("Empresa (Nome+ID) + Sócio (Nome+ID+Aux):")
        linhas.append(f"  Precisão: {metricas_calc['precisao_empresa_socio_completo']:.2%} | "
                     f"Recall: {metricas_calc['recall_empresa_socio_completo']:.2%} | "
                     f"F1: {metricas_calc['f1_empresa_socio_completo']:.2%}")
        linhas.append("")
        
        linhas.append("=" * 100)
        linhas.append("MÉTRICA GLOBAL - VALIDAÇÃO COMPLETA: 'ENCONTROU TUDO'".center(100))
        linhas.append("=" * 100)
        linhas.append("")
        
        linhas.append("DEFINIÇÃO:")
        linhas.append("-" * 100)
        linhas.append("Esta métrica valida se o sistema identificou CORRETAMENTE e COMPLETAMENTE")
        linhas.append("todos os dados de um documento, considerando:")
        linhas.append("  • Empresa correta: Nome + Identificador exatos")
        linhas.append("  • Todos os sócios da empresa: conforme gabarito (identificador/aux vazios quando vazios no gold)")
        linhas.append("  • Zero erros: Sem faltantes, sem extras, sem parciais")
        linhas.append("")
        
        linhas.append("CONTADORES:")
        linhas.append("-" * 100)
        linhas.append(f"Empresas com TODOS os dados corretos: {self.metricas_globais.empresas_com_socios_completos}")
        linhas.append(f"Empresas com ALGUNS dados corretos: {self.metricas_globais.empresas_com_socios_parciais}")
        linhas.append(f"Total empresas gabarito: {self.metricas_globais.total_empresas_gabarito}")
        linhas.append(f"Total empresas resposta: {self.metricas_globais.total_empresas_resposta}")
        linhas.append("")
        
        linhas.append("RESULTADOS:")
        linhas.append("-" * 100)
        linhas.append(f"Precisão Global Completa:      {metricas_calc['precisao_global_completa']:.2%}")
        linhas.append(f"Recall Global Completo:        {metricas_calc['recall_global_completa']:.2%}")
        linhas.append(f"F1-Score Global Completo:      {metricas_calc['f1_global_completa']:.2%}")
        linhas.append(f"Acurácia Global Completa:      {metricas_calc['acuracia_global_completa']:.2%}")
        linhas.append("")
        linhas.append(f"Taxa de Documentos Completamente Validados:  {metricas_calc['taxa_documentos_completos']:.1f}%")
        linhas.append(f"Taxa de Documentos Parcialmente Validados:   {metricas_calc['taxa_documentos_parciais']:.1f}%")
        linhas.append("")
        
        linhas.append("=" * 100)
        linhas.append("MÉTRICAS POR DOCUMENTO".center(100))
        linhas.append("=" * 100)
        linhas.append("")
        
        for doc_id in sorted(self.metricas_por_id.keys()):
            metricas_doc = self.metricas_por_id[doc_id]
            metricas_doc_calc = self.calcular_metricas(metricas_doc)
            
            status_completo = "✓ PASSOU" if metricas_doc.documento_todas_empresas_socios else "✗ NÃO PASSOU"
            
            linhas.append(f"DOCUMENTO: {doc_id} [{status_completo}]")
            linhas.append("-" * 100)
            linhas.append("")
            
            linhas.append("CONTADORES - EMPRESAS:")
            linhas.append(f"  Total gabarito: {metricas_doc.total_empresas_gabarito} | " +
                         f"Total resposta: {metricas_doc.total_empresas_resposta} | " +
                         f"Corretas: {metricas_doc.empresas_corretas} | " +
                         f"Extras: {metricas_doc.empresas_extras} | " +
                         f"Faltantes: {metricas_doc.empresas_faltantes}")
            linhas.append("")
            
            linhas.append("CONTADORES - PESSOAS:")
            linhas.append(f"  Total gabarito: {metricas_doc.total_pessoas_gabarito} | " +
                         f"Total resposta: {metricas_doc.total_pessoas_resposta} | " +
                         f"Nome+ID: {metricas_doc.pessoas_nome_cpf_corretas} | " +
                         f"Nome+Aux: {metricas_doc.pessoas_nome_rg_corretas} | " +
                         f"Completos: {metricas_doc.pessoas_completas}")
            linhas.append("")
            
            linhas.append("COMBINADAS EMPRESA+PESSOA:")
            linhas.append(f"  Empresa+Pessoa(Nome+ID): {metricas_doc.empresa_socio_nome_cpf} | " +
                         f"Empresa+Pessoa(Nome+Aux): {metricas_doc.empresa_socio_nome_rg} | " +
                         f"Empresa+Pessoa(Completo): {metricas_doc.empresa_socio_nome_cpf_rg}")
            linhas.append("")
            
            linhas.append("VALIDAÇÃO COMPLETA:")
            linhas.append(f"  Com tudo correto: {metricas_doc.empresas_com_socios_completos} | " +
                         f"Com parcial: {metricas_doc.empresas_com_socios_parciais}")
            linhas.append("")
            
            linhas.append("MÉTRICAS:")
            linhas.append(f"  Empresa - Prec: {metricas_doc_calc['precisao_empresas']:.1%} | Recall: {metricas_doc_calc['recall_empresas']:.1%} | F1: {metricas_doc_calc['f1_empresas']:.1%}")
            linhas.append(f"  Pessoa(Nome+ID) - Prec: {metricas_doc_calc['precisao_pessoas_nome_cpf']:.1%} | Recall: {metricas_doc_calc['recall_pessoas_nome_cpf']:.1%} | F1: {metricas_doc_calc['f1_pessoas_nome_cpf']:.1%}")
            linhas.append(f"  Pessoa(Nome+Aux) - Prec: {metricas_doc_calc['precisao_pessoas_nome_rg']:.1%} | Recall: {metricas_doc_calc['recall_pessoas_nome_rg']:.1%} | F1: {metricas_doc_calc['f1_pessoas_nome_rg']:.1%}")
            linhas.append(f"  Empresa+Pessoa(Nome+ID) - Prec: {metricas_doc_calc['precisao_empresa_socio_nome_cpf']:.1%} | Recall: {metricas_doc_calc['recall_empresa_socio_nome_cpf']:.1%} | F1: {metricas_doc_calc['f1_empresa_socio_nome_cpf']:.1%}")
            linhas.append(f"  Global Completa - Prec: {metricas_doc_calc['precisao_global_completa']:.1%} | Recall: {metricas_doc_calc['recall_global_completa']:.1%} | Taxa: {metricas_doc_calc['taxa_documentos_completos']:.1f}%")
            linhas.append("")
        
        linhas.append("=" * 100)
        linhas.append("FIM DO RELATÓRIO".center(100))
        linhas.append("=" * 100)
        
        relatorio = "\n".join(linhas)
        
        if arquivo_saida:
            try:
                with open(arquivo_saida, 'w', encoding='utf-8') as f:
                    f.write(relatorio)
                print(f"\n✅ Relatório TXT exportado: {arquivo_saida}")
            except Exception as e:
                print(f"\n❌ Erro ao exportar TXT: {e}")
        
        return relatorio
    
    def exportar_json(self, arquivo_saida: str):
        """Exporta métricas em JSON"""
        metricas_validacao_completa = self.calcular_media_validacao_completa()
        
        relatorio = {
            'configuracao': {
                'arquivos': {'gabarito': self.arquivo_gabarito, 'respostas': self.arquivo_respostas},
                'parametros_sistema': self.config_params,
                'limiar_similaridade': self.limiar_similaridade,
                'tolerancias': {
                    'nome_empresa': self.tolerancia_nome,
                    'nome_socio': self.tolerancia_nome_socio,
                },
            },
            'metricas_globais': {
                'empresas': {
                    'total_gabarito': self.metricas_globais.total_empresas_gabarito,
                    'total_resposta': self.metricas_globais.total_empresas_resposta,
                    'corretas': self.metricas_globais.empresas_corretas,
                    'nomes_corretos': self.metricas_globais.empresas_nomes_corretos,
                    'cnpjs_corretos': self.metricas_globais.empresas_cnpjs_corretos,
                    'extras': self.metricas_globais.empresas_extras,
                    'faltantes': self.metricas_globais.empresas_faltantes
                },
                'pessoas': {
                    'total_gabarito': self.metricas_globais.total_pessoas_gabarito,
                    'total_resposta': self.metricas_globais.total_pessoas_resposta,
                    'nome_cpf_corretas': self.metricas_globais.pessoas_nome_cpf_corretas,
                    'nome_rg_corretas': self.metricas_globais.pessoas_nome_rg_corretas,
                    'cpf_rg_corretas': self.metricas_globais.pessoas_cpf_rg_corretas,
                    'completas': self.metricas_globais.pessoas_completas
                },
                'combinadas_empresa_socio': {
                    'empresa_socio_nome_cpf': self.metricas_globais.empresa_socio_nome_cpf,
                    'empresa_socio_nome_rg': self.metricas_globais.empresa_socio_nome_rg,
                    'empresa_socio_completo': self.metricas_globais.empresa_socio_nome_cpf_rg
                },
                'validacao_completa': {
                    'empresas_com_socios_completos': self.metricas_globais.empresas_com_socios_completos,
                    'empresas_com_socios_parciais': self.metricas_globais.empresas_com_socios_parciais
                },
                'nova_metrica_todas_empresas_socios': metricas_validacao_completa,
                'metricas': self.calcular_metricas(self.metricas_globais)
            },
            'metricas_por_documento': {}
        }
        
        for doc_id, metricas_doc in self.metricas_por_id.items():
            relatorio['metricas_por_documento'][doc_id] = {
                'passou_validacao_completa': metricas_doc.documento_todas_empresas_socios,
                'empresas': {'total_gabarito': metricas_doc.total_empresas_gabarito, 
                            'corretas': metricas_doc.empresas_corretas},
                'pessoas': {'total_gabarito': metricas_doc.total_pessoas_gabarito, 
                           'completas': metricas_doc.pessoas_completas},
                'combinadas_empresa_socio': {'nome_rg': metricas_doc.empresa_socio_nome_rg},
                'validacao_completa': {
                    'empresas_com_socios_completos': metricas_doc.empresas_com_socios_completos,
                    'empresas_com_socios_parciais': metricas_doc.empresas_com_socios_parciais
                },
                'metricas': self.calcular_metricas(metricas_doc)
            }
        
        try:
            with open(arquivo_saida, 'w', encoding='utf-8') as f:
                json.dump(relatorio, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Relatório JSON exportado: {arquivo_saida}")
        except Exception as e:
            print(f"\n❌ Erro ao exportar JSON: {e}")
    
    def executar(self, exportar_txt: str = None, exportar_json: str = None) -> bool:
        """Executa validação completa"""
        if not self.carregar_arquivos():
            return False
        
        print("\n📊 Validando documentos...")
        self.validar_todos_documentos()
        
        relatorio = self.gerar_relatorio_txt(exportar_txt)
        print("\n" + relatorio)
        
        if exportar_json:
            self.exportar_json(exportar_json)
        
        print("\n✅ Validação concluída!")
        return True


def main():
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Validador de Entidades por ID V2 com Nova Métrica Global')
    parser.add_argument('gabarito', help='Arquivo CSV do gabarito')
    parser.add_argument('respostas', help='Arquivo CSV das respostas')
    parser.add_argument('--exportar-txt', help='Exportar relatório TXT')
    parser.add_argument('--exportar-json', help='Exportar relatório JSON')
    parser.add_argument('--limiar', type=float, default=0.95, help='Limiar de similaridade (fallback para tolerâncias)')
    parser.add_argument('--tolerancia-nome', type=float, default=None,
                        help='Similaridade mínima (0-1) para nome de empresa')
    parser.add_argument('--tolerancia-nome-socio', type=float, default=None,
                        help='Similaridade mínima (0-1) para nome de sócio')

    args = parser.parse_args()

    config_params = {
        "MODELO": "llama3.3:70b",
        "CHUNK_SIZE": 150,
        "EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    }

    validador = ValidadorEntidadesID(
        args.gabarito,
        args.respostas,
        args.limiar,
        config_params,
        tolerancia_nome=args.tolerancia_nome,
        tolerancia_nome_socio=args.tolerancia_nome_socio,
    )
    
    if not validador.executar(args.exportar_txt, args.exportar_json):
        sys.exit(1)


if __name__ == "__main__":
    main()