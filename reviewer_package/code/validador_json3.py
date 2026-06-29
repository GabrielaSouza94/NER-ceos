#!/usr/bin/env python3
"""
Validador de Documentos JSON - VERSÃO 3 (NOVA MÉTRICA GLOBAL)
Compara arquivos de gabarito e respostas, calculando métricas de precisão e acurácia
para empresas e sócios com múltiplas combinações de validação.

NOVAS MELHORIAS:
- Métrica global "Todas Empresas + Todos Sócios": valida se TODAS as empresas do documento
  foram acertadas (nome+CNPJ) e TODOS os sócios batem com o gabarito (nome + CPF/RG: valor
  igual ao gold se o gold preenche; se o gold está vazio, a resposta deve estar vazia).
- Cálculo de média de precisão, recall e F1 entre todos os documentos processados
- Rastreamento detalhado por documento da validação completa
"""

import json
import argparse
import sys
import unicodedata
import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from difflib import SequenceMatcher
from datetime import datetime


def extrair_parametros_extracao_de_json(caminho_arquivo: str) -> Dict[str, Any]:
    """
    Lê parâmetros de extração embutidos num JSON (resposta ou relatório).
    Formatos aceites na raiz do objeto:
    - chave 'parametros_extracao' (dict)
    - chave 'configuracao' com subchave 'parametros_extracao' (dict)
    Ficheiros de extração atuais (apenas doc_id -> lista de empresas) não têm estas chaves → retorna {}.
    """
    try:
        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, Any] = {}
    pe = data.get("parametros_extracao")
    if isinstance(pe, dict):
        out.update(pe)
    cfg = data.get("configuracao")
    if isinstance(cfg, dict):
        nested = cfg.get("parametros_extracao")
        if isinstance(nested, dict):
            out.update(nested)
    return out


@dataclass
class MetricasEmpresa:
    """Classe para armazenar métricas de validação de empresas e sócios"""
    
    # ===== MÉTRICAS DE EMPRESAS =====
    total_empresas_gabarito: int = 0
    total_empresas_resposta: int = 0
    empresas_corretas: int = 0
    nomes_corretos: int = 0
    cnpjs_corretos: int = 0
    empresas_extras: int = 0
    empresas_faltantes: int = 0
    
    # ===== MÉTRICAS DE SÓCIOS =====
    total_socios_gabarito: int = 0
    total_socios_resposta: int = 0
    
    # Combinações de validação de sócios (individuais)
    socios_nome_cpf_corretos: int = 0
    socios_nome_rg_corretos: int = 0
    socios_cpf_rg_corretos: int = 0
    socios_completos: int = 0  # sócio 100% alinhado ao gold (nome + CPF/RG conforme esperado)
    
    # Métricas individuais de sócios
    socios_nome_correto: int = 0
    socios_cpf_correto: int = 0
    socios_rg_correto: int = 0
    
    socios_extras: int = 0
    socios_faltantes: int = 0
    
    # ===== MÉTRICAS COMBINADAS EMPRESA + SÓCIO =====
    empresa_socio_nome_cpf: int = 0                # Empresa (nome+CNPJ) + Sócio (nome+CPF)
    empresa_socio_nome_rg: int = 0                 # Empresa (nome+CNPJ) + Sócio (nome+RG)
    empresa_socio_nome_cpf_rg: int = 0             # Empresa correta + sócio 100% conforme gold (campos esperados)
    
    # Validação Completa por Empresa
    empresas_com_socios_completos: int = 0         # Empresa correta + TODOS sócios conforme gabarito
    empresas_com_socios_parciais: int = 0          # Empresa correta + ALGUNS sócios completos
    
    # ===== NOVA MÉTRICA: TODAS EMPRESAS + TODOS SÓCIOS =====
    documento_todas_empresas_socios: bool = False  # True se TODAS as empresas + TODOS sócios corretos
    documento_num_empresas_validadas: int = 0      # Contador de empresas validadas neste documento


class ValidadorDocumentos:
    """Classe principal para validar documentos JSON"""
    
    def __init__(self, arquivo_gabarito: str, arquivo_resposta: str,
                 tolerancia_nome: float = 1.0, tolerancia_nome_socio: float = 1.0,
                 config_params: Dict = None):
        """Inicializa o validador com os caminhos dos arquivos"""
        self.arquivo_gabarito = arquivo_gabarito
        self.arquivo_resposta = arquivo_resposta
        self.gabarito = {}
        self.resposta = {}
        self.tolerancia_nome = tolerancia_nome
        self.tolerancia_nome_socio = tolerancia_nome_socio
        self.config_params = config_params or {}
        self.metricas_por_id = {}
        self.metricas_globais = MetricasEmpresa()
        self.relatorio_txt = []
        
        # Rastreamento para métrica global "Todas Empresas + Todos Sócios"
        self.documentos_validacao_completa = []
        self.metricas_validacao_completa_por_doc = {}
    
    def carregar_arquivos(self) -> bool:
        """Carrega os arquivos JSON do gabarito e das respostas"""
        try:
            with open(self.arquivo_gabarito, 'r', encoding='utf-8') as f:
                self.gabarito = json.load(f)
            
            with open(self.arquivo_resposta, 'r', encoding='utf-8') as f:
                self.resposta = json.load(f)
            
            return True
        except FileNotFoundError as e:
            print(f"Erro: Arquivo não encontrado - {e}")
            return False
        except json.JSONDecodeError as e:
            print(f"Erro: Arquivo JSON inválido - {e}")
            return False
        except Exception as e:
            print(f"Erro ao carregar arquivos: {e}")
            return False
    
    def normalizar_texto(self, texto: str) -> str:
        """Normaliza texto removendo acentos, pontuação e espaços extras"""
        if not texto:
            return ""
        
        texto = ''.join(
            c for c in unicodedata.normalize('NFD', texto)
            if unicodedata.category(c) != 'Mn'
        )
        texto = re.sub(r"[^\w\s]", " ", texto)
        texto = texto.upper()
        texto = re.sub(r"\s+", " ", texto).strip()
        
        return texto
    
    def normalizar_cnpj(self, cnpj: str) -> str:
        """Normaliza CNPJ removendo caracteres especiais"""
        if not cnpj:
            return ""
        return ''.join(filter(str.isdigit, cnpj or ""))
    
    def normalizar_cpf(self, cpf: str) -> str:
        """Normaliza CPF removendo caracteres especiais"""
        if not cpf:
            return ""
        return ''.join(filter(str.isdigit, cpf or ""))
    
    def normalizar_rg(self, rg: str) -> str:
        """Normaliza RG removendo caracteres especiais mas mantendo letras"""
        if not rg:
            return ""
        rg = re.sub(r"[^\w]", "", rg or "")
        return rg.upper()
    
    def similaridade(self, a: str, b: str) -> float:
        """Calcula similaridade entre duas strings"""
        return SequenceMatcher(None, a, b).ratio()
    
    def limpar_extensoes_empresa(self, nome: str) -> str:
        """Remove sufixos e extensões irrelevantes de nomes de empresas"""
        if not nome:
            return ""
        
        extensoes_irrelevantes = [
            "LTDA", "L T D A", "LTDA ME",
            "EIRELI", "E I R E L I",
            "ME", "M E", "EPP", "E P P",
            "SA", "S A", "SOCIEDADE LTDA",
            "EMPRESA INDIVIDUAL", "MICROEMPRESA"
        ]
        
        nome_limpo = nome.upper()
        for termo in extensoes_irrelevantes:
            nome_limpo = nome_limpo.replace(termo, "")
        
        nome_limpo = ' '.join(nome_limpo.split())
        return nome_limpo.strip()
    
    def comparar_empresa(self, empresa_gab: Dict, empresa_resp: Dict) -> Tuple[bool, bool, bool]:
        """
        Compara uma empresa do gabarito com uma da resposta
        Returns: Tupla (empresa_correta, nome_correto, cnpj_correto)
        """
        nome_gab = self.normalizar_texto(empresa_gab.get('empresa', ''))
        nome_resp = self.normalizar_texto(empresa_resp.get('empresa', ''))
        
        nome_gab = self.limpar_extensoes_empresa(nome_gab)
        nome_resp = self.limpar_extensoes_empresa(nome_resp)
        
        cnpj_gab = self.normalizar_cnpj(empresa_gab.get('cnpj', ''))
        cnpj_resp = self.normalizar_cnpj(empresa_resp.get('cnpj', ''))
        
        nome_similaridade = self.similaridade(nome_gab, nome_resp)
        nome_correto = nome_similaridade >= self.tolerancia_nome
        
        cnpj_correto = cnpj_gab == cnpj_resp
        empresa_correta = nome_correto and cnpj_correto
        
        return empresa_correta, nome_correto, cnpj_correto
    
    def comparar_socio(self, socio_gab: Dict, socio_resp: Dict) -> Dict[str, bool]:
        """
        Compara um sócio do gabarito com um da resposta.
        CPF/RG: se o gold não tem valor, a resposta deve estar vazia nesse campo;
        se o gold tem valor, a resposta deve coincidir (após normalização).
        """
        nome_gab = self.normalizar_texto(socio_gab.get('nome', ''))
        nome_resp = self.normalizar_texto(socio_resp.get('nome', ''))
        
        cpf_gab = self.normalizar_cpf(socio_gab.get('cpf', ''))
        cpf_resp = self.normalizar_cpf(socio_resp.get('cpf', ''))
        
        rg_gab = self.normalizar_rg(socio_gab.get('rg', ''))
        rg_resp = self.normalizar_rg(socio_resp.get('rg', ''))
        
        nome_similaridade = self.similaridade(nome_gab, nome_resp) if nome_gab else 0
        nome_correto = nome_similaridade >= self.tolerancia_nome_socio and nome_gab != ""
        
        if cpf_gab == "":
            cpf_correto = cpf_resp == ""
        else:
            cpf_correto = cpf_gab == cpf_resp

        if rg_gab == "":
            rg_correto = rg_resp == ""
        else:
            rg_correto = rg_gab == rg_resp
        
        resultado = {
            'nome_correto': nome_correto,
            'cpf_correto': cpf_correto,
            'rg_correto': rg_correto,
            'nome_cpf': nome_correto and cpf_correto,
            'nome_rg': nome_correto and rg_correto,
            'cpf_rg': cpf_correto and rg_correto,
            'completo': nome_correto and cpf_correto and rg_correto,
            'tem_nome': nome_gab != "",
            'tem_cpf': cpf_gab != "",
            'tem_rg': rg_gab != ""
        }
        
        return resultado
    
    def validar_socios_empresa(self, socios_gab: List, socios_resp: List) -> Dict[str, Any]:
        """Valida os sócios de uma empresa com rastreamento de combinações"""
        resultado = {
            'total_gabarito': len(socios_gab),
            'total_resposta': len(socios_resp),
            'nome_cpf_corretos': 0,
            'nome_rg_corretos': 0,
            'cpf_rg_corretos': 0,
            'completos': 0,
            'nome_correto': 0,
            'cpf_correto': 0,
            'rg_correto': 0,
            'extras': 0,
            'faltantes': 0,
            'todos_completos': False,
            'combo_empresa_socio_nome_cpf': 0,
            'combo_empresa_socio_nome_rg': 0,
            'combo_empresa_socio_completo': 0,
        }
        
        socios_resp_processados = set()
        socios_completos_count = 0
        
        for socio_gab in socios_gab:
            socio_encontrado = False
            melhor_match = None
            melhor_score = 0
            melhor_index = -1
            
            for i, socio_resp in enumerate(socios_resp):
                if i in socios_resp_processados:
                    continue
                
                comparacao = self.comparar_socio(socio_gab, socio_resp)
                
                score = sum([
                    comparacao['nome_correto'] * 3,
                    comparacao['cpf_correto'] * 2,
                    comparacao['rg_correto'] * 1
                ])
                
                if score > melhor_score:
                    melhor_score = score
                    melhor_match = comparacao
                    melhor_index = i
            
            if melhor_match and melhor_score > 0:
                socios_resp_processados.add(melhor_index)
                socio_encontrado = True
                
                if melhor_match['nome_cpf']:
                    resultado['nome_cpf_corretos'] += 1
                    resultado['combo_empresa_socio_nome_cpf'] += 1
                if melhor_match['nome_rg']:
                    resultado['nome_rg_corretos'] += 1
                    resultado['combo_empresa_socio_nome_rg'] += 1
                if melhor_match['cpf_rg']:
                    resultado['cpf_rg_corretos'] += 1
                if melhor_match['completo']:
                    resultado['completos'] += 1
                    socios_completos_count += 1
                    resultado['combo_empresa_socio_completo'] += 1
                if melhor_match['nome_correto']:
                    resultado['nome_correto'] += 1
                if melhor_match['cpf_correto']:
                    resultado['cpf_correto'] += 1
                if melhor_match['rg_correto']:
                    resultado['rg_correto'] += 1
            
            if not socio_encontrado:
                resultado['faltantes'] += 1
        
        resultado['extras'] = len(socios_resp) - len(socios_resp_processados)
        
        if (socios_completos_count == len(socios_gab) and
            len(socios_gab) == len(socios_resp) and
            resultado['extras'] == 0 and
            resultado['faltantes'] == 0):
            resultado['todos_completos'] = True
        
        return resultado
    
    def validar_documento(self, doc_id: str) -> MetricasEmpresa:
        """Valida um documento específico com rastreamento completo"""
        metricas = MetricasEmpresa()
        
        empresas_gab = self.gabarito.get(doc_id, [])
        empresas_resp = self.resposta.get(doc_id, [])
        
        metricas.total_empresas_gabarito = len(empresas_gab)
        metricas.total_empresas_resposta = len(empresas_resp)
        
        empresas_resp_processadas = set()
        empresas_validadas_completas = 0
        
        for emp_gab in empresas_gab:
            empresa_encontrada = False
            
            for i, emp_resp in enumerate(empresas_resp):
                if i in empresas_resp_processadas:
                    continue
                
                empresa_correta, nome_correto, cnpj_correto = self.comparar_empresa(emp_gab, emp_resp)
                
                if nome_correto or cnpj_correto:
                    empresas_resp_processadas.add(i)
                    empresa_encontrada = True
                    
                    if empresa_correta:
                        metricas.empresas_corretas += 1
                    if nome_correto:
                        metricas.nomes_corretos += 1
                    if cnpj_correto:
                        metricas.cnpjs_corretos += 1
                    
                    # Validar sócios desta empresa
                    socios_gab = emp_gab.get('socios', [])
                    socios_resp = emp_resp.get('socios', [])
                    
                    resultado_socios = self.validar_socios_empresa(socios_gab, socios_resp)
                    
                    metricas.total_socios_gabarito += resultado_socios['total_gabarito']
                    metricas.total_socios_resposta += resultado_socios['total_resposta']
                    metricas.socios_nome_cpf_corretos += resultado_socios['nome_cpf_corretos']
                    metricas.socios_nome_rg_corretos += resultado_socios['nome_rg_corretos']
                    metricas.socios_cpf_rg_corretos += resultado_socios['cpf_rg_corretos']
                    metricas.socios_completos += resultado_socios['completos']
                    metricas.socios_nome_correto += resultado_socios['nome_correto']
                    metricas.socios_cpf_correto += resultado_socios['cpf_correto']
                    metricas.socios_rg_correto += resultado_socios['rg_correto']
                    metricas.socios_extras += resultado_socios['extras']
                    metricas.socios_faltantes += resultado_socios['faltantes']
                    
                    # Métricas combinadas
                    if empresa_correta:
                        metricas.empresa_socio_nome_cpf += resultado_socios['combo_empresa_socio_nome_cpf']
                        metricas.empresa_socio_nome_rg += resultado_socios['combo_empresa_socio_nome_rg']
                        metricas.empresa_socio_nome_cpf_rg += resultado_socios['combo_empresa_socio_completo']
                        
                        if empresa_correta and resultado_socios['todos_completos']:
                            metricas.empresas_com_socios_completos += 1
                            empresas_validadas_completas += 1
                        elif empresa_correta and resultado_socios['completos'] > 0:
                            metricas.empresas_com_socios_parciais += 1
                    
                    break
            
            if not empresa_encontrada:
                metricas.empresas_faltantes += 1
                socios_gab = emp_gab.get('socios', [])
                metricas.total_socios_gabarito += len(socios_gab)
                metricas.socios_faltantes += len(socios_gab)
        
        empresas_extras_indices = set(range(len(empresas_resp))) - empresas_resp_processadas
        metricas.empresas_extras = len(empresas_extras_indices)
        
        for idx in empresas_extras_indices:
            socios_extras = empresas_resp[idx].get('socios', [])
            metricas.total_socios_resposta += len(socios_extras)
            metricas.socios_extras += len(socios_extras)
        
        # Validação completa do documento
        documento_passou = (
            metricas.total_empresas_gabarito > 0 and
            metricas.total_empresas_gabarito == metricas.empresas_corretas and
            metricas.empresas_extras == 0 and
            metricas.empresas_faltantes == 0 and
            metricas.total_socios_gabarito > 0 and
            metricas.total_socios_gabarito == metricas.socios_completos and
            metricas.socios_extras == 0 and
            metricas.socios_faltantes == 0 and
            empresas_validadas_completas == metricas.total_empresas_gabarito
        )
        
        metricas.documento_todas_empresas_socios = documento_passou
        metricas.documento_num_empresas_validadas = empresas_validadas_completas
        
        return metricas
    
    def validar_todos_documentos(self):
        """Valida todos os documentos e calcula métricas globais"""
        todos_ids = set(self.gabarito.keys()) | set(self.resposta.keys())
        
        for doc_id in sorted(todos_ids):
            metricas_doc = self.validar_documento(doc_id)
            self.metricas_por_id[doc_id] = metricas_doc
            
            # Acumular métricas globais - Empresas
            self.metricas_globais.total_empresas_gabarito += metricas_doc.total_empresas_gabarito
            self.metricas_globais.total_empresas_resposta += metricas_doc.total_empresas_resposta
            self.metricas_globais.empresas_corretas += metricas_doc.empresas_corretas
            self.metricas_globais.nomes_corretos += metricas_doc.nomes_corretos
            self.metricas_globais.cnpjs_corretos += metricas_doc.cnpjs_corretos
            self.metricas_globais.empresas_extras += metricas_doc.empresas_extras
            self.metricas_globais.empresas_faltantes += metricas_doc.empresas_faltantes
            
            # Acumular métricas globais - Sócios
            self.metricas_globais.total_socios_gabarito += metricas_doc.total_socios_gabarito
            self.metricas_globais.total_socios_resposta += metricas_doc.total_socios_resposta
            self.metricas_globais.socios_nome_cpf_corretos += metricas_doc.socios_nome_cpf_corretos
            self.metricas_globais.socios_nome_rg_corretos += metricas_doc.socios_nome_rg_corretos
            self.metricas_globais.socios_cpf_rg_corretos += metricas_doc.socios_cpf_rg_corretos
            self.metricas_globais.socios_completos += metricas_doc.socios_completos
            self.metricas_globais.socios_nome_correto += metricas_doc.socios_nome_correto
            self.metricas_globais.socios_cpf_correto += metricas_doc.socios_cpf_correto
            self.metricas_globais.socios_rg_correto += metricas_doc.socios_rg_correto
            self.metricas_globais.socios_extras += metricas_doc.socios_extras
            self.metricas_globais.socios_faltantes += metricas_doc.socios_faltantes
            
            # Acumular métricas combinadas globais
            self.metricas_globais.empresa_socio_nome_cpf += metricas_doc.empresa_socio_nome_cpf
            self.metricas_globais.empresa_socio_nome_rg += metricas_doc.empresa_socio_nome_rg
            self.metricas_globais.empresa_socio_nome_cpf_rg += metricas_doc.empresa_socio_nome_cpf_rg
            
            # Acumular métricas de validação completa
            self.metricas_globais.empresas_com_socios_completos += metricas_doc.empresas_com_socios_completos
            self.metricas_globais.empresas_com_socios_parciais += metricas_doc.empresas_com_socios_parciais
            
            # Rastreamento para nova métrica global
            if metricas_doc.documento_todas_empresas_socios:
                self.documentos_validacao_completa.append(doc_id)
            
            self.metricas_validacao_completa_por_doc[doc_id] = {
                'passou': metricas_doc.documento_todas_empresas_socios,
                'empresas_gab': metricas_doc.total_empresas_gabarito,
                'empresas_resp': metricas_doc.total_empresas_resposta,
                'empresas_corretas': metricas_doc.empresas_corretas,
                'socios_gab': metricas_doc.total_socios_gabarito,
                'socios_resp': metricas_doc.total_socios_resposta,
                'socios_completos': metricas_doc.socios_completos,
                'empresas_com_tudo': metricas_doc.empresas_com_socios_completos
            }
    
    def calcular_metricas(self, metricas: MetricasEmpresa) -> Dict[str, float]:
        """Calcula precisão, recall e F1-score para todas as combinações"""
        resultado = {}
        
        # ===== MÉTRICAS DE EMPRESAS =====
        if metricas.total_empresas_resposta > 0:
            resultado['precisao_empresas'] = metricas.empresas_corretas / metricas.total_empresas_resposta
            resultado['precisao_nomes'] = metricas.nomes_corretos / metricas.total_empresas_resposta
            resultado['precisao_cnpjs'] = metricas.cnpjs_corretos / metricas.total_empresas_resposta
        else:
            resultado['precisao_empresas'] = 0.0
            resultado['precisao_nomes'] = 0.0
            resultado['precisao_cnpjs'] = 0.0
        
        if metricas.total_empresas_gabarito > 0:
            resultado['recall_empresas'] = metricas.empresas_corretas / metricas.total_empresas_gabarito
            resultado['recall_nomes'] = metricas.nomes_corretos / metricas.total_empresas_gabarito
            resultado['recall_cnpjs'] = metricas.cnpjs_corretos / metricas.total_empresas_gabarito
        else:
            resultado['recall_empresas'] = 0.0
            resultado['recall_nomes'] = 0.0
            resultado['recall_cnpjs'] = 0.0
        
        if resultado['precisao_empresas'] + resultado['recall_empresas'] > 0:
            resultado['f1_empresas'] = 2 * (resultado['precisao_empresas'] * resultado['recall_empresas']) / \
                                      (resultado['precisao_empresas'] + resultado['recall_empresas'])
        else:
            resultado['f1_empresas'] = 0.0
        
        total_possivel = max(metricas.total_empresas_gabarito, metricas.total_empresas_resposta)
        if total_possivel > 0:
            resultado['acuracia_empresas'] = metricas.empresas_corretas / total_possivel
        else:
            resultado['acuracia_empresas'] = 0.0
        
        # ===== MÉTRICAS DE SÓCIOS =====
        if metricas.total_socios_resposta > 0:
            resultado['precisao_socios_nome_cpf'] = metricas.socios_nome_cpf_corretos / metricas.total_socios_resposta
            resultado['precisao_socios_nome_rg'] = metricas.socios_nome_rg_corretos / metricas.total_socios_resposta
            resultado['precisao_socios_cpf_rg'] = metricas.socios_cpf_rg_corretos / metricas.total_socios_resposta
            resultado['precisao_socios_completo'] = metricas.socios_completos / metricas.total_socios_resposta
            resultado['precisao_socios_nome'] = metricas.socios_nome_correto / metricas.total_socios_resposta
            resultado['precisao_socios_cpf'] = metricas.socios_cpf_correto / metricas.total_socios_resposta
            resultado['precisao_socios_rg'] = metricas.socios_rg_correto / metricas.total_socios_resposta
        else:
            resultado['precisao_socios_nome_cpf'] = 0.0
            resultado['precisao_socios_nome_rg'] = 0.0
            resultado['precisao_socios_cpf_rg'] = 0.0
            resultado['precisao_socios_completo'] = 0.0
            resultado['precisao_socios_nome'] = 0.0
            resultado['precisao_socios_cpf'] = 0.0
            resultado['precisao_socios_rg'] = 0.0
        
        if metricas.total_socios_gabarito > 0:
            resultado['recall_socios_nome_cpf'] = metricas.socios_nome_cpf_corretos / metricas.total_socios_gabarito
            resultado['recall_socios_nome_rg'] = metricas.socios_nome_rg_corretos / metricas.total_socios_gabarito
            resultado['recall_socios_cpf_rg'] = metricas.socios_cpf_rg_corretos / metricas.total_socios_gabarito
            resultado['recall_socios_completo'] = metricas.socios_completos / metricas.total_socios_gabarito
            resultado['recall_socios_nome'] = metricas.socios_nome_correto / metricas.total_socios_gabarito
            resultado['recall_socios_cpf'] = metricas.socios_cpf_correto / metricas.total_socios_gabarito
            resultado['recall_socios_rg'] = metricas.socios_rg_correto / metricas.total_socios_gabarito
        else:
            resultado['recall_socios_nome_cpf'] = 0.0
            resultado['recall_socios_nome_rg'] = 0.0
            resultado['recall_socios_cpf_rg'] = 0.0
            resultado['recall_socios_completo'] = 0.0
            resultado['recall_socios_nome'] = 0.0
            resultado['recall_socios_cpf'] = 0.0
            resultado['recall_socios_rg'] = 0.0
        
        for combo in ['nome_cpf', 'nome_rg', 'completo']:
            prec_key = f'precisao_socios_{combo}'
            rec_key = f'recall_socios_{combo}'
            if resultado[prec_key] + resultado[rec_key] > 0:
                resultado[f'f1_socios_{combo}'] = 2 * (resultado[prec_key] * resultado[rec_key]) / \
                                                  (resultado[prec_key] + resultado[rec_key])
            else:
                resultado[f'f1_socios_{combo}'] = 0.0
        
        total_socios_possivel = max(metricas.total_socios_gabarito, metricas.total_socios_resposta)
        if total_socios_possivel > 0:
            resultado['acuracia_socios_nome_cpf'] = metricas.socios_nome_cpf_corretos / total_socios_possivel
            resultado['acuracia_socios_nome_rg'] = metricas.socios_nome_rg_corretos / total_socios_possivel
            resultado['acuracia_socios_completo'] = metricas.socios_completos / total_socios_possivel
        else:
            resultado['acuracia_socios_nome_cpf'] = 0.0
            resultado['acuracia_socios_nome_rg'] = 0.0
            resultado['acuracia_socios_completo'] = 0.0
        
        # ===== NOVAS MÉTRICAS COMBINADAS EMPRESA+SÓCIO =====
        if metricas.total_socios_gabarito > 0:
            resultado['precisao_empresa_socio_nome_cpf'] = \
                metricas.empresa_socio_nome_cpf / max(metricas.total_socios_resposta, 1)
            resultado['precisao_empresa_socio_nome_rg'] = \
                metricas.empresa_socio_nome_rg / max(metricas.total_socios_resposta, 1)
            resultado['precisao_empresa_socio_completo'] = \
                metricas.empresa_socio_nome_cpf_rg / max(metricas.total_socios_resposta, 1)
            
            resultado['recall_empresa_socio_nome_cpf'] = \
                metricas.empresa_socio_nome_cpf / metricas.total_socios_gabarito
            resultado['recall_empresa_socio_nome_rg'] = \
                metricas.empresa_socio_nome_rg / metricas.total_socios_gabarito
            resultado['recall_empresa_socio_completo'] = \
                metricas.empresa_socio_nome_cpf_rg / metricas.total_socios_gabarito
        else:
            resultado['precisao_empresa_socio_nome_cpf'] = 0.0
            resultado['precisao_empresa_socio_nome_rg'] = 0.0
            resultado['precisao_empresa_socio_completo'] = 0.0
            resultado['recall_empresa_socio_nome_cpf'] = 0.0
            resultado['recall_empresa_socio_nome_rg'] = 0.0
            resultado['recall_empresa_socio_completo'] = 0.0
        
        for combo in ['nome_cpf', 'nome_rg', 'completo']:
            prec_key = f'precisao_empresa_socio_{combo}'
            rec_key = f'recall_empresa_socio_{combo}'
            if resultado[prec_key] + resultado[rec_key] > 0:
                resultado[f'f1_empresa_socio_{combo}'] = \
                    2 * (resultado[prec_key] * resultado[rec_key]) / \
                    (resultado[prec_key] + resultado[rec_key])
            else:
                resultado[f'f1_empresa_socio_{combo}'] = 0.0
        
        if total_socios_possivel > 0:
            resultado['acuracia_empresa_socio_nome_cpf'] = \
                metricas.empresa_socio_nome_cpf / total_socios_possivel
            resultado['acuracia_empresa_socio_nome_rg'] = \
                metricas.empresa_socio_nome_rg / total_socios_possivel
            resultado['acuracia_empresa_socio_completo'] = \
                metricas.empresa_socio_nome_cpf_rg / total_socios_possivel
        else:
            resultado['acuracia_empresa_socio_nome_cpf'] = 0.0
            resultado['acuracia_empresa_socio_nome_rg'] = 0.0
            resultado['acuracia_empresa_socio_completo'] = 0.0
        
        # ===== MÉTRICA GLOBAL COMPLETAMENTE CORRIGIDA =====
        if metricas.total_empresas_gabarito > 0:
            resultado['precisao_global_completa'] = \
                metricas.empresas_com_socios_completos / max(metricas.total_empresas_resposta, 1)
            resultado['recall_global_completa'] = \
                metricas.empresas_com_socios_completos / metricas.total_empresas_gabarito
            resultado['taxa_empresas_completas'] = \
                metricas.empresas_com_socios_completos / metricas.total_empresas_gabarito
            resultado['taxa_empresas_parciais'] = \
                metricas.empresas_com_socios_parciais / metricas.total_empresas_gabarito
        else:
            resultado['precisao_global_completa'] = 0.0
            resultado['recall_global_completa'] = 0.0
            resultado['taxa_empresas_completas'] = 0.0
            resultado['taxa_empresas_parciais'] = 0.0
        
        if resultado['precisao_global_completa'] + resultado['recall_global_completa'] > 0:
            resultado['f1_global_completa'] = \
                2 * (resultado['precisao_global_completa'] * resultado['recall_global_completa']) / \
                (resultado['precisao_global_completa'] + resultado['recall_global_completa'])
        else:
            resultado['f1_global_completa'] = 0.0
        
        if total_possivel > 0:
            resultado['acuracia_global_completa'] = \
                metricas.empresas_com_socios_completos / total_possivel
        else:
            resultado['acuracia_global_completa'] = 0.0
        
        return resultado
    
    def calcular_media_validacao_completa(self) -> Dict[str, Any]:
        """Calcula a média de precisão, recall e F1 para 'Todas Empresas + Todos Sócios'"""
        if not self.metricas_validacao_completa_por_doc:
            return {
                'documentos_processados': 0,
                'documentos_passaram_validacao_completa': 0,
                'taxa_documentos_completos': 0.0,
                'precisao_media': 0.0,
                'recall_media': 0.0,
                'f1_media': 0.0,
                'acuracia_media': 0.0
            }
        
        total_documentos = len(self.metricas_validacao_completa_por_doc)
        documentos_passaram = len(self.documentos_validacao_completa)
        
        precisoes = []
        recalls = []
        f1_scores = []
        acuracias = []
        
        for doc_id, metricas_doc in self.metricas_validacao_completa_por_doc.items():
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
        
        precisao_media = sum(precisoes) / len(precisoes) if precisoes else 0.0
        recall_media = sum(recalls) / len(recalls) if recalls else 0.0
        f1_media = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        acuracia_media = sum(acuracias) / len(acuracias) if acuracias else 0.0
        
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
        """Gera relatório completo em formato TXT"""
        self.relatorio_txt = []
        
        self.relatorio_txt.append("=" * 100)
        self.relatorio_txt.append("RELATÓRIO DE VALIDAÇÃO DE DOCUMENTOS - VERSÃO 3 (NOVA MÉTRICA GLOBAL)".center(100))
        self.relatorio_txt.append("=" * 100)
        self.relatorio_txt.append("")
        
        self.relatorio_txt.append(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        self.relatorio_txt.append("")
        
        self.relatorio_txt.append("ARQUIVOS ANALISADOS")
        self.relatorio_txt.append("-" * 50)
        self.relatorio_txt.append(f"Arquivo Gabarito: {self.arquivo_gabarito}")
        self.relatorio_txt.append(f"Arquivo Resposta: {self.arquivo_resposta}")
        self.relatorio_txt.append("")
        
        if self.config_params:
            self.relatorio_txt.append("PARÂMETROS DE CONFIGURAÇÃO DO SISTEMA DE EXTRAÇÃO")
            self.relatorio_txt.append("-" * 50)
            for param, valor in self.config_params.items():
                self.relatorio_txt.append(f"{param}: {valor}")
            self.relatorio_txt.append("")
        
        self.relatorio_txt.append("PARÂMETROS DE VALIDAÇÃO")
        self.relatorio_txt.append("-" * 50)
        self.relatorio_txt.append(f"Tolerância Nome Empresa: {self.tolerancia_nome:.0%}")
        self.relatorio_txt.append(f"Tolerância Nome Sócio: {self.tolerancia_nome_socio:.0%}")
        self.relatorio_txt.append("")
        
        # MÉTRICAS GLOBAIS
        self.relatorio_txt.append("=" * 100)
        self.relatorio_txt.append("MÉTRICAS GLOBAIS (TODOS OS DOCUMENTOS)".center(100))
        self.relatorio_txt.append("=" * 100)
        self.relatorio_txt.append("")
        
        self.relatorio_txt.append("EMPRESAS")
        self.relatorio_txt.append("-" * 50)
        self.relatorio_txt.append(f"Total de empresas no gabarito: {self.metricas_globais.total_empresas_gabarito}")
        self.relatorio_txt.append(f"Total de empresas na resposta: {self.metricas_globais.total_empresas_resposta}")
        self.relatorio_txt.append(f"Empresas corretas (Nome + CNPJ): {self.metricas_globais.empresas_corretas}")
        self.relatorio_txt.append(f"Nomes corretos: {self.metricas_globais.nomes_corretos}")
        self.relatorio_txt.append(f"CNPJs corretos: {self.metricas_globais.cnpjs_corretos}")
        self.relatorio_txt.append(f"Empresas extras (falsos positivos): {self.metricas_globais.empresas_extras}")
        self.relatorio_txt.append(f"Empresas faltantes (falsos negativos): {self.metricas_globais.empresas_faltantes}")
        self.relatorio_txt.append("")
        
        self.relatorio_txt.append("SÓCIOS - CONTADORES")
        self.relatorio_txt.append("-" * 50)
        self.relatorio_txt.append(f"Total de sócios no gabarito: {self.metricas_globais.total_socios_gabarito}")
        self.relatorio_txt.append(f"Total de sócios na resposta: {self.metricas_globais.total_socios_resposta}")
        self.relatorio_txt.append("")
        
        self.relatorio_txt.append("Sócios validados individualmente:")
        self.relatorio_txt.append(f"  • Sócios com Nome + CPF corretos: {self.metricas_globais.socios_nome_cpf_corretos}")
        self.relatorio_txt.append(f"  • Sócios com Nome + RG corretos: {self.metricas_globais.socios_nome_rg_corretos}")
        self.relatorio_txt.append(f"  • Sócios com CPF + RG corretos: {self.metricas_globais.socios_cpf_rg_corretos}")
        self.relatorio_txt.append(f"  • Sócios 100% conforme gabarito (nome + CPF/RG esperados): {self.metricas_globais.socios_completos}")
        self.relatorio_txt.append("")
        
        self.relatorio_txt.append("Campos individuais:")
        self.relatorio_txt.append(f"  • Nomes de sócios corretos: {self.metricas_globais.socios_nome_correto}")
        self.relatorio_txt.append(f"  • Dimensão CPF correta (igual ao gold ou vazio nos dois): {self.metricas_globais.socios_cpf_correto}")
        self.relatorio_txt.append(f"  • Dimensão RG correta (igual ao gold ou vazio nos dois): {self.metricas_globais.socios_rg_correto}")
        self.relatorio_txt.append(f"  • Sócios extras: {self.metricas_globais.socios_extras}")
        self.relatorio_txt.append(f"  • Sócios faltantes: {self.metricas_globais.socios_faltantes}")
        self.relatorio_txt.append("")
        
        # MÉTRICAS COMBINADAS EMPRESA+SÓCIO
        self.relatorio_txt.append("=" * 100)
        self.relatorio_txt.append("MÉTRICAS COMBINADAS EMPRESA + SÓCIO".center(100))
        self.relatorio_txt.append("=" * 100)
        self.relatorio_txt.append("")
        
        self.relatorio_txt.append("Pares Empresa+Sócio validados corretamente:")
        self.relatorio_txt.append("-" * 50)
        self.relatorio_txt.append(f"Empresa (Nome+CNPJ) + Sócio (Nome+CPF): {self.metricas_globais.empresa_socio_nome_cpf}")
        self.relatorio_txt.append(f"Empresa (Nome+CNPJ) + Sócio (Nome+RG): {self.metricas_globais.empresa_socio_nome_rg}")
        self.relatorio_txt.append(f"Empresa (Nome+CNPJ) + Sócio conforme gabarito (todos campos esperados): {self.metricas_globais.empresa_socio_nome_cpf_rg}")
        self.relatorio_txt.append("")
        
        self.relatorio_txt.append("VALIDAÇÃO COMPLETA POR EMPRESA")
        self.relatorio_txt.append("-" * 50)
        self.relatorio_txt.append(f"Empresas com TODOS os dados corretos:")
        self.relatorio_txt.append(f"  (Empresa correta + TODOS sócios 100% conforme gabarito): {self.metricas_globais.empresas_com_socios_completos}")
        self.relatorio_txt.append(f"Empresas com ALGUNS dados de sócios corretos:")
        self.relatorio_txt.append(f"  (Empresa correta + ALGUNS sócios 100% conforme gabarito): {self.metricas_globais.empresas_com_socios_parciais}")
        self.relatorio_txt.append("")
        
        # NOVA MÉTRICA GLOBAL
        metricas_validacao_completa = self.calcular_media_validacao_completa()
        
        self.relatorio_txt.append("=" * 100)
        self.relatorio_txt.append("NOVA MÉTRICA GLOBAL: TODAS EMPRESAS + TODOS SÓCIOS (MÉDIA POR DOCUMENTO)".center(100))
        self.relatorio_txt.append("=" * 100)
        self.relatorio_txt.append("")
        
        self.relatorio_txt.append("DESCRIÇÃO:")
        self.relatorio_txt.append("  Esta métrica valida se TODOS os documentos foram completamente validados, ou seja:")
        self.relatorio_txt.append("  - TODAS as empresas do documento foram acertadas (Nome + CNPJ)")
        self.relatorio_txt.append("  - TODOS os sócios batem com o gabarito: nome; CPF/RG iguais ao gold se preenchidos no gold,")
        self.relatorio_txt.append("    ou vazios na resposta quando vazios no gold (não deve haver dado inventado)")
        self.relatorio_txt.append("  - Não há empresas ou sócios extras ou faltantes")
        self.relatorio_txt.append("")
        
        self.relatorio_txt.append("RESUMO:")
        self.relatorio_txt.append("-" * 50)
        self.relatorio_txt.append(f"Total de documentos processados: {metricas_validacao_completa['documentos_processados']}")
        self.relatorio_txt.append(f"Documentos que passaram na validação completa: {metricas_validacao_completa['documentos_passaram_validacao_completa']}")
        self.relatorio_txt.append(f"Taxa de documentos completamente validados: {metricas_validacao_completa['taxa_documentos_completos']:.2%}")
        self.relatorio_txt.append("")
        
        self.relatorio_txt.append("MÉTRICAS DE DESEMPENHO (MÉDIA ENTRE TODOS OS DOCUMENTOS):")
        self.relatorio_txt.append("-" * 50)
        self.relatorio_txt.append(f"Precisão Média: {metricas_validacao_completa['precisao_media']:.2%}")
        self.relatorio_txt.append(f"Recall Média: {metricas_validacao_completa['recall_media']:.2%}")
        self.relatorio_txt.append(f"F1-Score Médio: {metricas_validacao_completa['f1_media']:.2%}")
        self.relatorio_txt.append(f"Acurácia Média: {metricas_validacao_completa['acuracia_media']:.2%}")
        self.relatorio_txt.append("")
        
        if metricas_validacao_completa['documentos_lista']:
            self.relatorio_txt.append("DOCUMENTOS QUE PASSARAM NA VALIDAÇÃO COMPLETA:")
            self.relatorio_txt.append("-" * 50)
            for doc_id in metricas_validacao_completa['documentos_lista']:
                self.relatorio_txt.append(f"  ✓ {doc_id}")
            self.relatorio_txt.append("")
        
        # MÉTRICAS DE DESEMPENHO
        metricas_calc_global = self.calcular_metricas(self.metricas_globais)
        
        self.relatorio_txt.append("=" * 100)
        self.relatorio_txt.append("MÉTRICAS DE DESEMPENHO GLOBAL".center(100))
        self.relatorio_txt.append("=" * 100)
        self.relatorio_txt.append("")
        
        self.relatorio_txt.append("EMPRESAS (Precisão, Recall, F1-Score, Acurácia)")
        self.relatorio_txt.append("-" * 50)
        self.relatorio_txt.append(f"Precisão: {metricas_calc_global['precisao_empresas']:.2%}  |  " +
                                f"Recall: {metricas_calc_global['recall_empresas']:.2%}  |  " +
                                f"F1-Score: {metricas_calc_global['f1_empresas']:.2%}  |  " +
                                f"Acurácia: {metricas_calc_global['acuracia_empresas']:.2%}")
        self.relatorio_txt.append("")
        
        self.relatorio_txt.append("SÓCIOS - COMBINAÇÕES")
        self.relatorio_txt.append("-" * 50)
        
        self.relatorio_txt.append("Nome + CPF:")
        self.relatorio_txt.append(f"  Precisão: {metricas_calc_global['precisao_socios_nome_cpf']:.2%}  |  " +
                                f"Recall: {metricas_calc_global['recall_socios_nome_cpf']:.2%}  |  " +
                                f"F1: {metricas_calc_global['f1_socios_nome_cpf']:.2%}  |  " +
                                f"Acurácia: {metricas_calc_global['acuracia_socios_nome_cpf']:.2%}")
        
        self.relatorio_txt.append("Nome + RG:")
        self.relatorio_txt.append(f"  Precisão: {metricas_calc_global['precisao_socios_nome_rg']:.2%}  |  " +
                                f"Recall: {metricas_calc_global['recall_socios_nome_rg']:.2%}  |  " +
                                f"F1: {metricas_calc_global['f1_socios_nome_rg']:.2%}  |  " +
                                f"Acurácia: {metricas_calc_global['acuracia_socios_nome_rg']:.2%}")
        
        self.relatorio_txt.append("Conforme gabarito (todos os campos esperados por sócio):")
        self.relatorio_txt.append(f"  Precisão: {metricas_calc_global['precisao_socios_completo']:.2%}  |  " +
                                f"Recall: {metricas_calc_global['recall_socios_completo']:.2%}  |  " +
                                f"F1: {metricas_calc_global['f1_socios_completo']:.2%}  |  " +
                                f"Acurácia: {metricas_calc_global['acuracia_socios_completo']:.2%}")
        self.relatorio_txt.append("")
        
        self.relatorio_txt.append("=" * 100)
        self.relatorio_txt.append("MÉTRICAS COMBINADAS EMPRESA + SÓCIO (Precisão, Recall, F1, Acurácia)".center(100))
        self.relatorio_txt.append("=" * 100)
        self.relatorio_txt.append("")
        
        self.relatorio_txt.append("Empresa (Nome+CNPJ) + Sócio (Nome+CPF):")
        self.relatorio_txt.append(f"  Precisão: {metricas_calc_global['precisao_empresa_socio_nome_cpf']:.2%}  |  " +
                                f"Recall: {metricas_calc_global['recall_empresa_socio_nome_cpf']:.2%}  |  " +
                                f"F1: {metricas_calc_global['f1_empresa_socio_nome_cpf']:.2%}  |  " +
                                f"Acurácia: {metricas_calc_global['acuracia_empresa_socio_nome_cpf']:.2%}")
        
        self.relatorio_txt.append("Empresa (Nome+CNPJ) + Sócio (Nome+RG):")
        self.relatorio_txt.append(f"  Precisão: {metricas_calc_global['precisao_empresa_socio_nome_rg']:.2%}  |  " +
                                f"Recall: {metricas_calc_global['recall_empresa_socio_nome_rg']:.2%}  |  " +
                                f"F1: {metricas_calc_global['f1_empresa_socio_nome_rg']:.2%}  |  " +
                                f"Acurácia: {metricas_calc_global['acuracia_empresa_socio_nome_rg']:.2%}")
        
        self.relatorio_txt.append("Empresa (Nome+CNPJ) + Sócio conforme gabarito:")
        self.relatorio_txt.append(f"  Precisão: {metricas_calc_global['precisao_empresa_socio_completo']:.2%}  |  " +
                                f"Recall: {metricas_calc_global['recall_empresa_socio_completo']:.2%}  |  " +
                                f"F1: {metricas_calc_global['f1_empresa_socio_completo']:.2%}  |  " +
                                f"Acurácia: {metricas_calc_global['acuracia_empresa_socio_completo']:.2%}")
        self.relatorio_txt.append("")
        
        self.relatorio_txt.append("=" * 100)
        self.relatorio_txt.append("MÉTRICA GLOBAL - VALIDAÇÃO COMPLETA (Por Empresa)".center(100))
        self.relatorio_txt.append("=" * 100)
        self.relatorio_txt.append("")
        
        self.relatorio_txt.append("'ENCONTROU TUDO': Empresa + todos sócios conforme gabarito (incl. CPF/RG vazios quando vazios no gold)")
        self.relatorio_txt.append("-" * 50)
        self.relatorio_txt.append(f"Precisão Global Completa: {metricas_calc_global['precisao_global_completa']:.2%}")
        self.relatorio_txt.append(f"Recall Global Completo: {metricas_calc_global['recall_global_completa']:.2%}")
        self.relatorio_txt.append(f"F1-Score Global Completo: {metricas_calc_global['f1_global_completa']:.2%}")
        self.relatorio_txt.append(f"Acurácia Global Completa: {metricas_calc_global['acuracia_global_completa']:.2%}")
        self.relatorio_txt.append("")
        self.relatorio_txt.append(f"Taxa de Empresas Completamente Validadas: {metricas_calc_global['taxa_empresas_completas']:.2%}")
        self.relatorio_txt.append(f"Taxa de Empresas Parcialmente Validadas: {metricas_calc_global['taxa_empresas_parciais']:.2%}")
        self.relatorio_txt.append("")
        
        # MÉTRICAS POR DOCUMENTO
        self.relatorio_txt.append("=" * 100)
        self.relatorio_txt.append("MÉTRICAS POR DOCUMENTO".center(100))
        self.relatorio_txt.append("=" * 100)
        self.relatorio_txt.append("")
        
        for doc_id in sorted(self.metricas_por_id.keys()):
            metricas_doc = self.metricas_por_id[doc_id]
            
            status_completo = "✓ PASSOU" if metricas_doc.documento_todas_empresas_socios else "✗ NÃO PASSOU"
            
            self.relatorio_txt.append(f"DOCUMENTO: {doc_id} [{status_completo}]")
            self.relatorio_txt.append("-" * 100)
            
            self.relatorio_txt.append("CONTADORES - EMPRESAS:")
            self.relatorio_txt.append(f"  Total gabarito: {metricas_doc.total_empresas_gabarito} | " +
                                    f"Total resposta: {metricas_doc.total_empresas_resposta} | " +
                                    f"Corretas: {metricas_doc.empresas_corretas} | " +
                                    f"Extras: {metricas_doc.empresas_extras} | " +
                                    f"Faltantes: {metricas_doc.empresas_faltantes}")
            
            self.relatorio_txt.append("CONTADORES - SÓCIOS:")
            self.relatorio_txt.append(f"  Total gabarito: {metricas_doc.total_socios_gabarito} | " +
                                    f"Total resposta: {metricas_doc.total_socios_resposta} | " +
                                    f"Nome+CPF: {metricas_doc.socios_nome_cpf_corretos} | " +
                                    f"Nome+RG: {metricas_doc.socios_nome_rg_corretos} | " +
                                    f"Conforme gabarito: {metricas_doc.socios_completos}")
            
            self.relatorio_txt.append("COMBINADAS EMPRESA+SÓCIO:")
            self.relatorio_txt.append(f"  Empresa+Sócio(Nome+CPF): {metricas_doc.empresa_socio_nome_cpf} | " +
                                    f"Empresa+Sócio(Nome+RG): {metricas_doc.empresa_socio_nome_rg} | " +
                                    f"Empresa+Sócio(conforme gabarito): {metricas_doc.empresa_socio_nome_cpf_rg}")
            
            self.relatorio_txt.append("VALIDAÇÃO COMPLETA:")
            self.relatorio_txt.append(f"  Com tudo correto: {metricas_doc.empresas_com_socios_completos} | " +
                                    f"Com parcial: {metricas_doc.empresas_com_socios_parciais}")
            
            metricas_calc_doc = self.calcular_metricas(metricas_doc)
            
            self.relatorio_txt.append("MÉTRICAS:")
            self.relatorio_txt.append(f"  Empresa - Prec: {metricas_calc_doc['precisao_empresas']:.1%} | Recall: {metricas_calc_doc['recall_empresas']:.1%} | F1: {metricas_calc_doc['f1_empresas']:.1%}")
            self.relatorio_txt.append(f"  Sócio(Nome+CPF) - Prec: {metricas_calc_doc['precisao_socios_nome_cpf']:.1%} | Recall: {metricas_calc_doc['recall_socios_nome_cpf']:.1%} | F1: {metricas_calc_doc['f1_socios_nome_cpf']:.1%}")
            self.relatorio_txt.append(f"  Sócio(Nome+RG) - Prec: {metricas_calc_doc['precisao_socios_nome_rg']:.1%} | Recall: {metricas_calc_doc['recall_socios_nome_rg']:.1%} | F1: {metricas_calc_doc['f1_socios_nome_rg']:.1%}")
            self.relatorio_txt.append(f"  Empresa+Sócio(Nome+CPF) - Prec: {metricas_calc_doc['precisao_empresa_socio_nome_cpf']:.1%} | Recall: {metricas_calc_doc['recall_empresa_socio_nome_cpf']:.1%}")
            self.relatorio_txt.append(f"  Global Completa - Prec: {metricas_calc_doc['precisao_global_completa']:.1%} | Recall: {metricas_calc_doc['recall_global_completa']:.1%} | Taxa: {metricas_calc_doc['taxa_empresas_completas']:.1%}")
            self.relatorio_txt.append("")
        
        self.relatorio_txt.append("=" * 100)
        self.relatorio_txt.append("FIM DO RELATÓRIO".center(100))
        self.relatorio_txt.append("=" * 100)
        
        relatorio_str = '\n'.join(self.relatorio_txt)
        
        if arquivo_saida:
            try:
                with open(arquivo_saida, 'w', encoding='utf-8') as f:
                    f.write(relatorio_str)
                print(f"\n✓ Relatório TXT exportado para: {arquivo_saida}")
            except Exception as e:
                print(f"\n✗ Erro ao exportar relatório TXT: {e}")
        
        return relatorio_str
    
    def gerar_relatorio(self):
        """Gera e imprime o relatório no console"""
        relatorio = self.gerar_relatorio_txt()
        print(relatorio)
    
    def exportar_relatorio_json(self, arquivo_saida: str):
        """Exporta o relatório em formato JSON"""
        metricas_validacao_completa = self.calcular_media_validacao_completa()
        
        relatorio = {
            'configuracao': {
                'arquivos': {
                    'gabarito': self.arquivo_gabarito,
                    'resposta': self.arquivo_resposta
                },
                'parametros_extracao': self.config_params,
                'tolerancias': {
                    'nome_empresa': self.tolerancia_nome,
                    'nome_socio': self.tolerancia_nome_socio
                }
            },
            'metricas_globais': {
                'contagens_empresas': {
                    'total_gabarito': self.metricas_globais.total_empresas_gabarito,
                    'total_resposta': self.metricas_globais.total_empresas_resposta,
                    'corretas': self.metricas_globais.empresas_corretas,
                    'nomes_corretos': self.metricas_globais.nomes_corretos,
                    'cnpjs_corretos': self.metricas_globais.cnpjs_corretos,
                    'extras': self.metricas_globais.empresas_extras,
                    'faltantes': self.metricas_globais.empresas_faltantes
                },
                'contagens_socios': {
                    'total_gabarito': self.metricas_globais.total_socios_gabarito,
                    'total_resposta': self.metricas_globais.total_socios_resposta,
                    'nome_cpf_corretos': self.metricas_globais.socios_nome_cpf_corretos,
                    'nome_rg_corretos': self.metricas_globais.socios_nome_rg_corretos,
                    'cpf_rg_corretos': self.metricas_globais.socios_cpf_rg_corretos,
                    'completos': self.metricas_globais.socios_completos,
                    'nome_correto': self.metricas_globais.socios_nome_correto,
                    'cpf_correto': self.metricas_globais.socios_cpf_correto,
                    'rg_correto': self.metricas_globais.socios_rg_correto,
                    'extras': self.metricas_globais.socios_extras,
                    'faltantes': self.metricas_globais.socios_faltantes
                },
                'contagens_combinadas_empresa_socio': {
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
                'contagens_empresas': {
                    'total_gabarito': metricas_doc.total_empresas_gabarito,
                    'total_resposta': metricas_doc.total_empresas_resposta,
                    'corretas': metricas_doc.empresas_corretas,
                    'nomes_corretos': metricas_doc.nomes_corretos,
                    'cnpjs_corretos': metricas_doc.cnpjs_corretos,
                    'extras': metricas_doc.empresas_extras,
                    'faltantes': metricas_doc.empresas_faltantes
                },
                'contagens_socios': {
                    'total_gabarito': metricas_doc.total_socios_gabarito,
                    'total_resposta': metricas_doc.total_socios_resposta,
                    'nome_cpf_corretos': metricas_doc.socios_nome_cpf_corretos,
                    'nome_rg_corretos': metricas_doc.socios_nome_rg_corretos,
                    'cpf_rg_corretos': metricas_doc.socios_cpf_rg_corretos,
                    'completos': metricas_doc.socios_completos,
                    'nome_correto': metricas_doc.socios_nome_correto,
                    'cpf_correto': metricas_doc.socios_cpf_correto,
                    'rg_correto': metricas_doc.socios_rg_correto,
                    'extras': metricas_doc.socios_extras,
                    'faltantes': metricas_doc.socios_faltantes
                },
                'contagens_combinadas_empresa_socio': {
                    'empresa_socio_nome_cpf': metricas_doc.empresa_socio_nome_cpf,
                    'empresa_socio_nome_rg': metricas_doc.empresa_socio_nome_rg,
                    'empresa_socio_completo': metricas_doc.empresa_socio_nome_cpf_rg
                },
                'validacao_completa': {
                    'empresas_com_socios_completos': metricas_doc.empresas_com_socios_completos,
                    'empresas_com_socios_parciais': metricas_doc.empresas_com_socios_parciais
                },
                'metricas': self.calcular_metricas(metricas_doc)
            }
        
        try:
            with open(arquivo_saida, 'w', encoding='utf-8') as f:
                json.dump(relatorio, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Relatório JSON exportado para: {arquivo_saida}")
        except Exception as e:
            print(f"\n✗ Erro ao exportar relatório JSON: {e}")
    
    def executar(self, exportar_txt: str = None, exportar_json: str = None) -> bool:
        """Executa o processo completo de validação"""
        print("Iniciando validação de documentos (versão 3 - com nova métrica global)...")
        
        if not self.carregar_arquivos():
            return False
        
        print("Arquivos carregados com sucesso!")
        print(f"Documentos no gabarito: {len(self.gabarito)}")
        print(f"Documentos na resposta: {len(self.resposta)}")
        
        self.validar_todos_documentos()
        self.gerar_relatorio()
        
        if exportar_txt:
            self.gerar_relatorio_txt(exportar_txt)
        
        if exportar_json:
            self.exportar_relatorio_json(exportar_json)
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Validador V3 - Com nova métrica global "Todas Empresas + Todos Sócios"'
    )
    parser.add_argument(
        'gabarito',
        help='Caminho para o arquivo JSON do gabarito'
    )
    parser.add_argument(
        'resposta',
        help='Caminho para o arquivo JSON das respostas'
    )
    parser.add_argument(
        '--exportar-txt',
        dest='exportar_txt',
        help='Caminho para exportar o relatório em formato TXT',
        default=None
    )
    parser.add_argument(
        '--exportar-json',
        dest='exportar_json',
        help='Caminho para exportar o relatório em formato JSON',
        default=None
    )
    parser.add_argument(
        '--tolerancia-nome',
        type=float,
        default=1.0,
        help='Similaridade mínima (0-1) para nome de empresa. Default: 1.0 (exato)'
    )
    parser.add_argument(
        '--tolerancia-nome-socio',
        type=float,
        default=1.0,
        help='Similaridade mínima (0-1) para nome de sócio. Default: 1.0 (exato)'
    )
    
    args = parser.parse_args()

    # Sem valores fixos: só entra o que existir no JSON de resposta (ou {}).
    # Quem orquestra o lote (ex.: rodar_validador_em_lote) continua a passar config lida do relatório.
    config_params = extrair_parametros_extracao_de_json(args.resposta)

    validador = ValidadorDocumentos(
        args.gabarito,
        args.resposta,
        args.tolerancia_nome,
        args.tolerancia_nome_socio,
        config_params
    )
    
    if not validador.executar(args.exportar_txt, args.exportar_json):
        sys.exit(1)


if __name__ == '__main__':
    main()