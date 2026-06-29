"""Parse de respostas LLM para code v1 — sem Pydantic."""

from __future__ import annotations

import json
import re
from typing import Any

SCHEMA_MARKERS = frozenset({"$defs", "$schema", "properties", "required", "title", "description"})

INVALID_TEXT = frozenset({"null", "none", "n/a", "não disponível", "não informado", "-", "na", "nd"})


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def is_schema_echo(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    keys = set(data.keys())
    if keys & SCHEMA_MARKERS:
        return True
    if "$ref" in keys or "allOf" in keys or "anyOf" in keys:
        return True
    props = data.get("properties")
    return isinstance(props, dict) and bool(props)


def extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def clean_raw_response(response: str) -> str:
    text = (response or "").strip()
    text = re.sub(
        r"<think>[\s\S]*?</think>",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    text = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", text, flags=re.IGNORECASE).strip()
    return text


def _unwrap_list(data: dict, list_key: str, alt_keys: tuple[str, ...]) -> list | None:
    if list_key in data:
        val = data[list_key]
        return val if isinstance(val, list) else [val]
    for alt in alt_keys:
        if alt in data:
            val = data[alt]
            if isinstance(val, list):
                return val
            if isinstance(val, dict) and list_key in val:
                inner = val[list_key]
                return inner if isinstance(inner, list) else [inner]
    return None


def _parse_json_items(raw: str, list_key: str, alt_keys: tuple[str, ...]) -> tuple[list[dict], str | None]:
    cleaned = clean_raw_response(raw)
    blob = extract_json_object(cleaned)
    if not blob:
        return [], "no_json"

    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        return [], f"json_decode:{e}"

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if is_schema_echo(data):
            return [], "schema_echo"
        items = _unwrap_list(data, list_key, alt_keys)
        if items is None:
            return [], "no_list_key"
    else:
        return [], "invalid_root"

    out = []
    for item in items:
        if isinstance(item, dict):
            out.append({k: _coerce_str(v) for k, v in item.items()})
    return out, None


def parse_empresas_response(raw: str) -> tuple[list[tuple[str, str]], str | None]:
    """Retorna [(nome, cnpj), ...] a partir de JSON ou texto legado."""
    items, err = _parse_json_items(
        raw,
        "empresas",
        ("empresas", "Empresa", "empresa", "companies"),
    )
    if err == "schema_echo":
        return [], "schema_echo"
    if items:
        result = []
        for item in items:
            nome = _coerce_str(item.get("nome") or item.get("name") or item.get("empresa"))
            cnpj = _coerce_str(item.get("cnpj") or item.get("CNPJ"))
            if nome:
                result.append((nome, cnpj))
        if result:
            return result, None
        if err is None:
            return [], "empty_json"

    text = clean_raw_response(raw)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    for line in text.split("\n"):
        line = line.strip()
        if "," in line and ";" in line:
            text = line
            break

    result = []
    for registro in [r.strip() for r in text.split(";") if r.strip()]:
        if "," in registro:
            partes = registro.split(",", 1)
            if len(partes) == 2:
                nome, cnpj = _coerce_str(partes[0]), _coerce_str(partes[1])
                if nome.lower() not in INVALID_TEXT:
                    result.append((nome, cnpj))
        else:
            nome = _coerce_str(registro)
            if nome and nome.lower() not in INVALID_TEXT:
                result.append((nome, ""))

    if result:
        return result, None
    return [], err or "empty_text"


def parse_pessoas_response(raw: str) -> tuple[list[tuple[str, str, str]], str | None]:
    """Retorna [(nome, cpf, rg), ...] a partir de JSON ou texto legado."""
    items, err = _parse_json_items(
        raw,
        "socios",
        ("socios", "sócios", "socio", "pessoas", "partners"),
    )
    if err == "schema_echo":
        return [], "schema_echo"
    if items:
        result = []
        for item in items:
            nome = _coerce_str(item.get("nome") or item.get("name"))
            cpf = _coerce_str(item.get("cpf") or item.get("CPF"))
            rg = _coerce_str(item.get("rg") or item.get("RG"))
            if nome:
                result.append((nome, cpf, rg))
        if result:
            return result, None
        if err is None:
            return [], "empty_json"

    text = clean_raw_response(raw)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    result = []
    for registro in [r.strip() for r in text.split(";") if r.strip()]:
        partes = [_coerce_str(p) for p in registro.split(",", 2)]
        while len(partes) < 3:
            partes.append("")
        nome, cpf, rg = partes[0], partes[1], partes[2]
        if nome and nome.lower() not in INVALID_TEXT:
            result.append((nome, cpf, rg))

    if result:
        return result, None
    return [], err or "empty_text"


def correction_for_error(error: str | None, kind: str) -> str | None:
    if not error:
        return None
    if error == "schema_echo":
        return (
            "Sua resposta anterior parecia um JSON Schema, não dados. "
            f"Responda EXCLUSIVAMENTE no formato texto: "
            f"{'NOME, CNPJ;' if kind == 'empresa' else 'NOME, CPF, RG;'}"
        )
    if error.startswith("json_decode") or error == "no_json":
        return (
            f"Responda EXCLUSIVAMENTE no formato texto (sem JSON, sem markdown): "
            f"{'NOME, CNPJ;' if kind == 'empresa' else 'NOME, CPF, RG;'}"
        )
    if "null" in error.lower():
        return "Não use a palavra null. Deixe o campo vazio se não souber."
    return (
        f"Responda EXCLUSIVAMENTE no formato: "
        f"{'NOME, CNPJ;' if kind == 'empresa' else 'NOME, CPF, RG;'}"
    )
