"""Etapa 9 - Trie para autocompletado con ranking por popularidad.

Cada nodo guarda sus hijos (dict carácter -> nodo) y, si termina una palabra,
los ids de producto asociados.  Además cada nodo mantiene una lista *top-K
cacheada* de los términos más populares de su subárbol, actualizada en la
inserción/actualización de peso, de modo que ``autocomplete(prefijo)`` cuesta
O(|prefijo| + K) en lugar de recorrer todo el subárbol.
    insert            O(L·K)        (L = longitud del término)
    autocomplete      O(|p| + K)
    memoria           O(total de caracteres) nodos
Alternativa: escaneo lineal con ``startswith`` O(n·|p|) por consulta.
"""
from __future__ import annotations

import unicodedata


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in t if not unicodedata.combining(c)).strip()


class _TNode:
    __slots__ = ("children", "terminal", "top")

    def __init__(self):
        self.children: dict[str, _TNode] = {}
        self.terminal = None          # (término, payload)
        self.top: list = []           # [(peso, término, payload)] desc


class Trie:
    def __init__(self, k: int = 8):
        self.root = _TNode()
        self.k = k
        self.weights: dict[str, float] = {}
        self.n_nodes = 1

    def _update_top(self, node: _TNode, w: float, term: str, payload):
        top = [t for t in node.top if t[1] != term]
        top.append((w, term, payload))
        top.sort(key=lambda t: (-t[0], t[1]))
        node.top = top[:self.k]

    def insert(self, term: str, payload=None, weight: float = 0.0) -> None:
        key = normalize(term)
        self.weights[key] = weight
        node = self.root
        self._update_top(node, weight, term, payload)
        for ch in key:
            nxt = node.children.get(ch)
            if nxt is None:
                nxt = node.children[ch] = _TNode()
                self.n_nodes += 1
            node = nxt
            self._update_top(node, weight, term, payload)
        node.terminal = (term, payload)

    def update_weight(self, term: str, weight: float) -> None:
        node = self._find(normalize(term))
        if node is None or node.terminal is None:
            return
        self.insert(node.terminal[0], node.terminal[1], weight)

    def _find(self, prefix: str):
        node = self.root
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def autocomplete(self, prefix: str, k: int | None = None) -> list:
        node = self._find(normalize(prefix))
        if node is None:
            return []
        return node.top[: (k or self.k)]

    def __contains__(self, term: str) -> bool:
        node = self._find(normalize(term))
        return node is not None and node.terminal is not None


def linear_autocomplete(terms: list[tuple[str, float]], prefix: str, k: int = 8):
    """Línea base: escaneo lineal O(n)."""
    p = normalize(prefix)
    hits = [(w, t) for t, w in terms if normalize(t).startswith(p)]
    hits.sort(key=lambda x: (-x[0], x[1]))
    return hits[:k]
