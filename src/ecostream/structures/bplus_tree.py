"""Etapa 4 (S5) - B+Tree como índice en disco.

Cada nodo es una **página de 4 KB** almacenada en un archivo binario.  Las
páginas se leen/escriben a través de un *buffer pool* LRU (etapa 3): un
acierto en el pool no toca el disco; una expulsión de una página sucia la
escribe (política write-back).  Así se puede medir el costo real en
**lecturas/escrituras de página**, que es la métrica relevante para un índice
de disco.

* Orden (fan-out) derivado del tamaño de página: ~200 claves/página para claves
  enteras de 64 bits => con 10^7 claves la altura es 4.
* Hojas enlazadas => consultas por rango O(log_B n + k/B) páginas.
* Búsqueda puntual O(log_B n) páginas; inserción O(log_B n) amortizado.

Alternativas descartadas: árbol AVL/rojo-negro (altura log2 n => ~24 accesos
a disco para 10^7 claves, frente a 4); tabla hash (O(1) puntual pero sin
rangos ordenados).
"""
from __future__ import annotations

import os
import pickle
import tempfile
from bisect import bisect_left as _bl, bisect_right as _br

from .lru_cache import LRUCache

PAGE_SIZE = 4096


class Pager:
    """Gestor de páginas de tamaño fijo en un archivo, con buffer pool LRU."""

    def __init__(self, path: str | None = None, page_size: int = PAGE_SIZE,
                 pool_pages: int = 1024):
        if path is None:
            fd, path = tempfile.mkstemp(prefix="bptree_", suffix=".idx")
            os.close(fd)
        self.path = path
        self.page_size = page_size
        self.f = open(path, "w+b")
        self.n_pages = 0
        self.disk_reads = 0
        self.disk_writes = 0
        self.dirty: set[int] = set()
        self.pool = LRUCache(pool_pages, on_evict=self._evict)

    # -- E/S física ---------------------------------------------------------
    def _write_page(self, pid: int, node) -> None:
        raw = pickle.dumps(node, protocol=pickle.HIGHEST_PROTOCOL)
        if len(raw) > self.page_size:
            raise ValueError(f"nodo de {len(raw)} B excede la página")
        self.f.seek(pid * self.page_size)
        self.f.write(raw.ljust(self.page_size, b"\0"))
        self.disk_writes += 1

    def _read_page(self, pid: int):
        self.f.seek(pid * self.page_size)
        raw = self.f.read(self.page_size)
        self.disk_reads += 1
        return pickle.loads(raw)

    def _evict(self, pid, node):
        if pid in self.dirty:
            self._write_page(pid, node)
            self.dirty.discard(pid)

    # -- API lógica ---------------------------------------------------------
    def alloc(self, node) -> int:
        pid = self.n_pages
        self.n_pages += 1
        self.put(pid, node)
        return pid

    def get(self, pid: int):
        node = self.pool.get(pid)
        if node is None:
            node = self._read_page(pid)
            self.pool.put(pid, node)
        return node

    def put(self, pid: int, node) -> None:
        self.dirty.add(pid)
        self.pool.put(pid, node)

    def flush(self) -> None:
        for pid, node in list(self.pool.items()):
            if pid in self.dirty:
                self._write_page(pid, node)
        self.dirty.clear()
        self.f.flush()

    def close(self, remove: bool = True) -> None:
        self.f.close()
        if remove and os.path.exists(self.path):
            os.remove(self.path)


# Nodo = [es_hoja, claves, hijos|valores, siguiente_hoja]
LEAF, KEYS, PTRS, NEXT = 0, 1, 2, 3


class BPlusTree:
    def __init__(self, path: str | None = None, pool_pages: int = 1024,
                 max_keys: int | None = None, page_size: int = PAGE_SIZE):
        self.pager = Pager(path, page_size=page_size, pool_pages=pool_pages)
        # 64 B de cabecera de pickle + ~20 B por par (clave int64, valor int64)
        self.max_keys = max_keys or (page_size - 96) // 20
        self.root = self.pager.alloc([True, [], [], -1])
        self.height = 1
        self.size = 0

    # ------------------------------------------------------------------
    def _leaf_for(self, key, path=None) -> int:
        pid = self.root
        node = self.pager.get(pid)
        while not node[LEAF]:
            i = _br(node[KEYS], key)
            if path is not None:
                path.append((pid, i))
            pid = node[PTRS][i]
            node = self.pager.get(pid)
        return pid

    def search(self, key):
        pid = self._leaf_for(key)
        node = self.pager.get(pid)
        i = _bl(node[KEYS], key)
        if i < len(node[KEYS]) and node[KEYS][i] == key:
            return node[PTRS][i]
        return None

    def __contains__(self, key):
        return self.search(key) is not None

    def insert(self, key, value) -> None:
        path: list[tuple[int, int]] = []
        pid = self._leaf_for(key, path)
        leaf = self.pager.get(pid)
        keys, vals = leaf[KEYS], leaf[PTRS]
        i = _bl(keys, key)
        if i < len(keys) and keys[i] == key:
            vals[i] = value
            self.pager.put(pid, leaf)
            return
        keys.insert(i, key)
        vals.insert(i, value)
        self.size += 1
        if len(keys) <= self.max_keys:
            self.pager.put(pid, leaf)
            return
        # --- división de hoja
        mid = len(keys) // 2
        right = [True, keys[mid:], vals[mid:], leaf[NEXT]]
        rpid = self.pager.alloc(right)
        del keys[mid:]
        del vals[mid:]
        leaf[NEXT] = rpid
        self.pager.put(pid, leaf)
        sep = right[KEYS][0]
        self._insert_in_parent(path, pid, sep, rpid)

    def _insert_in_parent(self, path, left_pid, sep, right_pid):
        while True:
            if not path:  # nueva raíz
                self.root = self.pager.alloc([False, [sep], [left_pid, right_pid], -1])
                self.height += 1
                return
            ppid, ci = path.pop()
            parent = self.pager.get(ppid)
            parent[KEYS].insert(ci, sep)
            parent[PTRS].insert(ci + 1, right_pid)
            if len(parent[KEYS]) <= self.max_keys:
                self.pager.put(ppid, parent)
                return
            # división de nodo interno: la clave del medio sube
            k = parent[KEYS]
            c = parent[PTRS]
            mid = len(k) // 2
            up = k[mid]
            right = [False, k[mid + 1:], c[mid + 1:], -1]
            rpid = self.pager.alloc(right)
            del k[mid:]
            del c[mid + 1:]
            self.pager.put(ppid, parent)
            left_pid, sep, right_pid = ppid, up, rpid

    def range(self, lo, hi):
        """Itera (clave, valor) con lo <= clave <= hi recorriendo hojas enlazadas."""
        pid = self._leaf_for(lo)
        while pid != -1:
            node = self.pager.get(pid)
            keys, vals = node[KEYS], node[PTRS]
            i = _bl(keys, lo)
            for j in range(i, len(keys)):
                if keys[j] > hi:
                    return
                yield keys[j], vals[j]
            pid = node[NEXT]
            lo = float("-inf")

    def bulk_load(self, sorted_items, fill: float = 0.9) -> None:
        """Carga masiva bottom-up de pares ordenados: O(n) y hojas al ~90%."""
        per = max(2, int(self.max_keys * fill))
        level = []           # (primera_clave, pid)
        prev_leaf = None
        items = list(sorted_items)
        self.size = len(items)
        for s in range(0, len(items), per):
            chunk = items[s:s + per]
            node = [True, [k for k, _ in chunk], [v for _, v in chunk], -1]
            pid = self.pager.alloc(node)
            if prev_leaf is not None:
                pn = self.pager.get(prev_leaf)
                pn[NEXT] = pid
                self.pager.put(prev_leaf, pn)
            prev_leaf = pid
            level.append((chunk[0][0], pid))
        if not level:
            return
        self.height = 1
        while len(level) > 1:
            nxt = []
            for s in range(0, len(level), per + 1):
                grp = level[s:s + per + 1]
                node = [False, [k for k, _ in grp[1:]], [p for _, p in grp], -1]
                nxt.append((grp[0][0], self.pager.alloc(node)))
            level = nxt
            self.height += 1
        self.root = level[0][1]

    @property
    def io(self) -> dict:
        p = self.pager
        return {"disk_reads": p.disk_reads, "disk_writes": p.disk_writes,
                "pages": p.n_pages, "pool_hit_rate": round(p.pool.hit_rate, 4),
                "height": self.height}

    def close(self):
        self.pager.close()
