"""Etapa 4 - Indexación principal en disco (S5): B+Tree.

Dos índices B+Tree paginados (páginas de 4 KB en archivo + buffer pool LRU):
* **Órdenes por tiempo**: clave compuesta (segundo << 20 | order_id mod 2^20)
  -> fila en el almacén de órdenes.  La etapa 8 obtiene las órdenes de la
  ventana temporal [t - W, t] con un *range scan* sobre las hojas enlazadas.
* **Catálogo por precio**: clave (precio en céntimos << 20 | product_id),
  cargado con *bulk load*; permite consultas por rango de precio del
  servicio (etapa 9), p.ej. "laptops entre S/ 2000 y S/ 3000".
Salida -> órdenes indexadas (consumidas por la etapa 8) y consultas de rango
para la etapa 9.
"""
from __future__ import annotations

from ..structures.bplus_tree import BPlusTree
from .base import StageMetrics, timed

SHIFT = 20
MASK = (1 << SHIFT) - 1


def time_key(ts: float, order_id: int) -> int:
    return (int(ts) << SHIFT) | (order_id & MASK)


class Indexacion:
    def __init__(self, catalog, pool_pages: int = 256, tracer=None):
        self.orders_idx = BPlusTree(pool_pages=pool_pages)
        self.order_rows: list = []
        self.price_idx = BPlusTree(pool_pages=pool_pages)
        items = sorted(((int(round(catalog.price[p] * 100)) << SHIFT) | p, p) for p in range(catalog.n))
        self.price_idx.bulk_load(items)
        self.metrics = StageMetrics("E4 B+Tree (índice en disco)")
        self.tracer = tracer

    def process(self, ev: dict) -> dict:
        o = ev.get("_order")
        if o is None or o.get("_indexed"):
            return ev
        with timed(self.metrics, 1):
            row = len(self.order_rows)
            self.order_rows.append(o)
            self.orders_idx.insert(time_key(o["ts"], o["order_id"]), row)
            o["_indexed"] = True
        self.metrics.items_out += 1
        if self.tracer and self.tracer.watching(ev):
            self.tracer(ev, "E4", f"orden {o['order_id']} indexada en B+Tree (altura "
                        f"{self.orders_idx.height}, {self.orders_idx.pager.n_pages} páginas de 4 KB)")
        return ev

    def orders_in_window(self, t_from: float, t_to: float):
        lo = time_key(t_from, 0)
        hi = (int(t_to) << SHIFT) | MASK
        for _, row in self.orders_idx.range(lo, hi):
            yield self.order_rows[row]

    def products_in_price_range(self, lo_soles: float, hi_soles: float):
        lo = int(round(lo_soles * 100)) << SHIFT
        hi = (int(round(hi_soles * 100)) << SHIFT) | MASK
        return [pid for _, pid in self.price_idx.range(lo, hi)]

    def io(self) -> dict:
        return {"ordenes": self.orders_idx.io, "precios": self.price_idx.io}
