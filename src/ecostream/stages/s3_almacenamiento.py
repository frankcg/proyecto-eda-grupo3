"""Etapa 3 - Almacenamiento, lookup y caché (S4).

* ``HashTable`` (propia) producto -> registro, usuario -> perfil,
  orden -> estado, y lista negra exacta (verificación de la etapa 6).
* ``LRUCache`` para los registros de producto más consultados (los productos
  siguen una distribución Zipf: pocos concentran la mayoría de accesos) y
  para resultados del servicio (etapa 9).
* ``SparseMatrix`` usuario x producto con pesos implícitos
  (vista 1, carrito 3, compra 5) -> conjuntos de compradores (etapa 7) e
  historial del usuario (etapa 9).
Enriquecimiento: completa precio faltante, categoría y marca desde el
catálogo y descuenta stock en compras.
Salida -> evento enriquecido hacia la etapa 4.
"""
from __future__ import annotations

from ..structures.hash_table import HashTable
from ..structures.lru_cache import LRUCache
from ..structures.sparse_matrix import SparseMatrix
from .base import StageMetrics, timed

WEIGHTS = {"view": 1.0, "cart": 3.0, "purchase": 5.0, "review": 2.0}


class Almacenamiento:
    def __init__(self, catalog, n_users: int, lru_capacity: int = 2000, tracer=None):
        self.catalog = catalog
        self.products = HashTable(catalog.n * 2)
        for pid in range(catalog.n):
            self.products.put(pid, catalog.record(pid))
        self.users = HashTable(1024)
        self.orders = HashTable(1024)          # order_id -> {"status", "items", "ts", ...}
        self.payments = HashTable(1024)        # order_id -> estado del pago (de la etapa 6)
        self.blacklist = HashTable(1024)       # verificación exacta de positivos del Bloom
        self.cache = LRUCache(lru_capacity)
        self.interactions = SparseMatrix(n_users, catalog.n)
        self.metrics = StageMetrics("E3 Hash table + LRU + matriz dispersa")
        self.tracer = tracer

    def get_product(self, pid: int):
        rec = self.cache.get(pid)
        if rec is None:
            rec = self.products.get(pid)
            if rec is not None:
                self.cache.put(pid, rec)
        return rec

    def user_profile(self, uid: int) -> dict:
        prof = self.users.get(uid)
        if prof is None:
            prof = {"purchased": [], "recent": [], "cart": set(), "kd_candidates": [], "events": 0}
            self.users.put(uid, prof)
        return prof

    def process(self, ev: dict) -> dict:
        m = self.metrics
        with timed(m, 1):
            pid = ev.get("product_id")
            rec = self.get_product(pid) if pid is not None else None
            if rec is not None:
                if ev.get("_price_missing"):
                    ev["price"] = rec["price"]
                    m.counters["precio_completado"] += 1
                ev["category_code"] = rec["category_code"]
                ev["brand"] = rec["brand"]
                ev["name"] = rec["name"]
            uid = ev.get("user_id")
            if uid is not None:
                prof = self.user_profile(uid)
                prof["events"] += 1
                t = ev["event_type"]
                if pid is not None and t in WEIGHTS:
                    self.interactions.add(uid, pid, WEIGHTS[t])
                if t == "cart" and pid is not None:
                    prof["cart"].add(pid)
                elif t == "remove_from_cart":
                    prof["cart"].discard(pid)
                elif t == "purchase" and pid is not None:
                    prof["purchased"].append(pid)
                    prof["cart"].discard(pid)
                    if rec is not None and rec["stock"] > 0:
                        rec["stock"] -= 1
                if pid is not None:
                    prof["recent"] = (prof["recent"] + [pid])[-10:]
            if ev["event_type"] == "purchase" and ev.get("order_id") is not None:
                o = self.orders.get(ev["order_id"])
                if o is None:
                    o = {"order_id": ev["order_id"], "user_id": uid, "items": [], "ts": ev["event_time"],
                         "shipping": ev.get("shipping"), "status": "pendiente", "amount": 0.0}
                    self.orders.put(ev["order_id"], o)
                o["items"].append(pid)
                o["amount"] += ev["price"] or 0.0
                pay = self.payments.get(ev["order_id"])
                if pay is not None:
                    o["status"] = pay
                ev["_order"] = o
        m.items_out += 1
        if self.tracer and self.tracer.watching(ev):
            extra = f"; producto '{ev.get('name')}' (S/ {ev.get('price')})" if rec else ""
            if rec is not None:
                self.tracer(ev, "E3", f"lookup O(1) en hash table vía LRU (hit-rate {self.cache.hit_rate:.0%}), "
                            f"perfil de usuario e interacción en matriz dispersa actualizados{extra}")
            else:
                self.tracer(ev, "E3", "evento de pago sin producto: se consulta/actualiza la tabla hash de "
                            "órdenes y pagos")
        return ev
