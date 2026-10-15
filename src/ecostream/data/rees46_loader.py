"""Cargador opcional del dataset real REES46 (Kaggle):
https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store

Uso:
    from ecostream.data.rees46_loader import iter_rees46
    for ev in iter_rees46("2019-Oct.csv", limit=10_000_000): ...

Convierte cada fila al esquema de eventos del pipeline.  Los campos que el
dataset no contiene (order_id, shipping, payment_token, device_id) se
derivan de forma determinista: una orden = compras de la misma
``user_session``; el resto se sintetiza con semilla fija, igual que en
``generator.py``.  Así el mismo pipeline corre sobre datos reales o
sintéticos sin cambios.
"""
from __future__ import annotations

import hashlib
import random

SHIP = ["express", "standard", "standard", "standard", "scheduled"]


def _h(x: str) -> int:
    return int.from_bytes(hashlib.blake2b(x.encode(), digest_size=6).digest(), "little")


def iter_rees46(path: str, limit: int | None = None, chunksize: int = 500_000, seed: int = 42):
    import pandas as pd
    rnd = random.Random(seed)
    eid = 0
    for df in pd.read_csv(path, chunksize=chunksize):
        for row in df.itertuples(index=False):
            if limit is not None and eid >= limit:
                return
            ts = pd.Timestamp(row.event_time).timestamp()
            sess = str(row.user_session)
            et = str(row.event_type)
            yield {
                "event_id": eid, "event_time": ts, "event_type": et,
                "product_id": int(row.product_id),
                "category_code": row.category_code if isinstance(row.category_code, str) else None,
                "brand": row.brand if isinstance(row.brand, str) else None,
                "price": row.price, "user_id": int(row.user_id), "user_session": _h(sess),
                "order_id": _h(sess) if et == "purchase" else None,
                "shipping": SHIP[_h(sess) % 5] if et == "purchase" else None,
                "payment_token": int(row.user_id) if et == "purchase" else None,
                "device_id": _h(sess) % 10**9, "premium": rnd.random() < 0.12, "query": None,
            }
            eid += 1
