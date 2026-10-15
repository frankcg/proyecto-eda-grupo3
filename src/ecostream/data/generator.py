"""Generador del stream de comercio electrónico (escenario B).

Base estructural: mismo esquema que el dataset público *eCommerce behavior
data from multi category store* (REES46, Kaggle):
    event_time, event_type, product_id, category_id, category_code, brand,
    price, user_id, user_session
más los campos que ese dataset no trae y que el pipeline exige (generados
sintéticamente, como recomienda el enunciado): order_id, shipping
(express/standard/scheduled), payment_token, device_id, ip, query, premium.

Propiedades inyectadas y controladas por semilla:
* Popularidad Zipf de productos y afinidad usuario->categorías (comunidades
  reales en el grafo de co-compra) + categorías complementarias (cross-sell).
* ~2 % de eventos duplicados (re-entregas "at-least-once") con retraso.
* ~0.5 % de registros sucios (tipo en mayúsculas/espacios, precio con coma
  decimal, precio faltante o negativo, categoría en mayúsculas).
* Fraude: tokens de pago comprometidos (lista negra) y ráfagas de
  *card testing* (decenas de intentos de pago en pocos minutos).
El generador es **columnar y vectorizado** (NumPy): 10M eventos en ~1 min.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CATEGORIES = [
    ("electronics.smartphone", "Smartphone"), ("electronics.audio.headphone", "Audífonos"),
    ("electronics.video.tv", "Televisor"), ("electronics.clocks", "Smartwatch"),
    ("electronics.tablet", "Tablet"), ("electronics.camera", "Cámara"),
    ("accessories.phone_case", "Funda"), ("accessories.charger", "Cargador"),
    ("computers.notebook", "Laptop"), ("computers.peripherals.mouse", "Mouse"),
    ("computers.peripherals.keyboard", "Teclado"), ("computers.components.monitor", "Monitor"),
    ("computers.bag", "Mochila para laptop"), ("appliances.kitchen.refrigerators", "Refrigeradora"),
    ("appliances.kitchen.microwave", "Microondas"), ("appliances.kitchen.blender", "Licuadora"),
    ("appliances.kitchen.coffee_machine", "Cafetera"), ("appliances.environment.air_conditioner", "Aire acondicionado"),
    ("appliances.environment.vacuum", "Aspiradora"), ("appliances.iron", "Plancha"),
    ("apparel.shoes.sneakers", "Zapatillas"), ("apparel.shirt", "Polo"),
    ("apparel.jeans", "Jean"), ("apparel.jacket", "Casaca"), ("apparel.bag", "Cartera"),
    ("sport.bicycle", "Bicicleta"), ("sport.fitness.dumbbells", "Mancuernas"),
    ("sport.fitness.yoga_mat", "Mat de yoga"), ("sport.ball", "Pelota"), ("sport.bottle", "Tomatodo"),
    ("kids.toys", "Juguete"), ("kids.carriage", "Coche de bebé"), ("kids.diapers", "Pañales"),
    ("furniture.living_room.sofa", "Sofá"), ("furniture.bedroom.bed", "Cama"),
    ("furniture.kitchen.table", "Mesa"), ("furniture.office.chair", "Silla de oficina"),
    ("beauty.perfume", "Perfume"), ("beauty.skin_care", "Crema facial"), ("beauty.hair_dryer", "Secadora de cabello"),
]
# complementos (cross-selling plantado)
COMPLEMENTS = {
    0: [6, 7, 1], 1: [0], 2: [1], 3: [0, 7], 4: [6, 7], 5: [7], 8: [9, 10, 12, 11], 9: [8, 10],
    10: [8, 9], 11: [8, 10], 12: [8], 13: [14], 14: [13], 15: [16], 16: [15], 20: [21, 22, 29],
    21: [22, 20], 22: [21, 23], 23: [22], 25: [29, 28], 26: [27, 29], 27: [26, 29], 31: [32, 30],
    32: [31], 33: [35], 34: [33], 35: [36], 36: [11], 37: [38], 38: [37], 39: [38],
}
BRANDS = ["samsung", "apple", "xiaomi", "lg", "sony", "huawei", "lenovo", "hp", "asus", "philips",
          "oster", "bosch", "nike", "adidas", "puma", "levis", "decathlon", "lego", "fisherprice",
          "ikea", "loreal", "nivea", "logitech", "jbl", "motorola", "canon", "electrolux", "mabe"]
BASE_PRICE = np.array([900, 150, 1800, 600, 1100, 1500, 40, 60, 3200, 70, 120, 800, 140, 2500, 450,
                       250, 600, 1600, 900, 120, 320, 60, 150, 280, 220, 1300, 180, 90, 80, 45,
                       130, 900, 60, 2200, 1500, 700, 650, 300, 110, 180], dtype=np.float64)

EVENT_TYPES = ["view", "search", "cart", "remove_from_cart", "purchase", "payment_attempt", "return", "review"]
ET = {n: i for i, n in enumerate(EVENT_TYPES)}
SHIPPING = ["", "express", "standard", "scheduled"]
T0 = 1788220800.0      # 2026-09-01 00:00:00 UTC


@dataclass
class Catalog:
    n: int
    category: np.ndarray
    brand: np.ndarray
    price: np.ndarray
    rating: np.ndarray
    discount: np.ndarray
    popularity: np.ndarray
    names: list
    embedding: np.ndarray            # 32-D (contenido) -> HNSW
    stock: np.ndarray

    def attributes(self) -> np.ndarray:
        """4-D estandarizado para el KD-Tree: log precio, rating, log popularidad, descuento."""
        A = np.column_stack([np.log1p(self.price), self.rating, np.log(self.popularity), self.discount])
        return ((A - A.mean(0)) / A.std(0)).astype(np.float32)

    def record(self, pid: int) -> dict:
        c = int(self.category[pid])
        return {"product_id": pid, "name": self.names[pid], "category_code": CATEGORIES[c][0],
                "category": c, "brand": BRANDS[int(self.brand[pid])], "price": float(self.price[pid]),
                "rating": float(self.rating[pid]), "stock": int(self.stock[pid])}


def make_catalog(n_products: int = 20_000, seed: int = 42, dim: int = 32) -> Catalog:
    rng = np.random.default_rng(seed)
    C = len(CATEGORIES)
    cat_w = 1 / np.arange(1, C + 1) ** 0.6
    category = rng.choice(C, n_products, p=cat_w / cat_w.sum())
    macro_brands = {
        "electronics": ["samsung", "apple", "xiaomi", "lg", "sony", "huawei", "motorola", "jbl", "canon"],
        "accessories": ["samsung", "apple", "xiaomi", "logitech", "huawei"],
        "computers": ["lenovo", "hp", "asus", "apple", "logitech", "lg", "samsung"],
        "appliances": ["lg", "samsung", "oster", "bosch", "philips", "electrolux", "mabe"],
        "apparel": ["nike", "adidas", "puma", "levis"], "sport": ["decathlon", "nike", "adidas", "puma"],
        "kids": ["lego", "fisherprice"], "furniture": ["ikea"], "beauty": ["loreal", "nivea", "philips"],
    }
    brand_pool = {c: np.array([BRANDS.index(b) for b in macro_brands[CATEGORIES[c][0].split(".")[0]]])
                  for c in range(C)}
    brand = np.array([brand_pool[c][rng.integers(0, len(brand_pool[c]))] for c in category])
    price = np.round(BASE_PRICE[category] * rng.lognormal(0, 0.45, n_products), 2)
    rating = np.clip(np.round(rng.normal(4.1, 0.6, n_products), 1), 1, 5)
    discount = np.round(rng.choice([0, 0, 0, 0.1, 0.2, 0.3], n_products), 2)
    popularity = 1 / (rng.permutation(n_products) + 1.0) ** 0.9
    names = []
    for i in range(n_products):
        c = category[i]
        names.append(f"{CATEGORIES[c][1]} {BRANDS[brand[i]].capitalize()} "
                     f"{chr(65 + rng.integers(0, 26))}{rng.integers(10, 999)}")
    cent = rng.normal(0, 1, (C, dim))
    bcent = rng.normal(0, 0.5, (len(BRANDS), dim))
    emb = cent[category] + bcent[brand] + rng.normal(0, 0.6, (n_products, dim))
    emb[:, 0] += (np.log(price) - np.log(price).mean())
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    stock = rng.integers(0, 500, n_products)
    return Catalog(n_products, category, brand, price, rating, discount, popularity, names,
                   emb.astype(np.float32), stock)


@dataclass
class StreamConfig:
    n_events: int = 1_000_000
    n_users: int | None = None
    days: float = 30.0
    dup_rate: float = 0.02
    dirty_rate: float = 0.005
    order_rate: float = 0.035          # órdenes por evento
    blacklist_size: int = 500
    blacklisted_order_rate: float = 0.004
    n_fraud_bursts: int | None = None
    seed: int = 42


@dataclass
class Stream:
    cols: dict
    n: int
    blacklist_tokens: np.ndarray
    fraud_devices: np.ndarray
    premium: np.ndarray
    meta: dict = field(default_factory=dict)


def _pick_products(rng, cat: Catalog, cats: np.ndarray, by_cat_idx, by_cat_cdf):
    out = np.empty(len(cats), dtype=np.int64)
    for c in np.unique(cats):
        sel = np.where(cats == c)[0]
        idx, cdf = by_cat_idx[c], by_cat_cdf[c]
        out[sel] = idx[np.minimum(np.searchsorted(cdf, rng.random(len(sel))), len(idx) - 1)]
    return out


def generate_stream(cat: Catalog, cfg: StreamConfig) -> Stream:
    rng = np.random.default_rng(cfg.seed)
    C = len(CATEGORIES)
    n_users = cfg.n_users or max(1000, cfg.n_events // 60)
    # afinidad usuario -> categoría principal (comunidades)
    cat_w = np.bincount(cat.category, minlength=C).astype(float)
    user_cat = rng.choice(C, n_users, p=cat_w / cat_w.sum())
    user_cat2 = rng.choice(C, n_users, p=cat_w / cat_w.sum())
    premium = rng.random(n_users) < 0.12
    by_cat_idx, by_cat_cdf = {}, {}
    for c in range(C):
        idx = np.where(cat.category == c)[0]
        if len(idx) == 0:
            idx = np.array([0])
        p = cat.popularity[idx]
        by_cat_idx[c] = idx
        by_cat_cdf[c] = np.cumsum(p) / p.sum()
    user_act = 1 / (rng.permutation(n_users) + 1.0) ** 0.5
    user_cdf = np.cumsum(user_act) / user_act.sum()

    # ---------------- órdenes (baskets) ----------------
    n_orders = max(1, int(cfg.n_events * cfg.order_rate))
    o_user = np.minimum(np.searchsorted(user_cdf, rng.random(n_orders)), n_users - 1)
    o_size = np.minimum(rng.geometric(0.45, n_orders), 6)
    o_time = T0 + np.sort(rng.random(n_orders)) * cfg.days * 86400
    o_ship = rng.choice([1, 2, 3], n_orders, p=[0.2, 0.65, 0.15])
    tokens_per_user = n_users * 1  # un token por usuario (más los de la lista negra)
    o_token = o_user.astype(np.int64)
    blacklist = np.arange(tokens_per_user, tokens_per_user + cfg.blacklist_size, dtype=np.int64)
    bl = rng.random(n_orders) < cfg.blacklisted_order_rate
    o_token[bl] = rng.choice(blacklist, bl.sum())
    o_device = (o_user * 3 + rng.integers(0, 2, n_orders)).astype(np.int64)
    line_order = np.repeat(np.arange(n_orders), o_size)
    pos = np.arange(len(line_order)) - np.repeat(np.cumsum(o_size) - o_size, o_size)
    first_cat = np.where(rng.random(n_orders) < 0.75, user_cat[o_user], user_cat2[o_user])
    lcat = first_cat[line_order].copy()
    # ítems siguientes: complemento (55 %), misma categoría (35 %), aleatorio (10 %)
    later = pos > 0
    r = rng.random(len(lcat))
    for c, comps in COMPLEMENTS.items():
        sel = later & (lcat == c) & (r < 0.55)
        if sel.any():
            lcat[sel] = rng.choice(comps, sel.sum())
    rnd_sel = later & (r > 0.90)
    lcat[rnd_sel] = rng.choice(C, rnd_sel.sum(), p=cat_w / cat_w.sum())
    lprod = _pick_products(rng, cat, lcat, by_cat_idx, by_cat_cdf)
    n_lines = len(lprod)

    # ---------------- ráfagas de card testing ----------------
    n_bursts = cfg.n_fraud_bursts if cfg.n_fraud_bursts is not None else max(3, cfg.n_events // 50_000)
    fraud_devices = np.arange(10**7, 10**7 + n_bursts, dtype=np.int64)
    b_len = rng.integers(25, 80, n_bursts)
    b_start = T0 + rng.random(n_bursts) * cfg.days * 86400
    b_dev = np.repeat(fraud_devices, b_len)
    b_time = np.repeat(b_start, b_len) + rng.random(b_len.sum()) * 180   # en <= 3 minutos
    b_token = np.repeat(rng.choice(blacklist, n_bursts) if n_bursts else np.array([], np.int64), b_len)
    # la mitad de las ráfagas usa tokens nuevos (no listados): solo el CMS las detecta
    fresh = np.repeat(rng.random(n_bursts) < 0.5, b_len)
    b_token = np.where(fresh, 2 * 10**7 + np.repeat(np.arange(n_bursts), b_len), b_token)
    n_burst = len(b_dev)

    # ---------------- resto de eventos ----------------
    n_pay = n_orders
    n_rest = max(0, cfg.n_events - n_lines - n_pay - n_burst)
    rest_type = rng.choice([ET["view"], ET["search"], ET["cart"], ET["remove_from_cart"], ET["return"], ET["review"]],
                           n_rest, p=[0.70, 0.09, 0.14, 0.03, 0.015, 0.025])
    rest_user = np.minimum(np.searchsorted(user_cdf, rng.random(n_rest)), n_users - 1)
    rest_cat = np.where(rng.random(n_rest) < 0.7, user_cat[rest_user], user_cat2[rest_user])
    rest_prod = _pick_products(rng, cat, rest_cat, by_cat_idx, by_cat_cdf)
    rest_time = T0 + rng.random(n_rest) * cfg.days * 86400

    # ---------------- ensamblado columnar ----------------
    etype = np.concatenate([np.full(n_lines, ET["purchase"]), np.full(n_pay, ET["payment_attempt"]),
                            np.full(n_burst, ET["payment_attempt"]), rest_type]).astype(np.int8)
    user = np.concatenate([o_user[line_order], o_user, np.full(n_burst, -1), rest_user])
    prod = np.concatenate([lprod, np.full(n_pay, -1), np.full(n_burst, -1), rest_prod])
    ts = np.concatenate([o_time[line_order] + pos * 0.5, o_time - 5.0, b_time, rest_time])
    order = np.concatenate([line_order, np.arange(n_orders), np.full(n_burst, -1), np.full(n_rest, -1)])
    ship = np.concatenate([o_ship[line_order], o_ship, np.zeros(n_burst, int), np.zeros(n_rest, int)]).astype(np.int8)
    token = np.concatenate([o_token[line_order], o_token, b_token, np.full(n_rest, -1)])
    device = np.concatenate([o_device[line_order], o_device, b_dev,
                             (rest_user * 3 + rng.integers(0, 2, n_rest)).astype(np.int64)])
    amount = np.zeros(len(etype))
    amount[:n_lines] = cat.price[lprod]
    np.add.at(amount, n_lines + line_order, cat.price[lprod])          # monto del pago = suma de la orden
    amount[n_lines + n_pay:n_lines + n_pay + n_burst] = np.round(rng.uniform(1, 30, n_burst), 2)
    price = np.where(prod >= 0, cat.price[np.maximum(prod, 0)], amount)
    session = np.where(user >= 0, user * 1000 + (ts - T0).astype(np.int64) // 3600, device)

    order_idx = np.argsort(ts, kind="stable")
    cols = {"event_time": ts, "event_type": etype, "product_id": prod, "user_id": user,
            "order_id": order, "shipping": ship, "payment_token": token, "device_id": device,
            "price": price, "user_session": session}
    cols = {k: v[order_idx] for k, v in cols.items()}
    n = len(etype)
    cols["event_id"] = np.arange(n, dtype=np.int64)
    # pequeño desorden de llegada (jitter de red)
    jitter = rng.normal(0, 0.8, n)
    cols["arrival"] = cols["event_time"] + np.abs(jitter)

    # ---------------- duplicados (re-entregas) ----------------
    n_dup = int(n * cfg.dup_rate)
    dsel = rng.choice(n, n_dup, replace=False)
    dup = {k: v[dsel] for k, v in cols.items()}
    dup["arrival"] = dup["arrival"] + rng.uniform(0.5, 60, n_dup)
    cols = {k: np.concatenate([cols[k], dup[k]]) for k in cols}
    # ---------------- suciedad ----------------
    N = len(cols["event_id"])
    cols["dirty"] = np.zeros(N, dtype=np.int8)
    dirty = rng.random(N) < cfg.dirty_rate
    cols["dirty"][dirty] = rng.integers(1, 6, dirty.sum())  # 1..5 tipos de suciedad
    # ---------------- búsquedas: prefijo del nombre ----------------
    arr = np.argsort(cols["arrival"], kind="stable")
    cols = {k: v[arr] for k, v in cols.items()}
    meta = {"n_orders": n_orders, "n_lines": n_lines, "n_bursts": n_bursts, "n_dup": n_dup,
            "n_dirty": int(dirty.sum()), "n_users": n_users, "blacklisted_orders": int(bl.sum())}
    return Stream(cols, N, blacklist, fraud_devices, premium, meta)


def iter_events(stream: Stream, cat: Catalog, start: int = 0, stop: int | None = None):
    """Materializa eventos como dicts (formato del stream) aplicando la suciedad."""
    c = stream.cols
    stop = stream.n if stop is None else min(stop, stream.n)
    rng = np.random.default_rng(stream.meta.get("seed", 7) + start)
    for i in range(start, stop):
        et = EVENT_TYPES[c["event_type"][i]]
        pid = int(c["product_id"][i])
        uid = int(c["user_id"][i])
        price = float(c["price"][i])
        ev = {
            "event_id": int(c["event_id"][i]),
            "event_time": float(c["event_time"][i]),
            "event_type": et,
            "product_id": pid if pid >= 0 else None,
            "category_code": CATEGORIES[int(cat.category[pid])][0] if pid >= 0 else None,
            "brand": BRANDS[int(cat.brand[pid])] if pid >= 0 else None,
            "price": price,
            "user_id": uid if uid >= 0 else None,
            "user_session": int(c["user_session"][i]),
            "order_id": int(c["order_id"][i]) if c["order_id"][i] >= 0 else None,
            "shipping": SHIPPING[int(c["shipping"][i])] or None,
            "payment_token": int(c["payment_token"][i]) if c["payment_token"][i] >= 0 else None,
            "device_id": int(c["device_id"][i]),
            "premium": bool(stream.premium[uid]) if uid >= 0 else False,
            "query": None,
        }
        if et == "search" and pid >= 0:
            name = cat.names[pid]
            ev["query"] = name[: int(rng.integers(3, min(12, len(name)) + 1))]
        d = int(c["dirty"][i])
        if d == 1:
            ev["event_type"] = f" {et.upper()} "
        elif d == 2:
            ev["price"] = f"{price:.2f}".replace(".", ",")
        elif d == 3:
            ev["price"] = None
        elif d == 4:
            ev["price"] = -abs(price)
        elif d == 5 and ev["category_code"]:
            ev["category_code"] = ev["category_code"].upper()
        yield ev
