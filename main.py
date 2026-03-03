import asyncio
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Optional
import websockets
import httpx
from dotenv import load_dotenv
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3

load_dotenv()

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
CLOB_HOST            = "https://clob.polymarket.com"
GAMMA_HOST           = "https://gamma-api.polymarket.com"
WS_HOST              = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
CHAIN_ID             = 137
PRIVATE_KEY          = os.getenv("PRIVATE_KEY", "")
FUNDER_ADDRESS       = os.getenv("FUNDER_ADDRESS", "")  # wallet holding USDC

SPREAD_PCT           = Decimal("0.02")        # 2 % spread around mid
ORDER_SIZE_USDC      = Decimal("10.0")        # USDC per side
MAX_CAPITAL_USDC     = Decimal("200.0")       # max capital per market
DAILY_LOSS_LIMIT     = Decimal("20.0")        # stop bot if daily loss exceeds this
MIN_REWARD_APR       = 20.0                   # % APR minimum to enter market
MIN_LIQUIDITY_USDC   = 500.0                  # minimum order-book liquidity
REQUOTE_INTERVAL     = 60                     # seconds between requotes
MARKET_SCAN_INTERVAL = 300                    # seconds between market scans
MAX_ACTIVE_MARKETS   = 3                      # max simultaneous markets
HEDGE_RETRY_TIMES    = 5                      # retries for hedge orders
LOG_FILE             = "bot_log.json"
# ─────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("polymarket_mm")


# ─────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────
@dataclass
class OrderState:
    order_id: str
    token_id: str
    side: str          # BUY / SELL
    price: Decimal
    size: Decimal
    filled: Decimal = Decimal("0")
    status: str = "OPEN"


@dataclass
class MarketPosition:
    condition_id: str
    yes_token: str
    no_token: str
    yes_cost: Decimal = Decimal("0")
    no_cost: Decimal = Decimal("0")
    yes_shares: Decimal = Decimal("0")
    no_shares: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    orders: dict = field(default_factory=dict)
    locked: bool = False


# ─────────────────────────────────────────
#  JSON LOGGER
# ─────────────────────────────────────────
def jlog(event: str, data: dict):
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **data,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    log.info("[%s] %s", event, json.dumps(data, default=str))


# ─────────────────────────────────────────
#  AUTH / SIGNING
# ─────────────────────────────────────────
class PolyAuth:
    def __init__(self, private_key: str):
        self.account = Account.from_key(private_key)
        self.address = self.account.address

    def sign_clob_auth(self) -> dict:
        ts = int(time.time())
        nonce = 0
        msg = f"This request grants CLOB API Access\nAddress: {self.address}\nTimestamp: {ts}\nNonce: {nonce}"
        signable = encode_defunct(text=msg)
        signed = self.account.sign_message(signable)
        return {
            "POLY_ADDRESS": self.address,
            "POLY_SIGNATURE": signed.signature.hex(),
            "POLY_TIMESTAMP": str(ts),
            "POLY_NONCE": str(nonce),
        }

    def sign_order(self, order_data: dict) -> str:
        msg_str = json.dumps(order_data, sort_keys=True, separators=(",", ":"))
        signable = encode_defunct(text=msg_str)
        signed = self.account.sign_message(signable)
        return signed.signature.hex()


# ─────────────────────────────────────────
#  CLOB CLIENT
# ─────────────────────────────────────────
class ClobClient:
    def __init__(self, auth: PolyAuth):
        self.auth = auth
        self.http = httpx.AsyncClient(timeout=30)
        self._headers: dict = {}
        self._api_key: str = ""
        self._api_secret: str = ""
        self._api_passphrase: str = ""

    async def _refresh_headers(self):
        auth_headers = self.auth.sign_clob_auth()
        self._headers = {
            "Content-Type": "application/json",
            **auth_headers,
        }

    async def derive_api_key(self):
        await self._refresh_headers()
        resp = await self.http.get(
            f"{CLOB_HOST}/auth/derive-api-key",
            headers=self._headers,
        )
        resp.raise_for_status()
        data = resp.json()
        self._api_key = data.get("apiKey", "")
        self._api_secret = data.get("secret", "")
        self._api_passphrase = data.get("passphrase", "")
        jlog("api_key_derived", {"key": self._api_key[:8] + "..."})

    def _get_l2_headers(self, method: str, path: str, body: str = "") -> dict:
        ts = str(int(time.time()))
        msg = ts + method.upper() + path + body
        from hmac import new as hmac_new
        from hashlib import sha256
        import base64
        sig = base64.b64encode(
            hmac_new(
                self._api_secret.encode(),
                msg.encode(),
                sha256,
            ).digest()
        ).decode()
        return {
            "POLY-API-KEY": self._api_key,
            "POLY-SIGNATURE": sig,
            "POLY-TIMESTAMP": ts,
            "POLY-PASSPHRASE": self._api_passphrase,
            "Content-Type": "application/json",
        }

    async def get_markets(self, limit: int = 100, offset: int = 0) -> list:
        params = {"limit": limit, "offset": offset, "closed": "false", "active": "true"}
        resp = await self.http.get(f"{GAMMA_HOST}/markets", params=params)
        resp.raise_for_status()
        return resp.json()

    async def get_orderbook(self, token_id: str) -> dict:
        resp = await self.http.get(f"{CLOB_HOST}/book", params={"token_id": token_id})
        resp.raise_for_status()
        return resp.json()

    async def get_rewards(self, condition_id: str) -> dict:
        try:
            resp = await self.http.get(
                f"{CLOB_HOST}/rewards/markets/{condition_id}",
                headers=self._get_l2_headers("GET", f"/rewards/markets/{condition_id}"),
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {}

    async def place_order(
        self,
        token_id: str,
        side: str,
        price: Decimal,
        size: Decimal,
    ) -> Optional[str]:
        order_id = str(uuid.uuid4())
        body_data = {
            "order_id": order_id,
            "token_id": token_id,
            "side": side.upper(),
            "price": str(price),
            "size": str(size),
            "type": "GTC",
            "funder": self.auth.address,
        }
        body_str = json.dumps(body_data, separators=(",", ":"))
        sig = self.auth.sign_order(body_data)
        body_data["signature"] = sig
        body_str = json.dumps(body_data, separators=(",", ":"))
        path = "/order"
        headers = self._get_l2_headers("POST", path, body_str)
        for attempt in range(3):
            try:
                resp = await self.http.post(
                    f"{CLOB_HOST}{path}",
                    content=body_str,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                placed_id = data.get("orderID", order_id)
                jlog("order_placed", {
                    "order_id": placed_id, "token_id": token_id,
                    "side": side, "price": str(price), "size": str(size),
                })
                return placed_id
            except Exception as e:
                jlog("order_place_error", {"attempt": attempt, "error": str(e)})
                await asyncio.sleep(2 ** attempt)
        return None

    async def cancel_order(self, order_id: str) -> bool:
        body_str = json.dumps({"orderID": order_id}, separators=(",", ":"))
        path = "/order"
        headers = self._get_l2_headers("DELETE", path, body_str)
        for attempt in range(3):
            try:
                resp = await self.http.delete(
                    f"{CLOB_HOST}{path}",
                    content=body_str,
                    headers=headers,
                )
                resp.raise_for_status()
                jlog("order_cancelled", {"order_id": order_id})
                return True
            except Exception as e:
                jlog("order_cancel_error", {"attempt": attempt, "error": str(e)})
                await asyncio.sleep(2 ** attempt)
        return False

    async def cancel_all_orders(self, token_id: str) -> bool:
        body_str = json.dumps({"tokenID": token_id}, separators=(",", ":"))
        path = "/orders"
        headers = self._get_l2_headers("DELETE", path, body_str)
        try:
            resp = await self.http.delete(
                f"{CLOB_HOST}{path}",
                content=body_str,
                headers=headers,
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            jlog("cancel_all_error", {"error": str(e)})
            return False

    async def get_open_orders(self) -> list:
        path = "/orders"
        headers = self._get_l2_headers("GET", path)
        try:
            resp = await self.http.get(f"{CLOB_HOST}{path}", headers=headers)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            jlog("get_orders_error", {"error": str(e)})
            return []

    async def close(self):
        await self.http.aclose()


# ─────────────────────────────────────────
#  MARKET SCANNER
# ─────────────────────────────────────────
async def estimate_reward_apr(client: ClobClient, condition_id: str, yes_token: str) -> float:
    try:
        rewards_data = await client.get_rewards(condition_id)
        if not rewards_data:
            return 0.0
        daily_reward = float(rewards_data.get("rewardsPerDay", 0) or 0)
        total_liquidity = float(rewards_data.get("totalLiquidity", 1) or 1)
        if total_liquidity <= 0:
            return 0.0
        daily_rate = daily_reward / total_liquidity
        apr = daily_rate * 365 * 100
        return apr
    except Exception:
        return 0.0


async def compute_book_liquidity(book: dict) -> float:
    try:
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        total = 0.0
        for level in bids[:5] + asks[:5]:
            total += float(level.get("size", 0)) * float(level.get("price", 0))
        return total
    except Exception:
        return 0.0


async def scan_markets(client: ClobClient) -> list:
    log.info("Scanning markets ...")
    candidates = []
    try:
        markets = await client.get_markets(limit=200)
        for m in markets:
            try:
                if m.get("closed") or not m.get("active"):
                    continue
                tokens = m.get("tokens", [])
                if len(tokens) < 2:
                    continue
                yes_token = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), None)
                no_token  = next((t for t in tokens if t.get("outcome", "").upper() == "NO"),  None)
                if not yes_token or not no_token:
                    continue

                yes_id = yes_token.get("token_id", "")
                no_id  = no_token.get("token_id", "")
                condition_id = m.get("conditionId", "")

                book_yes = await client.get_orderbook(yes_id)
                liquidity = await compute_book_liquidity(book_yes)

                if liquidity < MIN_LIQUIDITY_USDC:
                    continue

                apr = await estimate_reward_apr(client, condition_id, yes_id)
                if apr < MIN_REWARD_APR:
                    continue

                bids = book_yes.get("bids", [])
                asks = book_yes.get("asks", [])
                best_bid = Decimal(str(bids[0]["price"])) if bids else Decimal("0")
                best_ask = Decimal(str(asks[0]["price"])) if asks else Decimal("1")
                spread = best_ask - best_bid

                candidates.append({
                    "condition_id": condition_id,
                    "question": m.get("question", "")[:60],
                    "yes_token": yes_id,
                    "no_token":  no_id,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "spread": spread,
                    "liquidity": liquidity,
                    "reward_apr": apr,
                })
            except Exception:
                continue

        candidates.sort(key=lambda x: x["reward_apr"], reverse=True)
        selected = candidates[:MAX_ACTIVE_MARKETS]
        jlog("market_scan", {"found": len(candidates), "selected": len(selected)})
        return selected
    except Exception as e:
        jlog("scan_error", {"error": str(e)})
        return []


# ─────────────────────────────────────────
#  QUOTE ENGINE
# ─────────────────────────────────────────
def compute_quotes(best_bid: Decimal, best_ask: Decimal) -> tuple[Decimal, Decimal]:
    mid = (best_bid + best_ask) / 2
    half_spread = mid * SPREAD_PCT / 2
    buy_price  = (mid - half_spread).quantize(Decimal("0.001"), rounding=ROUND_DOWN)
    sell_price = (mid + half_spread).quantize(Decimal("0.001"), rounding=ROUND_DOWN)
    buy_price  = max(Decimal("0.01"), min(buy_price,  Decimal("0.99")))
    sell_price = max(Decimal("0.01"), min(sell_price, Decimal("0.99")))
    return buy_price, sell_price


async def place_two_sided_quotes(
    client: ClobClient,
    position: MarketPosition,
    best_bid: Decimal,
    best_ask: Decimal,
) -> tuple[Optional[str], Optional[str]]:
    buy_price, sell_price = compute_quotes(best_bid, best_ask)
    yes_size = (ORDER_SIZE_USDC / buy_price).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    no_size  = (ORDER_SIZE_USDC / (Decimal("1") - sell_price)).quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    yes_order_id = await client.place_order(position.yes_token, "BUY",  buy_price,  yes_size)
    no_order_id  = await client.place_order(position.no_token,  "BUY",  Decimal("1") - sell_price, no_size)

    if yes_order_id:
        position.orders[yes_order_id] = OrderState(
            order_id=yes_order_id, token_id=position.yes_token,
            side="BUY", price=buy_price, size=yes_size,
        )
    if no_order_id:
        position.orders[no_order_id] = OrderState(
            order_id=no_order_id, token_id=position.no_token,
            side="BUY", price=Decimal("1") - sell_price, size=no_size,
        )

    log.info(
        "Quotes placed | YES buy@%.3f  NO buy@%.3f  APR-focus mode",
        buy_price, Decimal("1") - sell_price,
    )
    return yes_order_id, no_order_id


async def requote(
    client: ClobClient,
    position: MarketPosition,
) -> tuple[Optional[str], Optional[str]]:
    await client.cancel_all_orders(position.yes_token)
    await client.cancel_all_orders(position.no_token)
    position.orders.clear()

    book = await client.get_orderbook(position.yes_token)
    bids = book.get("bids", [])
    asks = book.get("asks", [])
    if not bids or not asks:
        return None, None
    best_bid = Decimal(str(bids[0]["price"]))
    best_ask = Decimal(str(asks[0]["price"]))
    return await place_two_sided_quotes(client, position, best_bid, best_ask)


# ─────────────────────────────────────────
#  HEDGE ENGINE
# ─────────────────────────────────────────
async def hedge_position(
    client: ClobClient,
    position: MarketPosition,
    filled_side: str,
    filled_size: Decimal,
    filled_price: Decimal,
):
    jlog("hedge_start", {
        "condition_id": position.condition_id,
        "filled_side": filled_side,
        "filled_size": str(filled_size),
        "filled_price": str(filled_price),
    })

    if filled_side == "YES":
        position.yes_cost  += filled_price * filled_size
        position.yes_shares += filled_size
        hedge_token  = position.no_token
        hedge_price  = Decimal("1") - filled_price
    else:
        position.no_cost  += filled_price * filled_size
        position.no_shares += filled_size
        hedge_token  = position.yes_token
        hedge_price  = Decimal("1") - filled_price

    hedge_size = filled_size

    for attempt in range(HEDGE_RETRY_TIMES):
        hedge_id = await client.place_order(hedge_token, "BUY", hedge_price, hedge_size)
        if hedge_id:
            jlog("hedge_placed", {"order_id": hedge_id, "price": str(hedge_price), "size": str(hedge_size)})
            break
        await asyncio.sleep(2 ** attempt)

    total_cost = position.yes_cost + position.no_cost
    log.info("Total cost after hedge: %.4f USDC", total_cost)
    if total_cost <= Decimal("1.02") and position.yes_shares > 0 and position.no_shares > 0:
        position.locked = True
        net_pnl = Decimal("1") - total_cost
        position.realized_pnl += net_pnl
        jlog("position_locked", {
            "condition_id": position.condition_id,
            "total_cost": str(total_cost),
            "net_pnl": str(net_pnl),
        })
        log.info("POSITION LOCKED | PnL: %.4f USDC", net_pnl)


async def handle_both_filled(position: MarketPosition):
    total_cost = position.yes_cost + position.no_cost
    net_pnl    = Decimal("1") - total_cost
    position.realized_pnl += net_pnl
    position.locked = True
    jlog("both_sides_filled", {
        "condition_id": position.condition_id,
        "yes_cost": str(position.yes_cost),
        "no_cost":  str(position.no_cost),
        "total_cost": str(total_cost),
        "net_pnl": str(net_pnl),
    })
    log.info("BOTH SIDES FILLED | Cost: %.4f | PnL: %.4f", total_cost, net_pnl)


# ─────────────────────────────────────────
#  REWARD TRACKER
# ─────────────────────────────────────────
async def log_reward_estimate(client: ClobClient, position: MarketPosition):
    data = await client.get_rewards(position.condition_id)
    if not data:
        return
    daily = float(data.get("rewardsPerDay", 0) or 0)
    liquidity = float(data.get("totalLiquidity", 1) or 1)
    our_share = float(ORDER_SIZE_USDC * 2) / liquidity if liquidity > 0 else 0
    est_daily = daily * our_share
    jlog("reward_estimate", {
        "condition_id": position.condition_id,
        "daily_pool": daily,
        "our_share_pct": round(our_share * 100, 4),
        "est_daily_reward": round(est_daily, 4),
        "est_annual_reward": round(est_daily * 365, 2),
    })
    log.info(
        "Reward est | Daily pool: %.2f | Our share: %.2f%% | Daily: %.4f | Annual: %.2f",
        daily, our_share * 100, est_daily, est_daily * 365,
    )


# ─────────────────────────────────────────
#  WEBSOCKET LISTENER
# ─────────────────────────────────────────
class FillListener:
    def __init__(self, client: ClobClient, positions: dict, auth: PolyAuth):
        self.client    = client
        self.positions = positions
        self.auth      = auth
        self._running  = False

    async def _subscribe_msg(self) -> str:
        ts = str(int(time.time()))
        msg_txt = ts + "user"
        from hmac import new as hmac_new
        from hashlib import sha256
        import base64
        sig = base64.b64encode(
            hmac_new(
                self.client._api_secret.encode(),
                msg_txt.encode(),
                sha256,
            ).digest()
        ).decode()
        sub = {
            "auth": {
                "apiKey":     self.client._api_key,
                "secret":     self.client._api_secret,
                "passphrase": self.client._api_passphrase,
            },
            "type": "subscribe",
            "channel": "user",
        }
        return json.dumps(sub)

    async def listen(self):
        self._running = True
        backoff = 2
        while self._running:
            try:
                async with websockets.connect(
                    WS_HOST,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                ) as ws:
                    await ws.send(await self._subscribe_msg())
                    log.info("WebSocket connected – listening for fills ...")
                    backoff = 2
                    async for raw in ws:
                        await self._handle(raw)
            except Exception as e:
                jlog("ws_error", {"error": str(e)})
                log.warning("WebSocket error: %s – reconnecting in %ds", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def stop(self):
        self._running = False

    async def _handle(self, raw: str):
        try:
            msgs = json.loads(raw)
            if not isinstance(msgs, list):
                msgs = [msgs]
            for msg in msgs:
                event_type = msg.get("event_type", "")
                if event_type == "trade":
                    await self._process_fill(msg)
        except Exception as e:
            jlog("ws_parse_error", {"error": str(e)})

    async def _process_fill(self, msg: dict):
        order_id   = msg.get("orderID", "")
        token_id   = msg.get("assetId", "")
        side       = msg.get("side", "").upper()
        fill_price = Decimal(str(msg.get("price", "0")))
        fill_size  = Decimal(str(msg.get("size", "0")))

        jlog("fill_received", {
            "order_id": order_id,
            "token_id": token_id,
            "side":     side,
            "price":    str(fill_price),
            "size":     str(fill_size),
        })

        for cond_id, pos in self.positions.items():
            if pos.locked:
                continue
            if order_id not in pos.orders:
                continue

            order = pos.orders[order_id]
            order.filled += fill_size

            if token_id == pos.yes_token:
                filled_side = "YES"
            elif token_id == pos.no_token:
                filled_side = "NO"
            else:
                continue

            yes_filled = any(
                o.token_id == pos.yes_token and o.filled >= o.size * Decimal("0.95")
                for o in pos.orders.values()
            )
            no_filled = any(
                o.token_id == pos.no_token and o.filled >= o.size * Decimal("0.95")
                for o in pos.orders.values()
            )

            if yes_filled and no_filled:
                await handle_both_filled(pos)
            else:
                await hedge_position(self.client, pos, filled_side, fill_size, fill_price)
            break


# ─────────────────────────────────────────
#  RISK GUARD
# ─────────────────────────────────────────
class RiskGuard:
    def __init__(self):
        self.daily_loss   = Decimal("0")
        self.total_capital = Decimal("0")
        self._reset_day  = datetime.now(timezone.utc).date()

    def check_daily_loss(self, pnl_delta: Decimal) -> bool:
        today = datetime.now(timezone.utc).date()
        if today != self._reset_day:
            self.daily_loss = Decimal("0")
            self._reset_day = today
        if pnl_delta < 0:
            self.daily_loss += abs(pnl_delta)
        if self.daily_loss >= DAILY_LOSS_LIMIT:
            jlog("daily_loss_limit_hit", {"daily_loss": str(self.daily_loss)})
            log.error("DAILY LOSS LIMIT HIT: %.2f USDC – stopping bot.", self.daily_loss)
            return False
        return True

    def can_enter_market(self) -> bool:
        return self.total_capital < MAX_CAPITAL_USDC

    def add_capital(self, amount: Decimal):
        self.total_capital += amount

    def remove_capital(self, amount: Decimal):
        self.total_capital = max(Decimal("0"), self.total_capital - amount)


# ─────────────────────────────────────────
#  MAIN BOT
# ─────────────────────────────────────────
class PolymarketMMBot:
    def __init__(self):
        if not PRIVATE_KEY:
            log.error("PRIVATE_KEY not set in .env")
            sys.exit(1)
        self.auth      = PolyAuth(PRIVATE_KEY)
        self.client    = ClobClient(self.auth)
        self.positions: dict[str, MarketPosition] = {}
        self.risk      = RiskGuard()
        self.listener  = FillListener(self.client, self.positions, self.auth)
        self._stop     = False

    async def initialize(self):
        log.info("Initializing Polymarket MM Bot ...")
        log.info("Wallet: %s", self.auth.address)
        await self.client.derive_api_key()
        log.info("API key ready.")

    async def run_market(self, market_info: dict):
        cid = market_info["condition_id"]
        if cid in self.positions:
            return

        log.info("Entering market: %s", market_info["question"])
        pos = MarketPosition(
            condition_id=cid,
            yes_token=market_info["yes_token"],
            no_token=market_info["no_token"],
        )
        self.positions[cid] = pos
        self.risk.add_capital(ORDER_SIZE_USDC * 2)

        await place_two_sided_quotes(
            self.client, pos,
            market_info["best_bid"],
            market_info["best_ask"],
        )
        await log_reward_estimate(self.client, pos)

    async def requote_loop(self):
        while not self._stop:
            await asyncio.sleep(REQUOTE_INTERVAL)
            for cid, pos in list(self.positions.items()):
                if pos.locked:
                    continue
                try:
                    log.info("Requoting market %s ...", cid[:12])
                    await requote(self.client, pos)
                    await log_reward_estimate(self.client, pos)
                    total_pnl = sum(p.realized_pnl for p in self.positions.values())
                    if not self.risk.check_daily_loss(total_pnl):
                        await self.shutdown()
                        return
                except Exception as e:
                    jlog("requote_error", {"condition_id": cid, "error": str(e)})

    async def scan_loop(self):
        while not self._stop:
            try:
                active = sum(1 for p in self.positions.values() if not p.locked)
                if active < MAX_ACTIVE_MARKETS and self.risk.can_enter_market():
                    markets = await scan_markets(self.client)
                    for m in markets:
                        if m["condition_id"] not in self.positions:
                            await self.run_market(m)
                            await asyncio.sleep(2)
                            if sum(1 for p in self.positions.values() if not p.locked) >= MAX_ACTIVE_MARKETS:
                                break
            except Exception as e:
                jlog("scan_loop_error", {"error": str(e)})
            await asyncio.sleep(MARKET_SCAN_INTERVAL)

    async def status_loop(self):
        while not self._stop:
            await asyncio.sleep(30)
            total_pnl = sum(p.realized_pnl for p in self.positions.values())
            active    = sum(1 for p in self.positions.values() if not p.locked)
            locked    = sum(1 for p in self.positions.values() if p.locked)
            open_orders = sum(len(p.orders) for p in self.positions.values())
            log.info(
                "STATUS | Markets active:%d locked:%d | Open orders:%d | "
                "Total PnL: %.4f | Daily loss: %.4f",
                active, locked, open_orders, total_pnl, self.risk.daily_loss,
            )

    async def shutdown(self):
        self._stop = True
        self.listener.stop()
        log.info("Cancelling all open orders ...")
        for pos in self.positions.values():
            await self.client.cancel_all_orders(pos.yes_token)
            await self.client.cancel_all_orders(pos.no_token)
        total_pnl = sum(p.realized_pnl for p in self.positions.values())
        jlog("shutdown", {"total_realized_pnl": str(total_pnl)})
        log.info("Bot stopped. Total PnL: %.4f USDC", total_pnl)
        await self.client.close()

    async def run(self):
        await self.initialize()
        tasks = [
            asyncio.create_task(self.listener.listen()),
            asyncio.create_task(self.scan_loop()),
            asyncio.create_task(self.requote_loop()),
            asyncio.create_task(self.status_loop()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()


# ─────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    import signal

    bot = PolymarketMMBot()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _handle_signal(*_):
        log.info("Interrupt received – shutting down ...")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    try:
        loop.run_until_complete(bot.run())
    finally:
        loop.close()
