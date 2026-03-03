import asyncio
import json
import os
import sys
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from io import StringIO

os.environ.setdefault("PRIVATE_KEY", "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80")
os.environ.setdefault("FUNDER_ADDRESS", "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266")

import main
from main import (
    PolyAuth,
    ClobClient,
    MarketPosition,
    OrderState,
    RiskGuard,
    FillListener,
    compute_quotes,
    estimate_reward_apr,
    compute_book_liquidity,
    handle_both_filled,
    hedge_position,
    jlog,
    SPREAD_PCT,
    ORDER_SIZE_USDC,
    DAILY_LOSS_LIMIT,
    MAX_CAPITAL_USDC,
    MIN_REWARD_APR,
)


# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────
def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


MOCK_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"


# ─────────────────────────────────────────
#  1. AUTH TESTS
# ─────────────────────────────────────────
class TestPolyAuth(unittest.TestCase):

    def setUp(self):
        self.auth = PolyAuth(MOCK_KEY)

    def test_address_derived(self):
        self.assertTrue(self.auth.address.startswith("0x"))
        self.assertEqual(len(self.auth.address), 42)

    def test_sign_clob_auth_returns_required_keys(self):
        headers = self.auth.sign_clob_auth()
        for key in ("POLY_ADDRESS", "POLY_SIGNATURE", "POLY_TIMESTAMP", "POLY_NONCE"):
            self.assertIn(key, headers)

    def test_sign_clob_auth_address_matches(self):
        headers = self.auth.sign_clob_auth()
        self.assertEqual(headers["POLY_ADDRESS"], self.auth.address)

    def test_sign_order_returns_hex_string(self):
        sig = self.auth.sign_order({"token_id": "abc", "side": "BUY", "price": "0.5"})
        self.assertIsInstance(sig, str)
        self.assertTrue(len(sig) > 64)

    def test_sign_order_deterministic(self):
        data = {"token_id": "abc", "side": "BUY", "price": "0.5"}
        sig1 = self.auth.sign_order(data)
        sig2 = self.auth.sign_order(data)
        self.assertEqual(sig1, sig2)


# ─────────────────────────────────────────
#  2. COMPUTE QUOTES TESTS
# ─────────────────────────────────────────
class TestComputeQuotes(unittest.TestCase):

    def test_buy_below_mid(self):
        buy, sell = compute_quotes(Decimal("0.40"), Decimal("0.60"))
        mid = Decimal("0.50")
        self.assertLess(buy, mid)

    def test_sell_above_mid(self):
        buy, sell = compute_quotes(Decimal("0.40"), Decimal("0.60"))
        mid = Decimal("0.50")
        self.assertGreater(sell, mid)

    def test_spread_is_approximately_spread_pct(self):
        best_bid = Decimal("0.45")
        best_ask = Decimal("0.55")
        buy, sell = compute_quotes(best_bid, best_ask)
        mid = (best_bid + best_ask) / 2
        expected_spread = mid * SPREAD_PCT
        actual_spread = sell - buy
        self.assertAlmostEqual(float(actual_spread), float(expected_spread), delta=0.005)

    def test_prices_within_valid_range(self):
        for bid, ask in [
            (Decimal("0.01"), Decimal("0.99")),
            (Decimal("0.49"), Decimal("0.51")),
            (Decimal("0.90"), Decimal("0.95")),
        ]:
            buy, sell = compute_quotes(bid, ask)
            self.assertGreaterEqual(buy,  Decimal("0.01"))
            self.assertLessEqual(sell,    Decimal("0.99"))
            self.assertGreaterEqual(sell, Decimal("0.01"))

    def test_extreme_low_prices_clamped(self):
        buy, sell = compute_quotes(Decimal("0.001"), Decimal("0.002"))
        self.assertGreaterEqual(buy,  Decimal("0.01"))
        self.assertGreaterEqual(sell, Decimal("0.01"))

    def test_extreme_high_prices_clamped(self):
        buy, sell = compute_quotes(Decimal("0.998"), Decimal("0.999"))
        self.assertLessEqual(sell, Decimal("0.99"))


# ─────────────────────────────────────────
#  3. RISK GUARD TESTS
# ─────────────────────────────────────────
class TestRiskGuard(unittest.TestCase):

    def setUp(self):
        self.risk = RiskGuard()

    def test_initial_state(self):
        self.assertEqual(self.risk.daily_loss,    Decimal("0"))
        self.assertEqual(self.risk.total_capital, Decimal("0"))

    def test_can_enter_market_initially(self):
        self.assertTrue(self.risk.can_enter_market())

    def test_can_enter_market_blocked_when_capital_full(self):
        self.risk.add_capital(MAX_CAPITAL_USDC)
        self.assertFalse(self.risk.can_enter_market())

    def test_add_and_remove_capital(self):
        self.risk.add_capital(Decimal("50"))
        self.assertEqual(self.risk.total_capital, Decimal("50"))
        self.risk.remove_capital(Decimal("30"))
        self.assertEqual(self.risk.total_capital, Decimal("20"))

    def test_remove_capital_floor_zero(self):
        self.risk.remove_capital(Decimal("999"))
        self.assertEqual(self.risk.total_capital, Decimal("0"))

    def test_daily_loss_accumulates(self):
        self.risk.check_daily_loss(Decimal("-5"))
        self.assertEqual(self.risk.daily_loss, Decimal("5"))
        self.risk.check_daily_loss(Decimal("-3"))
        self.assertEqual(self.risk.daily_loss, Decimal("8"))

    def test_positive_pnl_does_not_increase_loss(self):
        self.risk.check_daily_loss(Decimal("10"))
        self.assertEqual(self.risk.daily_loss, Decimal("0"))

    def test_daily_loss_limit_returns_false(self):
        result = self.risk.check_daily_loss(-DAILY_LOSS_LIMIT - Decimal("1"))
        self.assertFalse(result)

    def test_below_daily_loss_limit_returns_true(self):
        result = self.risk.check_daily_loss(Decimal("-1"))
        self.assertTrue(result)

    def test_day_reset(self):
        from datetime import date, timedelta
        self.risk.daily_loss = DAILY_LOSS_LIMIT - Decimal("1")
        self.risk._reset_day = date.today() - timedelta(days=1)
        result = self.risk.check_daily_loss(Decimal("0"))
        self.assertEqual(self.risk.daily_loss, Decimal("0"))
        self.assertTrue(result)


# ─────────────────────────────────────────
#  4. MARKET POSITION TESTS
# ─────────────────────────────────────────
class TestMarketPosition(unittest.TestCase):

    def _make_pos(self):
        return MarketPosition(
            condition_id="cond_123",
            yes_token="yes_token_abc",
            no_token="no_token_def",
        )

    def test_default_values(self):
        pos = self._make_pos()
        self.assertEqual(pos.yes_cost,   Decimal("0"))
        self.assertEqual(pos.no_cost,    Decimal("0"))
        self.assertFalse(pos.locked)
        self.assertEqual(pos.realized_pnl, Decimal("0"))

    def test_orders_dict_is_independent(self):
        pos1 = self._make_pos()
        pos2 = self._make_pos()
        pos1.orders["x"] = OrderState("x", "tok", "BUY", Decimal("0.5"), Decimal("10"))
        self.assertNotIn("x", pos2.orders)


# ─────────────────────────────────────────
#  5. HANDLE BOTH FILLED
# ─────────────────────────────────────────
class TestHandleBothFilled(unittest.TestCase):

    def _make_pos(self, yes_cost, no_cost):
        pos = MarketPosition(
            condition_id="cond_test",
            yes_token="yes_tok",
            no_token="no_tok",
            yes_cost=Decimal(str(yes_cost)),
            no_cost=Decimal(str(no_cost)),
        )
        return pos

    def test_profit_scenario(self):
        pos = self._make_pos("0.45", "0.48")
        run(handle_both_filled(pos))
        self.assertTrue(pos.locked)
        self.assertAlmostEqual(float(pos.realized_pnl), 1.0 - 0.93, places=4)

    def test_loss_scenario(self):
        pos = self._make_pos("0.55", "0.52")
        run(handle_both_filled(pos))
        self.assertTrue(pos.locked)
        self.assertAlmostEqual(float(pos.realized_pnl), 1.0 - 1.07, places=4)

    def test_breakeven_scenario(self):
        pos = self._make_pos("0.50", "0.50")
        run(handle_both_filled(pos))
        self.assertTrue(pos.locked)
        self.assertAlmostEqual(float(pos.realized_pnl), 0.0, places=4)

    def test_pnl_formula(self):
        yes_c, no_c = Decimal("0.48"), Decimal("0.49")
        pos = self._make_pos(str(yes_c), str(no_c))
        run(handle_both_filled(pos))
        expected = Decimal("1") - (yes_c + no_c)
        self.assertEqual(pos.realized_pnl, expected)


# ─────────────────────────────────────────
#  6. HEDGE POSITION TESTS
# ─────────────────────────────────────────
class TestHedgePosition(unittest.TestCase):

    def _make_client(self):
        client = MagicMock(spec=ClobClient)
        client.place_order = AsyncMock(return_value="hedge_order_id_001")
        return client

    def _make_pos(self):
        return MarketPosition(
            condition_id="cond_hedge",
            yes_token="yes_tok",
            no_token="no_tok",
        )

    def test_yes_fill_updates_yes_cost(self):
        client = self._make_client()
        pos    = self._make_pos()
        run(hedge_position(client, pos, "YES", Decimal("10"), Decimal("0.48")))
        self.assertEqual(pos.yes_cost,   Decimal("10") * Decimal("0.48"))
        self.assertEqual(pos.yes_shares, Decimal("10"))

    def test_no_fill_updates_no_cost(self):
        client = self._make_client()
        pos    = self._make_pos()
        run(hedge_position(client, pos, "NO", Decimal("10"), Decimal("0.52")))
        self.assertEqual(pos.no_cost,   Decimal("10") * Decimal("0.52"))
        self.assertEqual(pos.no_shares, Decimal("10"))

    def test_hedge_order_placed(self):
        client = self._make_client()
        pos    = self._make_pos()
        run(hedge_position(client, pos, "YES", Decimal("5"), Decimal("0.45")))
        client.place_order.assert_called_once()
        args = client.place_order.call_args
        self.assertEqual(args[0][0], pos.no_token)

    def test_position_locked_when_cost_below_102(self):
        client = self._make_client()
        pos    = self._make_pos()
        pos.yes_cost   = Decimal("0.45")
        pos.yes_shares = Decimal("1")
        pos.no_shares  = Decimal("1")
        run(hedge_position(client, pos, "NO", Decimal("1"), Decimal("0.50")))
        self.assertTrue(pos.locked)

    def test_position_not_locked_when_cost_above_102(self):
        client = self._make_client()
        pos    = self._make_pos()
        pos.yes_cost   = Decimal("0.65")
        pos.yes_shares = Decimal("1")
        pos.no_shares  = Decimal("0")
        run(hedge_position(client, pos, "YES", Decimal("1"), Decimal("0.60")))
        self.assertFalse(pos.locked)


# ─────────────────────────────────────────
#  7. COMPUTE BOOK LIQUIDITY TESTS
# ─────────────────────────────────────────
class TestComputeBookLiquidity(unittest.TestCase):

    def test_empty_book(self):
        result = run(compute_book_liquidity({"bids": [], "asks": []}))
        self.assertEqual(result, 0.0)

    def test_single_level(self):
        book = {"bids": [{"price": "0.5", "size": "100"}], "asks": []}
        result = run(compute_book_liquidity(book))
        self.assertAlmostEqual(result, 50.0)

    def test_multiple_levels(self):
        book = {
            "bids": [
                {"price": "0.50", "size": "100"},
                {"price": "0.49", "size": "200"},
            ],
            "asks": [
                {"price": "0.51", "size": "150"},
            ],
        }
        result = run(compute_book_liquidity(book))
        expected = 0.50 * 100 + 0.49 * 200 + 0.51 * 150
        self.assertAlmostEqual(result, expected, places=2)

    def test_malformed_book_returns_zero(self):
        result = run(compute_book_liquidity({}))
        self.assertEqual(result, 0.0)

    def test_top_5_levels_used(self):
        bids = [{"price": str(0.5 - i * 0.01), "size": "10"} for i in range(10)]
        book = {"bids": bids, "asks": []}
        result = run(compute_book_liquidity(book))
        expected = sum((0.5 - i * 0.01) * 10 for i in range(5))
        self.assertAlmostEqual(result, expected, places=2)


# ─────────────────────────────────────────
#  8. ESTIMATE REWARD APR TESTS
# ─────────────────────────────────────────
class TestEstimateRewardAPR(unittest.TestCase):

    def _mock_client(self, rewards_data):
        client = MagicMock(spec=ClobClient)
        client.get_rewards = AsyncMock(return_value=rewards_data)
        return client

    def test_basic_apr_calculation(self):
        client = self._mock_client({
            "rewardsPerDay": 100,
            "totalLiquidity": 10000,
        })
        apr = run(estimate_reward_apr(client, "cond_id", "yes_tok"))
        self.assertAlmostEqual(apr, (100 / 10000) * 365 * 100, places=1)

    def test_zero_liquidity_returns_zero(self):
        client = self._mock_client({
            "rewardsPerDay": 100,
            "totalLiquidity": 0,
        })
        apr = run(estimate_reward_apr(client, "cond_id", "yes_tok"))
        self.assertEqual(apr, 0.0)

    def test_empty_rewards_returns_zero(self):
        client = self._mock_client({})
        apr = run(estimate_reward_apr(client, "cond_id", "yes_tok"))
        self.assertEqual(apr, 0.0)

    def test_none_rewards_returns_zero(self):
        client = self._mock_client(None)
        apr = run(estimate_reward_apr(client, "cond_id", "yes_tok"))
        self.assertEqual(apr, 0.0)

    def test_exception_returns_zero(self):
        client = MagicMock(spec=ClobClient)
        client.get_rewards = AsyncMock(side_effect=Exception("network error"))
        apr = run(estimate_reward_apr(client, "cond_id", "yes_tok"))
        self.assertEqual(apr, 0.0)

    def test_high_reward_high_apr(self):
        client = self._mock_client({
            "rewardsPerDay": 1000,
            "totalLiquidity": 1000,
        })
        apr = run(estimate_reward_apr(client, "cond_id", "yes_tok"))
        self.assertGreater(apr, MIN_REWARD_APR)


# ─────────────────────────────────────────
#  9. FILL LISTENER TESTS
# ─────────────────────────────────────────
class TestFillListener(unittest.TestCase):

    def _make_listener(self, positions=None):
        auth   = PolyAuth(MOCK_KEY)
        client = MagicMock(spec=ClobClient)
        client._api_key        = "test_key"
        client._api_secret     = "test_secret_key_32chars_padded!!"
        client._api_passphrase = "test_pass"
        client.place_order = AsyncMock(return_value="hedge_id_999")
        pos_dict = positions or {}
        return FillListener(client, pos_dict, auth), client

    def test_non_trade_event_ignored(self):
        listener, client = self._make_listener()
        raw = json.dumps([{"event_type": "book_update", "data": {}}])
        run(listener._handle(raw))
        client.place_order.assert_not_called()

    def test_fill_triggers_hedge_when_one_side_filled(self):
        pos = MarketPosition(
            condition_id="cond_fill",
            yes_token="tok_yes",
            no_token="tok_no",
        )
        order = OrderState(
            order_id="order_001",
            token_id="tok_yes",
            side="BUY",
            price=Decimal("0.48"),
            size=Decimal("10"),
        )
        pos.orders["order_001"] = order

        listener, client = self._make_listener({"cond_fill": pos})

        msg = json.dumps([{
            "event_type": "trade",
            "orderID": "order_001",
            "assetId": "tok_yes",
            "side": "BUY",
            "price": "0.48",
            "size": "10",
        }])
        run(listener._handle(msg))
        client.place_order.assert_called_once()

    def test_fill_on_unknown_order_does_nothing(self):
        pos = MarketPosition(
            condition_id="cond_unk",
            yes_token="tok_yes",
            no_token="tok_no",
        )
        listener, client = self._make_listener({"cond_unk": pos})
        msg = json.dumps([{
            "event_type": "trade",
            "orderID": "unknown_order_xyz",
            "assetId": "tok_yes",
            "side": "BUY",
            "price": "0.50",
            "size": "5",
        }])
        run(listener._handle(msg))
        client.place_order.assert_not_called()

    def test_fill_on_locked_position_ignored(self):
        pos = MarketPosition(
            condition_id="cond_locked",
            yes_token="tok_yes",
            no_token="tok_no",
            locked=True,
        )
        order = OrderState(
            order_id="order_locked",
            token_id="tok_yes",
            side="BUY",
            price=Decimal("0.5"),
            size=Decimal("10"),
        )
        pos.orders["order_locked"] = order
        listener, client = self._make_listener({"cond_locked": pos})
        msg = json.dumps([{
            "event_type": "trade",
            "orderID": "order_locked",
            "assetId": "tok_yes",
            "side": "BUY",
            "price": "0.5",
            "size": "10",
        }])
        run(listener._handle(msg))
        client.place_order.assert_not_called()

    def test_malformed_json_handled_gracefully(self):
        listener, client = self._make_listener()
        run(listener._handle("{invalid json}"))
        client.place_order.assert_not_called()

    def test_both_sides_fill_locks_position(self):
        pos = MarketPosition(
            condition_id="cond_both",
            yes_token="tok_yes",
            no_token="tok_no",
        )
        for oid, tok, sz in [
            ("ord_yes", "tok_yes", Decimal("10")),
            ("ord_no",  "tok_no",  Decimal("10")),
        ]:
            o = OrderState(oid, tok, "BUY", Decimal("0.5"), sz)
            o.filled = sz
            pos.orders[oid] = o

        listener, client = self._make_listener({"cond_both": pos})

        msg = json.dumps([{
            "event_type": "trade",
            "orderID": "ord_yes",
            "assetId": "tok_yes",
            "side": "BUY",
            "price": "0.5",
            "size": "10",
        }])
        run(listener._handle(msg))
        self.assertTrue(pos.locked)


# ─────────────────────────────────────────
#  10. CLOB CLIENT L2 HEADER TESTS
# ─────────────────────────────────────────
class TestClobClientL2Headers(unittest.TestCase):

    def setUp(self):
        self.auth   = PolyAuth(MOCK_KEY)
        self.client = ClobClient(self.auth)
        self.client._api_key        = "test_api_key"
        self.client._api_secret     = "test_secret_32chars_____________"
        self.client._api_passphrase = "test_passphrase"

    def test_l2_headers_keys_present(self):
        headers = self.client._get_l2_headers("GET", "/order")
        for key in ("POLY-API-KEY", "POLY-SIGNATURE", "POLY-TIMESTAMP", "POLY-PASSPHRASE"):
            self.assertIn(key, headers)

    def test_l2_headers_api_key_correct(self):
        headers = self.client._get_l2_headers("POST", "/order", "{}")
        self.assertEqual(headers["POLY-API-KEY"], "test_api_key")

    def test_l2_headers_passphrase_correct(self):
        headers = self.client._get_l2_headers("DELETE", "/orders")
        self.assertEqual(headers["POLY-PASSPHRASE"], "test_passphrase")

    def test_l2_headers_signature_is_base64(self):
        import base64
        headers = self.client._get_l2_headers("GET", "/order")
        sig = headers["POLY-SIGNATURE"]
        try:
            base64.b64decode(sig)
            valid = True
        except Exception:
            valid = False
        self.assertTrue(valid)

    def test_l2_headers_timestamp_is_numeric_string(self):
        headers = self.client._get_l2_headers("GET", "/order")
        ts = headers["POLY-TIMESTAMP"]
        self.assertTrue(ts.isdigit())
        self.assertGreater(int(ts), 1_000_000_000)


# ─────────────────────────────────────────
#  11. JSON LOGGER TESTS
# ─────────────────────────────────────────
class TestJLog(unittest.TestCase):

    def test_jlog_writes_valid_json(self):
        m = mock_open()
        with patch("builtins.open", m):
            jlog("test_event", {"key": "value", "num": 42})
        handle = m()
        written = handle.write.call_args[0][0]
        record = json.loads(written.strip())
        self.assertEqual(record["event"], "test_event")
        self.assertEqual(record["key"],   "value")
        self.assertEqual(record["num"],   42)
        self.assertIn("ts", record)

    def test_jlog_appends_newline(self):
        m = mock_open()
        with patch("builtins.open", m):
            jlog("ev", {})
        handle = m()
        written = handle.write.call_args[0][0]
        self.assertTrue(written.endswith("\n"))

    def test_jlog_opens_in_append_mode(self):
        m = mock_open()
        with patch("builtins.open", m):
            jlog("ev", {})
        m.assert_called_once_with(main.LOG_FILE, "a")


# ─────────────────────────────────────────
#  12. INTEGRATION: BOT INIT TESTS
# ─────────────────────────────────────────
class TestBotInit(unittest.TestCase):

    def test_missing_private_key_exits(self):
        with patch.dict(os.environ, {"PRIVATE_KEY": ""}):
            with self.assertRaises(SystemExit):
                main.PolymarketMMBot()

    def test_bot_creates_positions_dict(self):
        bot = main.PolymarketMMBot()
        self.assertIsInstance(bot.positions, dict)
        self.assertEqual(len(bot.positions), 0)

    def test_bot_creates_risk_guard(self):
        bot = main.PolymarketMMBot()
        self.assertIsInstance(bot.risk, RiskGuard)

    def test_bot_address_matches_key(self):
        bot = main.PolymarketMMBot()
        expected = PolyAuth(MOCK_KEY).address
        self.assertEqual(bot.auth.address, expected)


# ─────────────────────────────────────────
#  RUNNER
# ─────────────────────────────────────────
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        TestPolyAuth,
        TestComputeQuotes,
        TestRiskGuard,
        TestMarketPosition,
        TestHandleBothFilled,
        TestHedgePosition,
        TestComputeBookLiquidity,
        TestEstimateRewardAPR,
        TestFillListener,
        TestClobClientL2Headers,
        TestJLog,
        TestBotInit,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
