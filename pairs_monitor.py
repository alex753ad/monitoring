import streamlit as st
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════
# v8.0: CLUSTER DETECTION + CONTINUOUS THRESHOLD + HURST HARD GATE
# ═══════════════════════════════════════════════════════

def assess_entry_readiness(p):
    """
    Оценка готовности к входу. Единая логика для сканера и монитора.
    
    v8.0 ИЗМЕНЕНИЯ:
      - HARD GATE: Hurst ≥ 0.45 → max УСЛОВНО (никогда 🟢 ВХОД!)
        Нет подтверждения mean reversion → нельзя доверять сигналу
      - Hurst=0.500 (DFA fallback) → max СЛАБЫЙ (данных недостаточно)
    
    Обязательные (🟢 все должны быть True):
      1. Статус ≥ READY   2. |Z| ≥ Thr   3. Q ≥ 50   4. Dir ≠ NONE
    Желательные (🔵):
      5. FDR✅  6. Conf=HIGH  7. S≥60  8. ρ≥0.5  9. Stab≥3/4  10. Hurst<0.35
    FDR bypass (🟡): Q≥70 + Stab≥3/4 + ADF✅ + Hurst<0.35
    """
    mandatory = [
        ('Статус ≥ READY', p.get('signal', 'NEUTRAL') in ('SIGNAL', 'READY'), p.get('signal', 'NEUTRAL')),
        ('|Z| ≥ Thr', abs(p.get('zscore', 0)) >= p.get('threshold', 2.0),
         f"|{p.get('zscore',0):.2f}| vs {p.get('threshold',2.0)}"),
        ('Q ≥ 50', p.get('quality_score', 0) >= 50, f"Q={p.get('quality_score', 0)}"),
        ('Dir ≠ NONE', p.get('direction', 'NONE') != 'NONE', p.get('direction', 'NONE')),
    ]
    all_mandatory = all(m[1] for m in mandatory)
    
    fdr_ok = p.get('fdr_passed', False)
    stab_ok = p.get('stability_passed', 0) >= 3
    hurst_val = p.get('hurst', 0.5)
    hurst_ok = hurst_val < 0.35
    hurst_is_fallback = p.get('hurst_is_fallback', False) or hurst_val == 0.5
    
    optional = [
        ('FDR ✅', fdr_ok, '✅' if fdr_ok else '❌'),
        ('Conf=HIGH', p.get('confidence', 'LOW') == 'HIGH', p.get('confidence', 'LOW')),
        ('S ≥ 60', p.get('signal_score', 0) >= 60, f"S={p.get('signal_score', 0)}"),
        ('ρ ≥ 0.5', p.get('correlation', 0) >= 0.5, f"ρ={p.get('correlation', 0):.2f}"),
        ('Stab ≥ 3/4', stab_ok, f"{p.get('stability_passed',0)}/{p.get('stability_total',4)}"),
        ('Hurst < 0.35', hurst_ok, f"H={hurst_val:.3f}"),
    ]
    opt_count = sum(1 for _, met, _ in optional if met)
    
    fdr_bypass = (not fdr_ok and p.get('quality_score', 0) >= 70 and
                  stab_ok and p.get('adf_passed', False) and hurst_ok)
    
    if all_mandatory:
        # v8.0: HARD HURST GATES — без MR-подтверждения вход рискованный
        if hurst_is_fallback:
            # DFA fallback (0.500) — данных недостаточно для оценки MR
            level, label = 'CONDITIONAL', '🟡 СЛАБЫЙ ⚠️H=0.5'
        elif hurst_val >= 0.45:
            # Hurst подтверждает: спред НЕ mean-reverting
            level, label = 'CONDITIONAL', '🟡 УСЛОВНО ⚠️H≥0.45'
        elif opt_count >= 4:
            level, label = 'ENTRY', '🟢 ВХОД'
        elif opt_count >= 2 or fdr_bypass:
            level, label = 'CONDITIONAL', '🟡 УСЛОВНО'
        else:
            level, label = 'CONDITIONAL', '🟡 СЛАБЫЙ'
    else:
        level, label = 'WAIT', '⚪ ЖДАТЬ'
    
    return {'level': level, 'label': label, 'all_mandatory': all_mandatory,
            'mandatory': mandatory, 'optional': optional,
            'fdr_bypass': fdr_bypass, 'opt_count': opt_count}

# Импорт модуля mean reversion analysis v10.5
from mean_reversion_analysis import (
    calculate_hurst_exponent,
    calculate_rolling_zscore,
    calculate_adaptive_robust_zscore,
    calculate_crossing_density,
    calculate_rolling_correlation,
    calculate_ou_parameters,
    calculate_ou_score,
    calculate_quality_score,
    calculate_signal_score,
    calculate_trade_score,
    calculate_confidence,
    get_adaptive_signal,
    sanitize_pair,
    kalman_hedge_ratio,
    kalman_select_delta,
    apply_fdr_correction,
    check_cointegration_stability,
    adf_test_spread,
    estimate_exit_time,
    validate_ou_quality
)
from statsmodels.tools import add_constant

# Конфигурация страницы
st.set_page_config(
    page_title="Crypto Pairs Trading Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .signal-long {
        color: #00cc00;
        font-weight: bold;
    }
    .signal-short {
        color: #ff0000;
        font-weight: bold;
    }
    .signal-neutral {
        color: #888888;
    }
    /* Исправление читаемости для темной темы */
    .stMarkdown, .stText, p, span, div {
        color: inherit !important;
    }
    /* Таблица - темный текст на светлом фоне для читаемости */
    .dataframe {
        background-color: white !important;
        color: black !important;
    }
    .dataframe td, .dataframe th {
        color: black !important;
    }
    /* Метрики - улучшенная видимость */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
    }
    /* v6.0: Entry readiness */
    .entry-ready { 
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        color: white; padding: 12px; border-radius: 8px; 
        text-align: center; font-weight: bold; font-size: 1.1em;
        margin: 8px 0; border: 2px solid #4caf50;
    }
    .entry-conditional {
        background: linear-gradient(135deg, #e65100 0%, #f57c00 100%);
        color: white; padding: 12px; border-radius: 8px;
        text-align: center; font-weight: bold; font-size: 1.1em;
        margin: 8px 0; border: 2px solid #ff9800;
    }
    .entry-wait {
        background: #424242; color: #bdbdbd; padding: 12px; border-radius: 8px;
        text-align: center; font-size: 1.1em; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Инициализация session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'pairs_data' not in st.session_state:
    st.session_state.pairs_data = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'selected_pair_index' not in st.session_state:
    st.session_state.selected_pair_index = int(0)
if 'settings' not in st.session_state:
    # Сохранение последних настроек
    st.session_state.settings = {
        'exchange': 'okx',          # OKX по умолчанию (v7.1: Binance/Bybit заблокированы на облачных серверах)
        'timeframe': '4h',          # 4h таймфрейм
        'lookback_days': 90,        # 90 дней (v9: увеличен для надёжности DFA и Kalman)
        'top_n_coins': 150,         # 150 монет (v7.1: больше монет = больше комбинаций пар)
        'max_pairs_display': 30,    # 30 пар максимум
        'pvalue_threshold': 0.03,   # 0.03
        'zscore_threshold': 2.3,    # 2.3
        'max_halflife_hours': 28,   # 28 часов
        'hide_stablecoins': True,   # v10.4: скрыть стейблкоины / LST / wrapped
        'corr_prefilter': 0.3,      # v10.4: пропускать пары с |ρ| < порога (0=выкл)
    }

# v10.4: Стейблкоины, LST и wrapped-токены (торговля невыгодна из-за узкого спреда)
STABLE_LST_TOKENS = {
    'USDC', 'USDT', 'DAI', 'USDG', 'TUSD', 'BUSD', 'FDUSD', 'PYUSD',  # stablecoins
    'STETH', 'BETH', 'CBETH', 'RETH', 'WSTETH', 'METH',                 # ETH LST
    'JITOSOL', 'MSOL', 'BNSOL',                                          # SOL LST
    'WBTC', 'TBTC',                                                       # wrapped BTC
    'XAUT', 'PAXG',                                                       # gold tokens
}

class CryptoPairsScanner:
    # v7.1: Порядок fallback — OKX и KuCoin работают с HuggingFace/облачных серверов
    # Binance и Bybit блокируют CloudFront (403 Forbidden) с AWS/HF
    FALLBACK_CHAIN = ['okx', 'kucoin', 'bybit', 'binance']
    
    def __init__(self, exchange_name='binance', timeframe='1d', lookback_days=30):
        self.exchange_name = exchange_name
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        
        # v7.1: Умный fallback — пробуем запрошенную биржу, при ошибке перебираем цепочку
        tried = set()
        exchanges_to_try = [exchange_name] + [e for e in self.FALLBACK_CHAIN if e != exchange_name]
        
        last_error = None
        for exch in exchanges_to_try:
            if exch in tried:
                continue
            tried.add(exch)
            try:
                self.exchange = getattr(ccxt, exch)({'enableRateLimit': True})
                self.exchange.load_markets()
                if exch != exchange_name:
                    st.warning(f"⚠️ {exchange_name.upper()} недоступен. Переключился на **{exch.upper()}** ✅")
                self.exchange_name = exch
                return  # Успешно подключились
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if '403' in str(e) or '451' in str(e) or 'forbidden' in error_str or 'restricted' in error_str or 'cloudfront' in error_str:
                    continue  # Гео-блокировка — пробуем следующую
                elif 'timeout' in error_str or 'connection' in error_str:
                    continue  # Сетевая ошибка — пробуем следующую
                else:
                    continue  # Любая ошибка — пробуем следующую
        
        # Все биржи недоступны
        raise Exception(f"❌ Все биржи недоступны. Последняя ошибка: {last_error}")
        
    def get_top_coins(self, limit=100):
        """Получить топ монет по объему торгов"""
        try:
            markets = self.exchange.load_markets()
            tickers = self.exchange.fetch_tickers()
            
            # Определяем базовую валюту в зависимости от биржи
            if self.exchange_name == 'bybit':
                base_currency = 'USDT'
                # Bybit использует формат BTC/USDT:USDT для futures, нам нужен только spot
                usdt_pairs = {k: v for k, v in tickers.items() 
                            if f'/{base_currency}' in k 
                            and ':' not in k  # Исключаем futures
                            and 'info' in v}
            else:
                # Для других бирж (Binance, OKX, etc)
                base_currency = 'USDT'
                usdt_pairs = {k: v for k, v in tickers.items() 
                            if f'/{base_currency}' in k and ':USDT' not in k}
            
            # Сортируем по объему (разные биржи используют разные поля)
            valid_pairs = []
            for symbol, ticker in usdt_pairs.items():
                try:
                    volume = float(ticker.get('quoteVolume', 0)) or float(ticker.get('volume', 0))
                    if volume > 0:
                        valid_pairs.append((symbol, volume))
                except:
                    continue
            
            # Сортируем по объему
            sorted_pairs = sorted(valid_pairs, key=lambda x: x[1], reverse=True)
            
            # Берем топ монет
            top_coins = [pair[0].replace(f'/{base_currency}', '') for pair in sorted_pairs[:limit]]
            
            if len(top_coins) > 0:
                st.info(f"📊 Загружено {len(top_coins)} монет с {self.exchange_name.upper()}")
                return top_coins
            else:
                raise Exception("Не удалось получить данные о монетах")
            
        except Exception as e:
            st.error(f"Ошибка при получении топ монет с {self.exchange_name}: {e}")
            
            # Fallback: возвращаем популярные монеты
            st.warning("🔄 Используется fallback список популярных монет")
            return ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 
                   'MATIC', 'LINK', 'UNI', 'ATOM', 'LTC', 'ETC', 'XLM', 
                   'NEAR', 'APT', 'ARB', 'OP', 'DOGE']
    
    def fetch_ohlcv(self, symbol, limit=None):
        """Получить исторические данные"""
        try:
            if limit is None:
                # Конвертируем дни в количество баров
                bars_per_day = {'1h': 24, '4h': 6, '1d': 1, '2h': 12, '15m': 96}.get(self.timeframe, 6)
                limit = self.lookback_days * bars_per_day
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df['close']
        except Exception as e:
            return None
    
    def test_cointegration(self, series1, series2):
        """
        Тест на коинтеграцию v9.0:
          1. Engle-Granger → p-value (статистическая значимость)
          2. Kalman Filter → адаптивный HR + trading spread
          3. Rolling Z-score на Kalman spread
          4. Fallback на OLS если Kalman не сработал
        """
        try:
            valid_data = pd.concat([series1, series2], axis=1).dropna()
            if len(valid_data) < 20:
                return None

            s1 = valid_data.iloc[:, 0]
            s2 = valid_data.iloc[:, 1]

            # 1. Engle-Granger (p-value)
            score, pvalue, _ = coint(s1, s2)

            # 2. Kalman Filter для HR
            kf = kalman_hedge_ratio(s1.values, s2.values, delta=1e-4)

            if kf is not None and not np.isnan(kf['hr_final']) and abs(kf['hr_final']) < 1e6:
                # Kalman path
                hedge_ratio = kf['hr_final']
                intercept = kf['intercept_final']
                spread = pd.Series(kf['spread'], index=s1.index)
                hr_std = kf['hr_std']
                hr_series = kf['hedge_ratios']
                use_kalman = True
            else:
                # Fallback: OLS
                s2_const = add_constant(s2)
                model = OLS(s1, s2_const).fit()
                hedge_ratio = model.params.iloc[1] if len(model.params) > 1 else model.params.iloc[0]
                intercept = model.params.iloc[0] if len(model.params) > 1 else 0.0
                spread = s1 - hedge_ratio * s2 - intercept
                hr_std = 0.0
                hr_series = None
                use_kalman = False

            # 3. Half-life из spread
            spread_lag = spread.shift(1)
            spread_diff = spread - spread_lag
            spread_diff = spread_diff.dropna()
            spread_lag = spread_lag.dropna()
            model_hl = OLS(spread_diff, spread_lag).fit()
            halflife = -np.log(2) / model_hl.params.iloc[0] if model_hl.params.iloc[0] < 0 else np.inf

            # 4. v10: Adaptive Robust Z-score (MAD + HL-зависимое окно)
            hours_per_bar = {'1h': 1, '2h': 2, '4h': 4, '1d': 24,
                             '15m': 0.25}.get(self.timeframe, 4)
            hl_hours = halflife * 24  # halflife в днях → часы
            hl_bars = hl_hours / hours_per_bar if hl_hours < 9999 else None

            zscore, zscore_series, z_window = calculate_adaptive_robust_zscore(
                spread.values, halflife_bars=hl_bars
            )

            # v10.2: Rolling correlation — TF-aware window
            corr_windows = {'1h': 120, '2h': 60, '4h': 60, '1d': 30, '15m': 360}
            corr_w = corr_windows.get(self.timeframe, 60)
            corr_w = min(corr_w, len(s1) // 3)
            corr, corr_series = calculate_rolling_correlation(
                s1.values, s2.values, window=max(10, corr_w)
            )

            return {
                'pvalue': pvalue,
                'zscore': zscore,
                'zscore_series': zscore_series,
                'hedge_ratio': hedge_ratio,
                'intercept': intercept,
                'halflife': halflife,
                'spread': spread,
                'score': score,
                'use_kalman': use_kalman,
                'hr_std': hr_std,
                'hr_series': hr_series,
                'z_window': z_window,
                'correlation': corr,
            }
        except Exception as e:
            return None
    
    def scan_pairs(self, coins, max_pairs=50, progress_bar=None, max_halflife_hours=720,
                   hide_stablecoins=True, corr_prefilter=0.3):
        """Сканировать все пары (v10.5: parallel download + stablecoin filter + correlation pre-filter)"""
        
        # Загружаем данные ПАРАЛЛЕЛЬНО (v10.5: ускорение в 3-8×)
        st.info(f"📥 Загружаю данные для {len(coins)} монет...")
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        price_data = {}
        
        def _fetch_one(coin):
            """Загрузить одну монету (для параллельного вызова)."""
            symbol = f"{coin}/USDT"
            prices = self.fetch_ohlcv(symbol)
            if prices is not None and len(prices) > 20:
                return coin, prices
            return coin, None
        
        # Параллельная загрузка (8 потоков — OKX rate limit ~20 req/sec)
        max_workers = 8
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_one, c): c for c in coins}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                if progress_bar and done_count % 5 == 0:
                    progress_bar.progress(
                        done_count / len(coins) * 0.3,
                        f"📥 Загружено {done_count}/{len(coins)} монет"
                    )
                try:
                    coin, prices = future.result(timeout=30)
                    if prices is not None:
                        price_data[coin] = prices
                except Exception:
                    pass
        
        if len(price_data) < 2:
            st.error("❌ Недостаточно данных для анализа")
            return []
        
        # v10.4: Correlation pre-filter (ускорение в 3-5×)
        skip_pairs = set()
        if corr_prefilter > 0:
            coin_list = list(price_data.keys())
            # Align all series to common length
            min_len = min(len(price_data[c]) for c in coin_list)
            returns_dict = {}
            for c in coin_list:
                p = price_data[c].values[-min_len:]
                r = np.diff(np.log(p + 1e-10))
                returns_dict[c] = r
            
            for i, c1 in enumerate(coin_list):
                for c2 in coin_list[i+1:]:
                    rho = np.corrcoef(returns_dict[c1], returns_dict[c2])[0, 1]
                    if abs(rho) < corr_prefilter:
                        skip_pairs.add((c1, c2))
            
            if skip_pairs:
                total_all = len(coin_list) * (len(coin_list) - 1) // 2
                st.info(f"⚡ Корр. фильтр (|ρ| < {corr_prefilter}): пропущено {len(skip_pairs)}/{total_all} пар")
        
        # v10.4: Stablecoin/LST filter  
        stable_skipped = 0
        
        total_combinations = len(price_data) * (len(price_data) - 1) // 2
        st.info(f"🔍 Фаза 1: Коинтеграция для {total_combinations} пар из {len(price_data)} монет...")
        processed = 0
        
        # ═══════ ФАЗА 1: Быстрый тест коинтеграции для ВСЕХ пар ═══════
        # Собираем ВСЕ p-values (ключевое исправление FDR!)
        all_pvalues = []
        candidates = []  # (coin1, coin2, result) для пар с p < 0.10
        
        for i, coin1 in enumerate(price_data.keys()):
            for coin2 in list(price_data.keys())[i+1:]:
                processed += 1
                if progress_bar:
                    progress_bar.progress(
                        0.3 + processed / total_combinations * 0.35,  # Фаза 1 = 30-65%
                        f"Фаза 1: {processed}/{total_combinations}"
                    )
                
                # v10.4: Skip stablecoin/LST pairs (both coins must be stable to skip)
                if hide_stablecoins:
                    if coin1 in STABLE_LST_TOKENS and coin2 in STABLE_LST_TOKENS:
                        all_pvalues.append(1.0)
                        stable_skipped += 1
                        continue
                    # Пары типа ETH/STETH, SOL/JITOSOL — один актив + его LST
                    c1u, c2u = coin1.upper(), coin2.upper()
                    if (c1u in c2u or c2u in c1u) and (coin1 in STABLE_LST_TOKENS or coin2 in STABLE_LST_TOKENS):
                        all_pvalues.append(1.0)
                        stable_skipped += 1
                        continue
                
                # v10.4: Skip uncorrelated pairs (pre-filter)
                if (coin1, coin2) in skip_pairs:
                    all_pvalues.append(1.0)
                    continue
                
                result = self.test_cointegration(price_data[coin1], price_data[coin2])
                
                if result:
                    all_pvalues.append(result['pvalue'])
                    
                    # Сохраняем кандидатов (p < 0.15 для запаса — v10 relaxed)
                    halflife_hours = result['halflife'] * 24
                    if result['pvalue'] < 0.15 and halflife_hours <= max_halflife_hours:
                        candidates.append((coin1, coin2, result, len(all_pvalues) - 1))
                else:
                    all_pvalues.append(1.0)  # Не удалось — p=1
        
        # ═══════ FDR на ВСЕХ p-values ═══════
        if len(all_pvalues) == 0:
            return []
        
        adj_pvalues, fdr_rejected = apply_fdr_correction(all_pvalues, alpha=0.05)
        
        total_fdr_passed = int(np.sum(fdr_rejected))
        st.info(f"🔬 FDR: {total_fdr_passed} из {len(all_pvalues)} пар прошли (α=0.05)")
        if stable_skipped > 0:
            st.info(f"🚫 Пропущено {stable_skipped} стейблкоин/LST пар")
        
        # ═══════ ФАЗА 2: Дорогие метрики только для кандидатов ═══════
        st.info(f"🔍 Фаза 2: Детальный анализ {len(candidates)} кандидатов...")
        results = []
        dt = {'1h': 1/24, '4h': 1/6, '1d': 1}.get(self.timeframe, 1/6)
        
        for idx_c, (coin1, coin2, result, pval_idx) in enumerate(candidates):
            if progress_bar:
                progress_bar.progress(
                    0.65 + (idx_c + 1) / len(candidates) * 0.35,
                    f"Фаза 2: {idx_c + 1}/{len(candidates)}"
                )
            
            fdr_passed = bool(fdr_rejected[pval_idx])
            pvalue_adj = float(adj_pvalues[pval_idx])
            
            # Hurst (DFA)
            hurst = calculate_hurst_exponent(result['spread'])
            hurst_is_fallback = (hurst == 0.5)
            
            # OU
            ou_params = calculate_ou_parameters(result['spread'], dt=dt)
            ou_score = calculate_ou_score(ou_params, hurst)
            is_valid, reason = validate_ou_quality(ou_params, hurst)
            
            # Stability
            stability = check_cointegration_stability(
                price_data[coin1].values, price_data[coin2].values
            )
            
            # v10: количество баров
            n_bars = len(result['spread']) if result.get('spread') is not None else 0
            hr_std_val = result.get('hr_std', 0.0)
            
            # [v10.1] Sanitizer — жёсткие исключения (с min_bars + HR uncertainty)
            san_ok, san_reason = sanitize_pair(
                hedge_ratio=result['hedge_ratio'],
                stability_passed=stability['windows_passed'],
                stability_total=stability['total_windows'],
                zscore=result['zscore'],
                n_bars=n_bars,
                hr_std=hr_std_val
            )
            if not san_ok:
                continue
            
            # [NEW] ADF-тест спреда
            adf = adf_test_spread(result['spread'])
            
            # [v10] Crossing Density — частота пересечений нуля
            crossing_d = calculate_crossing_density(
                result.get('zscore_series', np.array([])),
                window=min(n_bars, 100)
            )
            
            # [v10.1] Confidence (с HR uncertainty)
            confidence, conf_checks, conf_total = calculate_confidence(
                hurst=hurst,
                stability_score=stability['stability_score'],
                fdr_passed=fdr_passed,
                adf_passed=adf['is_stationary'],
                zscore=result['zscore'],
                hedge_ratio=result['hedge_ratio'],
                hurst_is_fallback=hurst_is_fallback,
                hr_std=hr_std_val
            )
            
            # [v10.1] Quality Score (с HR uncertainty penalty)
            q_score, q_breakdown = calculate_quality_score(
                hurst=hurst,
                ou_params=ou_params,
                pvalue_adj=pvalue_adj,
                stability_score=stability['stability_score'],
                hedge_ratio=result['hedge_ratio'],
                adf_passed=adf['is_stationary'],
                hurst_is_fallback=hurst_is_fallback,
                crossing_density=crossing_d,
                n_bars=n_bars,
                hr_std=hr_std_val
            )
            
            # [v8.1] Signal Score (capped by Quality)
            s_score, s_breakdown = calculate_signal_score(
                zscore=result['zscore'],
                ou_params=ou_params,
                confidence=confidence,
                quality_score=q_score
            )
            
            # [v8.0] Adaptive Signal — continuous threshold + hurst
            stab_ratio = stability['stability_score']  # 0.0–1.0
            try:
                state, direction, threshold = get_adaptive_signal(
                    zscore=result['zscore'],
                    confidence=confidence,
                    quality_score=q_score,
                    timeframe=self.timeframe,
                    stability_ratio=stab_ratio,
                    fdr_passed=fdr_passed,
                    hurst=hurst  # v11.0: continuous threshold uses Hurst
                )
            except TypeError:
                # Backward compat — старый модуль без hurst/fdr_passed
                try:
                    state, direction, threshold = get_adaptive_signal(
                        zscore=result['zscore'],
                        confidence=confidence,
                        quality_score=q_score,
                        timeframe=self.timeframe,
                        stability_ratio=stab_ratio,
                        fdr_passed=fdr_passed,
                    )
                except TypeError:
                    state, direction, threshold = get_adaptive_signal(
                        zscore=result['zscore'],
                        confidence=confidence,
                        quality_score=q_score,
                        timeframe=self.timeframe,
                        stability_ratio=stab_ratio,
                    )
            
            halflife_hours = result['halflife'] * 24
            
            # v10: Z-warning
            z_warning = abs(result['zscore']) > 4.0
            
            results.append({
                'pair': f"{coin1}/{coin2}",
                'coin1': coin1,
                'coin2': coin2,
                'pvalue': result['pvalue'],
                'pvalue_adj': pvalue_adj,
                'fdr_passed': fdr_passed,
                'zscore': result['zscore'],
                'zscore_series': result.get('zscore_series'),
                'hedge_ratio': result['hedge_ratio'],
                'intercept': result.get('intercept', 0.0),
                'halflife_days': result['halflife'],
                'halflife_hours': halflife_hours,
                'spread': result['spread'],
                'signal': state,
                'direction': direction,
                'threshold': threshold,
                'hurst': hurst,
                'hurst_is_fallback': hurst_is_fallback,
                'theta': ou_params['theta'] if ou_params else 0,
                'mu': ou_params['mu'] if ou_params else 0,
                'sigma': ou_params['sigma'] if ou_params else 0,
                'halflife_ou': ou_params['halflife_ou'] * 24 if ou_params else 999,
                'ou_score': ou_score,
                'ou_valid': is_valid,
                'ou_reason': reason,
                'stability_score': stability['stability_score'],
                'stability_passed': stability['windows_passed'],
                'stability_total': stability['total_windows'],
                'is_stable': stability['is_stable'],
                'adf_pvalue': adf['adf_pvalue'],
                'adf_passed': adf['is_stationary'],
                'quality_score': q_score,
                'quality_breakdown': q_breakdown,
                'signal_score': s_score,
                'signal_breakdown': s_breakdown,
                'trade_score': q_score,
                'trade_breakdown': q_breakdown,
                'confidence': confidence,
                'conf_checks': conf_checks,
                'conf_total': conf_total,
                # v9: Kalman
                'use_kalman': result.get('use_kalman', False),
                'hr_std': result.get('hr_std', 0.0),
                'hr_series': result.get('hr_series'),
                # v10: new metrics
                'n_bars': n_bars,
                'z_warning': z_warning,
                'z_window': result.get('z_window', 30),
                'crossing_density': crossing_d,
                'correlation': result.get('correlation', 0.0),
                # v10.1: HR uncertainty ratio
                'hr_uncertainty': (hr_std_val / result['hedge_ratio']
                                   if result['hedge_ratio'] > 0 and hr_std_val > 0
                                   else 0.0),
            })
        
        # Сортируем: v6.0 — сначала по entry readiness, потом по Signal, потом по Quality
        signal_order = {'SIGNAL': 0, 'READY': 1, 'WATCH': 2, 'NEUTRAL': 3}
        entry_order = {'ENTRY': 0, 'CONDITIONAL': 1, 'WAIT': 2}
        
        for r in results:
            ea = assess_entry_readiness(r)
            r['_entry_level'] = ea['level']
            r['_entry_label'] = ea['label']
            r['_fdr_bypass'] = ea['fdr_bypass']
            r['_opt_count'] = ea['opt_count']
            r['_all_mandatory'] = ea['all_mandatory']
        
        results.sort(key=lambda x: (
            entry_order.get(x.get('_entry_level', 'WAIT'), 3),
            signal_order.get(x['signal'], 4),
            -x['quality_score']
        ))
        
        # v10.2: Cluster detection — найти активы, повторяющиеся в 3+ SIGNAL-парах
        signal_pairs = [r for r in results if r['signal'] == 'SIGNAL']
        if signal_pairs:
            from collections import Counter
            coin_counts = Counter()
            for r in signal_pairs:
                coin_counts[r['coin1']] += 1
                coin_counts[r['coin2']] += 1
            # Кластеры: актив в 3+ SIGNAL-парах
            clusters = {coin: count for coin, count in coin_counts.items() if count >= 3}
            # Пометить каждую пару кластером
            for r in results:
                cluster_coins = []
                if r['coin1'] in clusters:
                    cluster_coins.append(f"{r['coin1']}({clusters[r['coin1']]})")
                if r['coin2'] in clusters:
                    cluster_coins.append(f"{r['coin2']}({clusters[r['coin2']]})")
                r['cluster'] = ', '.join(cluster_coins) if cluster_coins else ''
            
            if clusters:
                sorted_clusters = sorted(clusters.items(), key=lambda x: -x[1])
                cluster_msg = ', '.join(f"**{c}** ({n} пар)" for c, n in sorted_clusters)
                st.warning(f"🔗 Кластеры в SIGNAL: {cluster_msg} — это не {sum(clusters.values())} независимых сделок!")
        else:
            for r in results:
                r['cluster'] = ''
        
        if len(results) > 0:
            entry_ready = sum(1 for r in results if r.get('_entry_level') == 'ENTRY')
            entry_cond = sum(1 for r in results if r.get('_entry_level') == 'CONDITIONAL')
            st.success(f"✅ Найдено {len(results)} пар (FDR: {total_fdr_passed}) | 🟢 ВХОД: {entry_ready} | 🟡 УСЛОВНО: {entry_cond}")
        
        return results[:max_pairs]
    
    def get_signal(self, zscore, threshold=2):
        """Определить торговый сигнал"""
        if zscore > threshold:
            return "SHORT"
        elif zscore < -threshold:
            return "LONG"
        else:
            return "NEUTRAL"

def plot_spread_chart(spread_data, pair_name, zscore):
    """График спреда с Z-score"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'Спред пары {pair_name}', 'Z-Score во времени'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # График спреда
    fig.add_trace(
        go.Scatter(x=spread_data.index, y=spread_data.values, 
                  name='Spread', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Средняя линия
    mean = spread_data.mean()
    std = spread_data.std()
    
    fig.add_hline(y=mean, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=mean + 2*std, line_dash="dot", line_color="red", row=1, col=1)
    fig.add_hline(y=mean - 2*std, line_dash="dot", line_color="green", row=1, col=1)
    
    # Z-score график
    zscore_series = (spread_data - mean) / std
    colors = ['red' if z > 2 else 'green' if z < -2 else 'gray' for z in zscore_series]
    
    fig.add_trace(
        go.Scatter(x=zscore_series.index, y=zscore_series.values,
                  name='Z-Score', mode='lines+markers',
                  line=dict(color='purple'), marker=dict(size=4)),
        row=2, col=1
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=2, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=-2, line_dash="dot", line_color="green", row=2, col=1)
    
    fig.update_xaxes(title_text="Дата", row=2, col=1)
    fig.update_yaxes(title_text="Спред", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True, hovermode='x unified')
    
    return fig

# === ИНТЕРФЕЙС ===

st.markdown('<p class="main-header">🔍 Crypto Pairs Trading Scanner</p>', unsafe_allow_html=True)
st.caption("Версия 8.0 | Continuous Threshold + Hurst Hard Gate + Cluster Detection + Direction Conflicts")
st.markdown("---")

# Sidebar - настройки
with st.sidebar:
    st.header("⚙️ Настройки")
    
    exchange = st.selectbox(
        "Биржа",
        ['okx', 'kucoin', 'binance', 'bybit'],
        index=['okx', 'kucoin', 'binance', 'bybit'].index(st.session_state.settings['exchange']),
        help="⚠️ Binance и Bybit блокируют облачные серверы (HuggingFace, Railway). Используйте OKX или KuCoin. Автоматический fallback включён.",
        key='exchange_select'
    )
    st.session_state.settings['exchange'] = exchange
    
    timeframe = st.selectbox(
        "Таймфрейм",
        ['1h', '4h', '1d'],
        index=['1h', '4h', '1d'].index(st.session_state.settings['timeframe']),
        key='timeframe_select'
    )
    st.session_state.settings['timeframe'] = timeframe
    
    lookback_days = st.slider(
        "Период анализа (дней)",
        min_value=7,
        max_value=90,
        value=st.session_state.settings['lookback_days'],
        step=7,
        key='lookback_slider'
    )
    st.session_state.settings['lookback_days'] = lookback_days
    
    top_n_coins = st.slider(
        "Количество монет для анализа",
        min_value=20,
        max_value=200,
        value=st.session_state.settings['top_n_coins'],
        step=10,
        help="Больше монет = больше пар. 100 монет ≈ 4950 пар, 150 ≈ 11000+",
        key='coins_slider'
    )
    st.session_state.settings['top_n_coins'] = top_n_coins
    
    max_pairs_display = st.slider(
        "Максимум пар в результатах",
        min_value=10,
        max_value=100,
        value=st.session_state.settings['max_pairs_display'],
        step=10,
        key='max_pairs_slider'
    )
    st.session_state.settings['max_pairs_display'] = max_pairs_display
    
    st.markdown("---")
    st.subheader("🎯 Фильтры качества")
    
    pvalue_threshold = st.slider(
        "P-value порог",
        min_value=0.01,
        max_value=0.10,
        value=st.session_state.settings['pvalue_threshold'],
        step=0.01,
        key='pvalue_slider'
    )
    st.session_state.settings['pvalue_threshold'] = pvalue_threshold
    
    zscore_threshold = st.slider(
        "Z-score порог для сигнала",
        min_value=1.5,
        max_value=3.0,
        value=st.session_state.settings['zscore_threshold'],
        step=0.1,
        key='zscore_slider'
    )
    st.session_state.settings['zscore_threshold'] = zscore_threshold
    
    st.markdown("---")
    st.subheader("⏱️ Фильтр по времени возврата")
    
    max_halflife_hours = st.slider(
        "Максимальный Half-life (часы)",
        min_value=6,
        max_value=50,  # 50 часов максимум
        value=min(st.session_state.settings['max_halflife_hours'], 50),
        step=2,
        help="Время возврата к среднему. Для 4h: 12-28ч быстрые, 28-50ч стандарт",
        key='halflife_slider'
    )
    st.session_state.settings['max_halflife_hours'] = max_halflife_hours
    
    st.info(f"📊 Текущий фильтр: до {max_halflife_hours} часов ({max_halflife_hours/24:.1f} дней)")
    
    # v10.4: Фильтры мусорных пар
    st.markdown("---")
    st.subheader("🚫 Фильтры пар")
    
    hide_stablecoins = st.checkbox(
        "Скрыть стейблкоины / LST / wrapped",
        value=st.session_state.settings['hide_stablecoins'],
        help="USDC/DAI, ETH/STETH, XAUT/PAXG — идеальная коинтеграция, но спред < 0.5% → убыточно",
        key='hide_stable_chk'
    )
    st.session_state.settings['hide_stablecoins'] = hide_stablecoins
    
    corr_prefilter = st.slider(
        "Корреляционный пре-фильтр",
        min_value=0.0, max_value=0.6, 
        value=st.session_state.settings['corr_prefilter'],
        step=0.05,
        help="Пропускать пары с |ρ| < порога. 0.3 = ускорение 3-5×. 0 = выкл.",
        key='corr_prefilter_slider'
    )
    st.session_state.settings['corr_prefilter'] = corr_prefilter
    
    # НОВОЕ: Фильтры Hurst + OU Process
    st.markdown("---")
    st.subheader("🔬 Mean Reversion Analysis")
    
    st.info("""
    **DFA Hurst** (v6.0):
    • H < 0.35 → Strong mean-reversion ✅
    • H < 0.48 → Mean-reverting ✅
    • H ≈ 0.50 → Random walk ⚪
    • H > 0.55 → Trending ❌
    """)
    
    # Hurst фильтр
    max_hurst = st.slider(
        "Максимальный Hurst",
        min_value=0.0,
        max_value=1.0,
        value=0.55,  # Обновлено для нового метода
        step=0.05,
        help="H < 0.40 = отлично, H < 0.50 = хорошо, H > 0.60 = избегать",
        key='max_hurst'
    )
    
    # OU theta фильтр
    min_theta = st.slider(
        "Минимальная скорость возврата (θ)",
        min_value=0.0,
        max_value=3.0,
        value=0.0,  # Выключен по умолчанию!
        step=0.1,
        help="θ > 1.0 = быстрый возврат. 0.0 = показать все",
        key='min_theta'
    )
    
    # Quality Score фильтр (v8.0)
    min_quality = st.slider(
        "Мин. Quality Score",
        min_value=0, max_value=100, value=0, step=5,
        help="Качество пары (FDR + Stability + Hurst + ADF + HR). 0 = все",
        key='min_quality'
    )
    
    # Signal state фильтр
    signal_filter = st.multiselect(
        "Показывать статусы",
        options=["SIGNAL", "READY", "WATCH", "NEUTRAL"],
        default=["SIGNAL", "READY", "WATCH", "NEUTRAL"],
        help="SIGNAL=вход, READY=почти, WATCH=мониторинг",
        key='signal_filter'
    )
    
    # FDR фильтр
    fdr_only = st.checkbox(
        "Только FDR-подтверждённые",
        value=False,
        help="Только пары, прошедшие Benjamini-Hochberg",
        key='fdr_only'
    )
    
    # Stability фильтр
    stable_only = st.checkbox(
        "Только стабильные пары",
        value=False,
        help="Коинтеграция ≥3/4 подокон",
        key='stable_only'
    )
    
    # v6.0: Entry readiness filter
    st.markdown("---")
    st.subheader("🟢 Готовность к входу")
    entry_filter = st.multiselect(
        "Показывать уровни",
        ["🟢 ВХОД", "🟡 УСЛОВНО", "🟡 СЛАБЫЙ", "⚪ ЖДАТЬ"],
        default=["🟢 ВХОД", "🟡 УСЛОВНО", "🟡 СЛАБЫЙ", "⚪ ЖДАТЬ"],
        key='entry_filter'
    )
    
    auto_refresh = st.checkbox("Автообновление", value=False, key='auto_refresh_check')
    
    if auto_refresh:
        refresh_interval = st.slider(
            "Интервал обновления (минуты)",
            min_value=5,
            max_value=60,
            value=15,
            step=5,
            key='refresh_interval_slider'
        )
    
    st.markdown("---")
    st.markdown("### 📖 Как использовать:")
    st.markdown("""
    1. **Нажмите "Запустить сканер"**
    2. **Дождитесь результатов** (1-3 минуты)
    3. **Найдите пары с сигналами:**
       - 🟢 LONG - покупать первую монету
       - 🔴 SHORT - продавать первую монету
    4. **Проверьте графики** для подтверждения
    5. **Кликните на строку** → откроется анализ
    6. **Добавьте в отслеживание** для мониторинга
    """)
    
    st.markdown("---")

# Основная область
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    if st.button("🚀 Запустить сканер", type="primary", use_container_width=True):
        st.session_state.running = True

with col2:
    if st.button("⏹️ Остановить", use_container_width=True):
        st.session_state.running = False

with col3:
    if st.session_state.last_update:
        st.metric("Последнее обновление", 
                 st.session_state.last_update.strftime("%H:%M:%S"))

# Запуск сканера
if st.session_state.running or (auto_refresh and st.session_state.pairs_data is not None):
    try:
        scanner = CryptoPairsScanner(
            exchange_name=exchange,
            timeframe=timeframe,
            lookback_days=lookback_days
        )
        
        # Прогресс бар
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0, "Инициализация...")
        
        # Получаем топ монеты
        top_coins = scanner.get_top_coins(limit=top_n_coins)
        
        if not top_coins:
            st.error("❌ Не удалось получить список монет. Проверьте подключение к интернету или попробуйте другую биржу.")
            st.session_state.running = False
        else:
            # Сканируем пары
            pairs_results = scanner.scan_pairs(
                top_coins, 
                max_pairs=max_pairs_display, 
                progress_bar=progress_bar,
                max_halflife_hours=max_halflife_hours,
                hide_stablecoins=st.session_state.settings['hide_stablecoins'],
                corr_prefilter=st.session_state.settings['corr_prefilter'],
            )
            
            progress_placeholder.empty()
            
            st.session_state.pairs_data = pairs_results
            st.session_state.last_update = datetime.now()
            st.session_state.running = False  # v7.1: КРИТИЧНО — без этого выбор пары перезапускает скан
            
            if auto_refresh:
                time.sleep(refresh_interval * 60)
                st.rerun()
            
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
        st.info("💡 Попробуйте: уменьшить количество монет, изменить таймфрейм или выбрать другую биржу")
        st.session_state.running = False

# Отображение результатов
if st.session_state.pairs_data is not None:
    pairs = st.session_state.pairs_data
    
    # Фильтрация v8.0
    if 'max_hurst' in st.session_state and 'min_theta' in st.session_state:
        filtered_pairs = []
        for p in pairs:
            if p.get('hurst', 0.5) > st.session_state.max_hurst:
                continue
            if p.get('theta', 0) < st.session_state.min_theta:
                continue
            if st.session_state.get('min_quality', 0) > 0 and p.get('quality_score', 0) < st.session_state.min_quality:
                continue
            if st.session_state.get('signal_filter') and p.get('signal', 'NEUTRAL') not in st.session_state.signal_filter:
                continue
            if st.session_state.get('fdr_only', False) and not p.get('fdr_passed', False):
                continue
            if st.session_state.get('stable_only', False) and not p.get('is_stable', False):
                continue
            # v6.0: Entry readiness filter
            entry_label = p.get('_entry_label', '⚪ ЖДАТЬ')
            ef = st.session_state.get('entry_filter', [])
            if ef and entry_label not in ef:
                continue
            filtered_pairs.append(p)
        
        if len(filtered_pairs) < len(pairs):
            st.info(f"🔬 Фильтры: {len(pairs)} → {len(filtered_pairs)} пар")
        
        pairs = filtered_pairs
    
    if len(pairs) == 0:
        st.warning("⚠️ Коинтегрированных пар не найдено с текущими параметрами")
        st.info("""
        **Попробуйте:**
        - Увеличить период анализа (60-90 дней)
        - Увеличить P-value порог до 0.10
        - Уменьшить количество монет (сфокусироваться на топ-20)
        - Изменить таймфрейм на 4h или 1h
        - Ослабить фильтры Hurst/OU
        - Отключить FDR и Stability фильтры
        """)
    else:
        st.success(f"✅ Найдено {len(pairs)} коинтегрированных пар")
    
        # v6.0: Entry readiness summary
        mc1, mc2, mc3, mc4, mc5, mc6, mc7 = st.columns(7)
        mc1.metric("🟢 ВХОД", sum(1 for p in pairs if p.get('_entry_level') == 'ENTRY'))
        mc2.metric("🟡 УСЛОВНО", sum(1 for p in pairs if p.get('_entry_level') == 'CONDITIONAL'))
        mc3.metric("⚪ ЖДАТЬ", sum(1 for p in pairs if p.get('_entry_level') == 'WAIT'))
        mc4.metric("🔴 SIGNAL", sum(1 for p in pairs if p['signal'] == 'SIGNAL'))
        mc5.metric("🟡 READY", sum(1 for p in pairs if p['signal'] == 'READY'))
        mc6.metric("👁 WATCH", sum(1 for p in pairs if p['signal'] == 'WATCH'))
        mc7.metric("⭐ HIGH conf", sum(1 for p in pairs if p.get('confidence') == 'HIGH'))
        
        st.markdown("---")
        
        # ═══════ v8.0: CLUSTER DETECTION + DIRECTION CONFLICTS ═══════
        signal_pairs = [p for p in pairs if p.get('signal') in ('SIGNAL', 'READY')]
        if signal_pairs:
            from collections import Counter
            coin_count = Counter()
            coin_dirs = {}  # coin → set of actions (LONG/SHORT)
            
            for p in signal_pairs:
                c1, c2 = p['coin1'], p['coin2']
                d = p.get('direction', 'NONE')
                coin_count[c1] += 1
                coin_count[c2] += 1
                if d == 'LONG':
                    coin_dirs.setdefault(c1, set()).add('LONG')
                    coin_dirs.setdefault(c2, set()).add('SHORT')
                elif d == 'SHORT':
                    coin_dirs.setdefault(c1, set()).add('SHORT')
                    coin_dirs.setdefault(c2, set()).add('LONG')
            
            # Cluster warnings
            clusters = [(c, n) for c, n in coin_count.most_common() if n >= 3]
            if clusters:
                cluster_msg = "⚠️ **Кластеры (монета в 3+ парах):** "
                cluster_msg += ", ".join([f"**{c}** ({n} пар)" for c, n in clusters])
                cluster_msg += "\n\nТорговля несколькими парами с одной монетой = завышенный риск. Это НЕ независимые сделки!"
                st.warning(cluster_msg)
            
            # Direction conflicts
            conflicts = [(c, dirs) for c, dirs in coin_dirs.items() if len(dirs) > 1]
            if conflicts:
                conflict_msg = "🚨 **Конфликты направления:**\n"
                for c, dirs in conflicts:
                    conflict_pairs_long = [p['pair'] for p in signal_pairs if (p['coin1'] == c and p.get('direction') == 'LONG') or (p['coin2'] == c and p.get('direction') == 'SHORT')]
                    conflict_pairs_short = [p['pair'] for p in signal_pairs if (p['coin1'] == c and p.get('direction') == 'SHORT') or (p['coin2'] == c and p.get('direction') == 'LONG')]
                    conflict_msg += f"\n**{c}**: LONG в {conflict_pairs_long}, SHORT в {conflict_pairs_short}"
                conflict_msg += "\n\nОдна монета одновременно LONG и SHORT → позиции частично гасят друг друга. Выберите лучшую пару!"
                st.error(conflict_msg)
        
        st.markdown("---")
        
        # Таблица результатов
        st.subheader("📊 Коинтегрированные пары")
        
        st.info("💡 **Кликните на строку** | 🟢 ВХОД = все обязательные ОК | 🟡 УСЛОВНО = обяз. ОК но мало желательных | ⚪ ЖДАТЬ = не входить")
    
    # Проверка что есть пары для отображения
    if len(pairs) > 0:
        df_display = pd.DataFrame([{
            'Пара': p['pair'],
            'Вход': p.get('_entry_label', '⚪ ЖДАТЬ'),
            'Статус': p['signal'],
            'Dir': p.get('direction', ''),
            'Q': p.get('quality_score', 0),
            'S': p.get('signal_score', 0),
            'Conf': p.get('confidence', '?'),
            'Z': round(p['zscore'], 2),
            'Thr': p.get('threshold', 2.0),
            'FDR': ('✅' if p.get('fdr_passed', False) 
                    else ('🟡' if p.get('_fdr_bypass', False) else '❌')),
            'Hurst': round(p.get('hurst', 0.5), 3),
            'Stab': f"{p.get('stability_passed', 0)}/{p.get('stability_total', 4)}",
            'HL': (
                f"{p.get('halflife_hours', p['halflife_days']*24):.1f}ч" 
                if p.get('halflife_hours', p['halflife_days']*24) < 48 
                else '∞'
            ),
            'HR': round(p['hedge_ratio'], 4),
            'ρ': round(p.get('correlation', 0), 2),
            'Opt': f"{p.get('_opt_count', 0)}/6",
        } for p in pairs])
    else:
        df_display = pd.DataFrame(columns=[
            'Пара', 'Вход', 'Статус', 'Dir', 'Q', 'S', 'Conf', 'Z', 'Thr',
            'FDR', 'Hurst', 'Stab', 'HL', 'HR', 'ρ', 'Opt'
        ])
    
    # Функция для выбора строки
    def dataframe_with_selections(df):
        df_with_selections = df.copy()
        df_with_selections.insert(0, "Выбрать", False)
        
        edited_df = st.data_editor(
            df_with_selections,
            hide_index=True,
            column_config={"Выбрать": st.column_config.CheckboxColumn(required=True)},
            disabled=df.columns,
            use_container_width=True
        )
        
        selected_indices = list(np.where(edited_df.Выбрать)[0])
        return selected_indices
    
    selected_rows = dataframe_with_selections(df_display)
    
    if len(selected_rows) > 0:
        st.session_state.selected_pair_index = int(selected_rows[0])  # v7.1: cast np.int64 → int
    
    # Детальный анализ выбранной пары
    if len(pairs) > 0:
        st.markdown("---")
        st.subheader("📈 Детальный анализ пары")
        
        pair_options = [p['pair'] for p in pairs]
        
        # Ограничиваем индекс
        safe_index = int(st.session_state.selected_pair_index)
        if safe_index >= len(pair_options):
            safe_index = 0
        
        # Selectbox с index из session_state (обновляется по checkbox)
        selected_pair = st.selectbox(
            "Выберите пару для анализа:",
            pair_options,
            index=safe_index,
            key='pair_selector_main'
        )
        
        # Синхронизируем обратно
        st.session_state.selected_pair_index = int(pair_options.index(selected_pair))
        
        selected_data = next(p for p in pairs if p['pair'] == selected_pair)
    else:
        # Нет пар — не показываем детальный анализ
        st.info("📊 Запустите сканер для получения результатов")
        st.stop()
    
    # ═══════ v6.0: ENTRY READINESS PANEL ═══════
    ea = assess_entry_readiness(selected_data)
    
    if ea['level'] == 'ENTRY':
        st.markdown(f'<div class="entry-ready">🟢 ГОТОВ К ВХОДУ — все обязательные ОК + {ea["opt_count"]}/6 желательных</div>', unsafe_allow_html=True)
    elif ea['level'] == 'CONDITIONAL':
        st.markdown(f'<div class="entry-conditional">🟡 УСЛОВНЫЙ ВХОД — обязательные ОК, {ea["opt_count"]}/6 желательных</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="entry-wait">⚪ НЕ ВХОДИТЬ — не все обязательные критерии выполнены</div>', unsafe_allow_html=True)
    
    chk1, chk2 = st.columns(2)
    with chk1:
        st.markdown("**🟢 Обязательные (все = ✅):**")
        for name, met, val in ea['mandatory']:
            st.markdown(f"  {'✅' if met else '❌'} **{name}** → `{val}`")
    with chk2:
        st.markdown("**🔵 Желательные (больше = лучше):**")
        for name, met, val in ea['optional']:
            st.markdown(f"  {'✅' if met else '⬜'} {name} → `{val}`")
        if ea['fdr_bypass']:
            st.info("🟡 **FDR bypass:** Q≥70 + Stab≥3/4 + ADF✅ + Hurst<0.35")
    
    # ═══════ ЗАГОЛОВОК С АДАПТИВНЫМ СИГНАЛОМ ═══════
    state = selected_data.get('signal', 'NEUTRAL')
    direction = selected_data.get('direction', 'NONE')
    conf = selected_data.get('confidence', '?')
    threshold = selected_data.get('threshold', 2.0)
    
    state_emoji = {'SIGNAL': '🔴', 'READY': '🟡', 'WATCH': '👁', 'NEUTRAL': '⚪'}.get(state, '⚪')
    conf_emoji = {'HIGH': '⭐', 'MEDIUM': '🔵', 'LOW': '⚫'}.get(conf, '⚫')
    dir_emoji = {'LONG': '🟢↑', 'SHORT': '🔴↓', 'NONE': ''}.get(direction, '')
    
    st.markdown(f"### {state_emoji} **{state}** {dir_emoji} | {conf_emoji} {conf} | **{selected_pair}**")
    
    if state == 'SIGNAL':
        st.success(f"🎯 **ВХОД {direction}** | Z={selected_data['zscore']:.2f} | Порог для этой пары: |Z| ≥ {threshold}")
    elif state == 'READY':
        st.info(f"⏳ **ГОТОВНОСТЬ {direction}** | Z={selected_data['zscore']:.2f} | До порога ({threshold}): {abs(threshold - abs(selected_data['zscore'])):.2f}")
    elif state == 'WATCH':
        st.info(f"👁 **МОНИТОРИНГ** | Z={selected_data['zscore']:.2f} | Порог: {threshold}")
    
    # ⚠️ Предупреждения
    warnings_list = []
    if selected_data.get('hurst_is_fallback', False):
        warnings_list.append("⚠️ Hurst = 0.5 (DFA fallback — данных недостаточно)")
    if abs(selected_data['zscore']) > 5:
        warnings_list.append(f"⚠️ |Z| = {abs(selected_data['zscore']):.1f} > 5 — аномалия")
    elif selected_data.get('z_warning', False):
        warnings_list.append(f"⚠️ |Z| = {abs(selected_data['zscore']):.1f} > 4.0 — приближается к аномалии, возможен структурный сдвиг")
    if not selected_data.get('fdr_passed', False) and not ea.get('fdr_bypass', False):
        warnings_list.append("⚠️ FDR не пройден (и bypass не активен)")
    if not selected_data.get('adf_passed', False):
        warnings_list.append("⚠️ ADF: спред нестационарен")
    n_bars = selected_data.get('n_bars', 0)
    if 0 < n_bars < 100:
        warnings_list.append(f"⚠️ Мало данных: {n_bars} баров (< 100). Результаты менее надёжны")
    if warnings_list:
        st.warning("\n".join(warnings_list))
    
    # ═══════ DUAL SCORE: Quality + Signal ═══════
    q_score = selected_data.get('quality_score', 0)
    s_score = selected_data.get('signal_score', 0)
    q_bd = selected_data.get('quality_breakdown', {})
    s_bd = selected_data.get('signal_breakdown', {})
    
    score_col1, score_col2 = st.columns(2)
    
    with score_col1:
        q_emoji = "🟢" if q_score >= 60 else "🟡" if q_score >= 40 else "🔴"
        st.metric(f"{q_emoji} Quality Score", f"{q_score}/100", 
                  "Надёжность пары")
        if q_bd:
            st.caption(" | ".join([f"{k}:{v}" for k, v in q_bd.items()]))
    
    with score_col2:
        s_emoji = "🟢" if s_score >= 60 else "🟡" if s_score >= 30 else "⚪"
        st.metric(f"{s_emoji} Signal Score", f"{s_score}/100",
                  "Момент входа")
        if s_bd:
            st.caption(" | ".join([f"{k}:{v}" for k, v in s_bd.items()]))
    
    # ═══════ МЕТРИКИ ═══════
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        z_str = f"{selected_data['zscore']:.2f}"
        if selected_data.get('z_warning', False):
            z_str += " ⚠️"
        st.metric("Z-Score", z_str)
    with col2:
        st.metric("P-adj", f"{selected_data.get('pvalue_adj', selected_data['pvalue']):.4f}")
    with col3:
        hl_hours = selected_data.get('halflife_hours', selected_data['halflife_days'] * 24)
        st.metric("Half-life", f"{hl_hours:.1f}ч" if hl_hours < 48 else "∞")
    with col4:
        st.metric("Confidence", f"{conf} ({selected_data.get('conf_checks', 0)}/{selected_data.get('conf_total', 6)})")
    with col5:
        st.metric("Порог Z", f"±{threshold}")
    with col6:
        n_bars = selected_data.get('n_bars', 0)
        bars_emoji = "🟢" if n_bars >= 300 else "🟡" if n_bars >= 100 else "🔴"
        st.metric("Баров", f"{n_bars} {bars_emoji}")
    
    # ═══════ MEAN REVERSION ANALYSIS v8.0 ═══════
    if 'hurst' in selected_data and 'theta' in selected_data:
        st.markdown("---")
        st.subheader("🔬 Mean Reversion Analysis (v10.5)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            hurst = selected_data['hurst']
            if selected_data.get('hurst_is_fallback', False):
                h_st = "⚠️ Fallback"
            elif hurst < 0.35:
                h_st = "🟢 Strong MR"
            elif hurst < 0.48:
                h_st = "🟢 Reverting"
            elif hurst < 0.52:
                h_st = "⚪ Random"
            else:
                h_st = "🔴 Trending"
            st.metric("Hurst (DFA)", f"{hurst:.3f}", h_st)
        
        with col2:
            theta = selected_data['theta']
            t_st = "✅ Быстрый" if theta > 1.0 else "⚠️ Средний" if theta > 0.5 else "❌ Медленный"
            st.metric("θ (Скорость)", f"{theta:.3f}", t_st)
        
        with col3:
            hr = selected_data['hedge_ratio']
            hr_unc = selected_data.get('hr_uncertainty', 0)
            if hr_unc > 0.5:
                hr_st = f"⚠️ ±{hr_unc:.0%}"
            elif hr_unc > 0.2:
                hr_st = f"🟡 ±{hr_unc:.0%}"
            elif hr_unc > 0:
                hr_st = f"✅ ±{hr_unc:.0%}"
            elif 0.2 <= abs(hr) <= 5.0:
                hr_st = "✅ OK"
            else:
                hr_st = "⚠️ Экстрем."
            st.metric("Hedge Ratio", f"{hr:.4f}", hr_st)
        
        with col4:
            if theta > 0:
                exit_time = estimate_exit_time(
                    current_z=selected_data['zscore'], theta=theta, target_z=0.5
                )
                st.metric("Прогноз", f"{exit_time * 24:.1f}ч", "до Z=0.5")
            else:
                st.metric("Прогноз", "∞", "Нет возврата")
        
        # Проверки
        checks_col1, checks_col2 = st.columns(2)
        with checks_col1:
            fdr_s = "✅" if selected_data.get('fdr_passed', False) else "❌"
            adf_s = "✅" if selected_data.get('adf_passed', False) else "❌"
            stab = f"{selected_data.get('stability_passed', 0)}/{selected_data.get('stability_total', 4)}"
            stab_e = "✅" if selected_data.get('is_stable', False) else "⚠️"
            kf_s = "🔷 Kalman" if selected_data.get('use_kalman', False) else "○ OLS"
            hr_unc = selected_data.get('hr_std', 0)
            st.info(f"""
            **Проверки:**
            {fdr_s} FDR (p-adj={selected_data.get('pvalue_adj', 0):.4f})
            {adf_s} ADF (p={selected_data.get('adf_pvalue', 1.0):.4f})
            {stab_e} Стабильность: {stab} окон
            **HR метод:** {kf_s} (±{hr_unc:.4f})
            """)
        
        with checks_col2:
            if theta > 2.0:
                t_msg = "🟢 Очень быстрый (~{:.1f}ч)".format(-np.log(0.5)/theta * 24)
            elif theta > 1.0:
                t_msg = "🟢 Быстрый (~{:.1f}ч)".format(-np.log(0.5)/theta * 24)
            elif theta > 0.5:
                t_msg = "🟡 Средний (~{:.1f}ч)".format(-np.log(0.5)/theta * 24)
            else:
                t_msg = "🔴 Медленный"
            st.info(f"""
            **OU Process:** {t_msg}
            
            **Adaptive порог:** |Z| ≥ {threshold}
            ({conf} confidence → {'сниженный' if threshold < 2.0 else 'стандартный'} порог)
            """)
        
        # v10: дополнительные метрики
        v10_col1, v10_col2, v10_col3 = st.columns(3)
        with v10_col1:
            zw = selected_data.get('z_window', 30)
            st.metric("Z-окно", f"{zw} баров", "адаптивное (HL×2.5)")
        with v10_col2:
            cd = selected_data.get('crossing_density', 0)
            cd_emoji = "🟢" if cd >= 0.05 else "🟡" if cd >= 0.03 else "🔴"
            st.metric("Crossing Density", f"{cd:.3f} {cd_emoji}",
                       "активный" if cd >= 0.03 else "застрял")
        with v10_col3:
            corr = selected_data.get('correlation', 0)
            corr_emoji = "🟢" if corr >= 0.7 else "🟡" if corr >= 0.4 else "⚪"
            st.metric("Корреляция (ρ)", f"{corr:.3f} {corr_emoji}")
    
    # График спреда
    if selected_data['spread'] is not None:
        fig = plot_spread_chart(selected_data['spread'], selected_pair, selected_data['zscore'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Калькулятор размера позиции
    st.markdown("---")
    st.subheader("💰 Калькулятор размера позиции")
    
    col1, col2 = st.columns(2)
    
    with col1:
        total_capital = st.number_input(
            "💵 Общая сумма для входа (USD)",
            min_value=10.0,
            max_value=1000000.0,
            value=100.0,  # $100 по умолчанию
            step=10.0,
            help="Сколько всего хотите вложить в эту пару",
            key=f"capital_{selected_pair}"
        )
        
        commission_rate = st.number_input(
            "💸 Комиссия биржи (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            help="Обычно 0.1% для мейкеров, 0.075% на Binance с BNB",
            key=f"commission_{selected_pair}"
        )
    
    with col2:
        hedge_ratio = selected_data['hedge_ratio']
        
        st.markdown("### 📊 Распределение капитала:")
        
        # Расчет позиций с учетом hedge ratio
        position1 = total_capital / (1 + hedge_ratio)
        position2 = position1 * hedge_ratio
        
        # Учет комиссий (вход + выход, обе стороны)
        commission_total = (position1 + position2) * (commission_rate / 100) * 2
        effective_capital = total_capital - commission_total
        
        coin1, coin2 = selected_data['coin1'], selected_data['coin2']
        signal = selected_data['signal']
        direction = selected_data.get('direction', 'NONE')
        
        if direction == 'LONG' or (direction == 'NONE' and signal == 'LONG'):
            st.success(f"""
            **🟢 LONG позиция:**
            
            **{coin1}:** 🟢 LONG (КУПИТЬ) — ${position1:.2f}
            **{coin2}:** 🔴 SHORT (ПРОДАТЬ) — ${position2:.2f}
            
            💸 Комиссии: ${commission_total:.2f}
            💰 Эффективно: ${effective_capital:.2f}
            """)
        elif direction == 'SHORT' or (direction == 'NONE' and signal == 'SHORT'):
            st.error(f"""
            **🔴 SHORT позиция:**
            
            **{coin1}:** 🔴 SHORT (ПРОДАТЬ) — ${position1:.2f}
            **{coin2}:** 🟢 LONG (КУПИТЬ) — ${position2:.2f}
            
            💸 Комиссии: ${commission_total:.2f}
            💰 Эффективно: ${effective_capital:.2f}
            """)
        else:
            st.info(f"""
            **⚪ Нет сигнала:**
            
            **{coin1}:** ${position1:.2f}
            **{coin2}:** ${position2:.2f}
            
            ⏳ Ждите сигнал (|Z| > порога)
            """)
    
    # Детальная разбивка
    st.markdown("### 📝 Детальная разбивка позиции")
    
    breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)
    
    # v7.1: Определяем направление для каждой монеты
    dir_label = selected_data.get('direction', 'NONE')
    if dir_label == 'LONG':
        coin1_dir, coin2_dir = "🟢 LONG", "🔴 SHORT"
    elif dir_label == 'SHORT':
        coin1_dir, coin2_dir = "🔴 SHORT", "🟢 LONG"
    else:
        coin1_dir, coin2_dir = "⚪", "⚪"
    
    with breakdown_col1:
        st.metric(f"{coin1} {coin1_dir}", f"${position1:.2f}", 
                 f"{(position1/total_capital)*100:.1f}% от капитала")
    
    with breakdown_col2:
        st.metric(f"{coin2} {coin2_dir}", f"${position2:.2f}",
                 f"{(position2/total_capital)*100:.1f}% от капитала")
    
    with breakdown_col3:
        st.metric("Hedge Ratio", f"{hedge_ratio:.4f}",
                 f"1:{hedge_ratio:.4f}")
    
    # Калькулятор прибыли/убытков
    st.markdown("---")
    st.subheader("🎯 Расчет прибыли и стоп-лосса")
    
    entry_z = selected_data['zscore']
    
    # Стоп-лосс и цели
    if abs(entry_z) > 0:
        if entry_z < 0:  # LONG
            stop_z = entry_z - 1.0
            tp1_z = entry_z + (abs(entry_z) * 0.4)
            target_z = 0.0
        else:  # SHORT
            stop_z = entry_z + 1.0
            tp1_z = entry_z - (abs(entry_z) * 0.4)
            target_z = 0.0
        
        # Процент изменения Z-score
        stop_loss_pct = ((abs(stop_z - entry_z) / abs(entry_z)) * 100)
        tp1_pct = ((abs(tp1_z - entry_z) / abs(entry_z)) * 100)
        target_pct = 100.0
        
        # Реалистичная прибыль для парного арбитража (~6% при полном цикле)
        # Формула: (движение_Z / 100) × капитал × 0.06
        hedge_efficiency = 0.06  # 6% типичная прибыль при полном движении к Z=0
        
        stop_loss_usd = -total_capital * (stop_loss_pct / 100) * hedge_efficiency
        tp1_usd = total_capital * (tp1_pct / 100) * hedge_efficiency
        target_usd = total_capital * (target_pct / 100) * hedge_efficiency
        
        pnl_col1, pnl_col2, pnl_col3 = st.columns(3)
        
        with pnl_col1:
            st.markdown("**🛡️ Стоп-лосс**")
            st.metric("Z-score", f"{stop_z:.2f}")
            st.error(f"Убыток: **${abs(stop_loss_usd):.2f}**")
            st.caption(f"(-{stop_loss_pct:.1f}% от входа)")
        
        with pnl_col2:
            st.markdown("**💰 Take Profit 1**")
            st.metric("Z-score", f"{tp1_z:.2f}")
            st.success(f"Прибыль: **${tp1_usd:.2f}**")
            st.caption(f"(+{tp1_pct:.1f}%, закрыть 50%)")
        
        with pnl_col3:
            st.markdown("**🎯 Полная цель**")
            st.metric("Z-score", "0.00")
            st.success(f"Прибыль: **${target_usd:.2f}**")
            st.caption(f"(+{target_pct:.0f}%, полный выход)")
        
        # Risk/Reward
        risk_reward = abs(target_usd / stop_loss_usd) if stop_loss_usd != 0 else 0
        
        st.markdown("---")
        
        rr_col1, rr_col2, rr_col3 = st.columns(3)
        
        with rr_col1:
            st.metric("💎 Потенциал прибыли", f"${target_usd:.2f}")
        
        with rr_col2:
            st.metric("⚠️ Максимальный риск", f"${abs(stop_loss_usd):.2f}")
        
        with rr_col3:
            if risk_reward >= 2:
                emoji = "🟢"
                assessment = "Отлично!"
            elif risk_reward >= 1.5:
                emoji = "🟡"
                assessment = "Приемлемо"
            else:
                emoji = "🔴"
                assessment = "Слабо"
            
            st.metric(f"{emoji} Risk/Reward", f"{risk_reward:.2f}:1")
            st.caption(assessment)
    
    # Рекомендации по торговле
    st.markdown("---")
    st.markdown("### 💡 Торговая рекомендация")
    
    rec_direction = selected_data.get('direction', 'NONE')
    rec_thr = selected_data.get('threshold', 2.0)
    adaptive_stop = max(rec_thr + 2.0, 4.0)  # v8.0: stop всегда на 2 Z-единицы дальше порога
    
    if rec_direction == 'LONG':
        st.success(f"""
        **Стратегия: 🟢 LONG**
        - 🟢 **КУПИТЬ** {selected_data['coin1']}
        - 🔴 **ПРОДАТЬ** {selected_data['coin2']} (шорт)
        - **Соотношение:** 1:{selected_data['hedge_ratio']:.4f}
        - **Таргет:** Z-score → 0 (mean revert)
        - **Стоп-лосс:** Z < -{adaptive_stop:.1f} (адаптивный: порог {rec_thr} + 2.0)
        """)
    elif rec_direction == 'SHORT':
        st.error(f"""
        **Стратегия: 🔴 SHORT**
        - 🔴 **ПРОДАТЬ** {selected_data['coin1']} (шорт)
        - 🟢 **КУПИТЬ** {selected_data['coin2']}
        - **Соотношение:** 1:{selected_data['hedge_ratio']:.4f}
        - **Таргет:** Z-score → 0 (mean revert)
        - **Стоп-лосс:** Z > +{adaptive_stop:.1f} (адаптивный: порог {rec_thr} + 2.0)
        """)
    else:
        st.info("⚪ Нет активного сигнала. Дождитесь |Z-score| > порога")
    
    # v8.0: Детальный анализ пары — CSV export
    st.markdown("---")
    st.markdown("### 📥 Экспорт детального анализа пары")
    
    detail_data = {
        'Параметр': [
            'Пара', 'Направление', 'Статус', 'Вход',
            'Z-score', 'Threshold', 'Quality Score', 'Signal Score',
            'Confidence', 'P-value (adj)', 'FDR', 'ADF',
            'Hurst (DFA)', 'Hurst fallback?', 'Half-life (ч)', 'Theta',
            'Hedge Ratio', 'HR uncertainty', 'Корреляция ρ',
            'Stability', 'Crossing Density', 'Z-window',
            'Kalman HR', 'N баров',
            f'{selected_data["coin1"]} Action', f'{selected_data["coin2"]} Action',
        ],
        'Значение': [
            selected_data['pair'],
            selected_data.get('direction', 'NONE'),
            selected_data.get('signal', 'NEUTRAL'),
            selected_data.get('_entry_label', '⚪ ЖДАТЬ'),
            round(selected_data['zscore'], 4),
            selected_data.get('threshold', 2.0),
            selected_data.get('quality_score', 0),
            selected_data.get('signal_score', 0),
            selected_data.get('confidence', '?'),
            round(selected_data.get('pvalue_adj', selected_data['pvalue']), 6),
            '✅' if selected_data.get('fdr_passed') else '❌',
            '✅' if selected_data.get('adf_passed') else '❌',
            round(selected_data.get('hurst', 0.5), 4),
            '⚠️ YES' if selected_data.get('hurst_is_fallback') else 'NO',
            round(selected_data.get('halflife_hours', selected_data['halflife_days']*24), 2),
            round(selected_data.get('theta', 0), 4),
            round(selected_data['hedge_ratio'], 6),
            round(selected_data.get('hr_uncertainty', 0), 4),
            round(selected_data.get('correlation', 0), 4),
            f"{selected_data.get('stability_passed', 0)}/{selected_data.get('stability_total', 4)}",
            round(selected_data.get('crossing_density', 0), 4),
            selected_data.get('z_window', 30),
            '✅' if selected_data.get('use_kalman') else '❌ OLS',
            selected_data.get('n_bars', 0),
            'LONG (КУПИТЬ)' if rec_direction == 'LONG' else ('SHORT (ПРОДАТЬ)' if rec_direction == 'SHORT' else '-'),
            'SHORT (ПРОДАТЬ)' if rec_direction == 'LONG' else ('LONG (КУПИТЬ)' if rec_direction == 'SHORT' else '-'),
        ]
    }
    df_detail = pd.DataFrame(detail_data)
    csv_detail = df_detail.to_csv(index=False)
    st.download_button(
        "📥 Скачать детальный анализ пары (CSV)",
        csv_detail,
        f"detail_{selected_pair.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv",
        key="detail_csv_btn"
    )
    
    # Экспорт данных — расширенный CSV (v7.1)
    st.markdown("---")
    
    # v7.1: Расширенный CSV с направлением и всеми метриками для бэктеста
    export_rows = []
    for p in pairs:
        export_rows.append({
            'Пара': p['pair'],
            'Coin1': p['coin1'],
            'Coin2': p['coin2'],
            'Вход': p.get('_entry_label', '⚪ ЖДАТЬ'),
            'Статус': p['signal'],
            'Направление': p.get('direction', 'NONE'),
            'Coin1_Action': ('LONG' if p.get('direction') == 'LONG' else 'SHORT' if p.get('direction') == 'SHORT' else ''),
            'Coin2_Action': ('SHORT' if p.get('direction') == 'LONG' else 'LONG' if p.get('direction') == 'SHORT' else ''),
            'Quality': p.get('quality_score', 0),
            'Signal_Score': p.get('signal_score', 0),
            'Confidence': p.get('confidence', '?'),
            'Z-score': round(p['zscore'], 4),
            'Threshold': p.get('threshold', 2.0),
            'P-value': round(p['pvalue'], 6),
            'P-value_adj': round(p.get('pvalue_adj', p['pvalue']), 6),
            'FDR': p.get('fdr_passed', False),
            'Hurst': round(p.get('hurst', 0.5), 4),
            'Half-life_hours': round(p.get('halflife_hours', p['halflife_days']*24), 2),
            'Hedge_Ratio': round(p['hedge_ratio'], 6),
            'HR_uncertainty': round(p.get('hr_uncertainty', 0), 4),
            'Correlation': round(p.get('correlation', 0), 4),
            'Stability': f"{p.get('stability_passed', 0)}/{p.get('stability_total', 4)}",
            'ADF_passed': p.get('adf_passed', False),
            'Theta': round(p.get('theta', 0), 4),
            'Crossing_Density': round(p.get('crossing_density', 0), 4),
            'Z_window': p.get('z_window', 30),
            'Kalman': p.get('use_kalman', False),
            'N_bars': p.get('n_bars', 0),
            'Opt_criteria': f"{p.get('_opt_count', 0)}/6",
            'FDR_bypass': p.get('_fdr_bypass', False),
            'Cluster': p.get('cluster', ''),
        })
    
    df_export = pd.DataFrame(export_rows)
    csv_data = df_export.to_csv(index=False)
    
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="📥 Скачать результаты (CSV)",
            data=csv_data,
            file_name=f"pairs_scan_{exchange}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    with dl_col2:
        # Краткая таблица (как раньше)
        csv_short = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Краткая таблица (CSV)",
            data=csv_short,
            file_name=f"pairs_short_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Нажмите 'Запустить сканер' для начала анализа")
    
    # Инструкция
    st.markdown("""
    ### 🎯 Что делает этот скринер:
    
    1. **Загружает данные** топ-100 криптовалют с Binance
    2. **Тестирует все пары** на статистическую коинтеграцию
    3. **Находит возможности** для парного арбитража
    4. **Показывает сигналы** на основе Z-score
    
    ### 📚 Как торговать:
    
    - **Z-score > +2**: Пара переоценена → SHORT первая монета, LONG вторая
    - **Z-score < -2**: Пара недооценена → LONG первая монета, SHORT вторая
    - **Z-score → 0**: Закрытие позиции (возврат к среднему)
    
    ### ⚠️ Важно:
    - Используйте стоп-лоссы
    - Учитывайте комиссии биржи
    - Проверяйте ликвидность пар
    - Это не финансовая рекомендация
    """)

# Footer
st.markdown("---")
st.caption("⚠️ Disclaimer: Этот инструмент предназначен только для образовательных целей. Не является финансовой рекомендацией.")
# VERSION: 7.1
# LAST UPDATED: 2026-02-19
# FIXES v7.1:
#   [FIX] Smart exchange fallback: Binance→OKX→KuCoin→Bybit (Binance/Bybit 403 on HuggingFace/cloud)
#   [FIX] st.session_state.running=False after scan — prevents rescan on pair selection
#   [FIX] get_adaptive_signal() try/except TypeError for backward compat
#   [NEW] Direction labels (LONG/SHORT) in position calculator + breakdown
#   [NEW] Extended CSV export with all metrics + direction + coin actions
#   [NEW] Coin limit increased to 150 default, 200 max
