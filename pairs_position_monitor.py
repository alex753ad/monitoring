"""
Pairs Position Monitor v1.0
Мониторинг открытых позиций по парному трейдингу.

Показывает:
  - Текущий Z-score каждой позиции в реальном времени
  - MTM P&L (mark-to-market)
  - Сигналы к выходу (mean reversion / stop / timeout)
  - Визуализация спреда и Z-score

Запуск: streamlit run pairs_position_monitor.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import time
import json
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════
# CORE MATH (standalone — не зависит от analysis module)
# ═══════════════════════════════════════════════════════

def kalman_hr(s1, s2, delta=1e-4, ve=1e-3):
    s1, s2 = np.array(s1, float), np.array(s2, float)
    n = min(len(s1), len(s2))
    if n < 10: return None
    s1, s2 = s1[:n], s2[:n]
    init_n = min(30, n // 3)
    try:
        X = np.column_stack([np.ones(init_n), s2[:init_n]])
        beta = np.linalg.lstsq(X, s1[:init_n], rcond=None)[0]
    except: beta = np.array([0.0, 1.0])
    P = np.eye(2); Q = np.eye(2) * delta; R = ve
    hrs, ints, spread = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(n):
        x = np.array([1.0, s2[t]]); P += Q
        e = s1[t] - x @ beta; S = x @ P @ x + R
        K = P @ x / S; beta += K * e
        P -= np.outer(K, x) @ P; P = (P + P.T) / 2
        np.fill_diagonal(P, np.maximum(np.diag(P), 1e-10))
        hrs[t], ints[t] = beta[1], beta[0]
        spread[t] = s1[t] - beta[1] * s2[t] - beta[0]
    return {'hrs': hrs, 'intercepts': ints, 'spread': spread,
            'hr': float(hrs[-1]), 'intercept': float(ints[-1])}


def calc_zscore(spread, halflife_bars=None, min_w=10, max_w=60):
    spread = np.array(spread, float); n = len(spread)
    if halflife_bars and not np.isinf(halflife_bars) and halflife_bars > 0:
        w = int(np.clip(2.5 * halflife_bars, min_w, max_w))
    else: w = 30
    w = min(w, max(10, n // 2))
    zs = np.full(n, np.nan)
    for i in range(w, n):
        lb = spread[i - w:i]; med = np.median(lb)
        mad = np.median(np.abs(lb - med)) * 1.4826
        if mad < 1e-10:
            s = np.std(lb)
            zs[i] = (spread[i] - np.mean(lb)) / s if s > 1e-10 else 0
        else: zs[i] = (spread[i] - med) / mad
    return zs, w


def calc_halflife(spread):
    s = np.array(spread, float)
    if len(s) < 10: return 999
    sl, sd = s[:-1], np.diff(s)
    n = len(sl); denom = n * np.sum(sl**2) - np.sum(sl)**2
    if abs(denom) < 1e-10: return 999
    b = (n * np.sum(sl * sd) - np.sum(sl) * np.sum(sd)) / denom
    return float(-np.log(2) / b) if b < 0 else 999


# ═══════════════════════════════════════════════════════
# POSITIONS FILE (JSON persistence)
# ═══════════════════════════════════════════════════════
POSITIONS_FILE = "positions.json"

def load_positions():
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    return []

def save_positions(positions):
    with open(POSITIONS_FILE, 'w') as f:
        json.dump(positions, f, indent=2, default=str)


def add_position(coin1, coin2, direction, entry_z, entry_hr, 
                 entry_price1, entry_price2, timeframe, notes=""):
    positions = load_positions()
    pos = {
        'id': len(positions) + 1,
        'coin1': coin1, 'coin2': coin2,
        'direction': direction,
        'entry_z': entry_z,
        'entry_hr': entry_hr,
        'entry_price1': entry_price1,
        'entry_price2': entry_price2,
        'entry_time': datetime.now().isoformat(),
        'timeframe': timeframe,
        'status': 'OPEN',
        'notes': notes,
        'exit_z_target': 0.5,
        'stop_z': 4.5,
    }
    positions.append(pos)
    save_positions(positions)
    return pos


def close_position(pos_id, exit_price1, exit_price2, exit_z, reason):
    positions = load_positions()
    for p in positions:
        if p['id'] == pos_id and p['status'] == 'OPEN':
            p['status'] = 'CLOSED'
            p['exit_price1'] = exit_price1
            p['exit_price2'] = exit_price2
            p['exit_z'] = exit_z
            p['exit_time'] = datetime.now().isoformat()
            p['exit_reason'] = reason
            # P&L
            r1 = (exit_price1 - p['entry_price1']) / p['entry_price1']
            r2 = (exit_price2 - p['entry_price2']) / p['entry_price2']
            hr = p['entry_hr']
            if p['direction'] == 'LONG':
                raw = r1 - hr * r2
            else:
                raw = -r1 + hr * r2
            p['pnl_pct'] = round(raw / (1 + abs(hr)) * 100, 3)
            break
    save_positions(positions)


# ═══════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════

@st.cache_data(ttl=120)
def fetch_prices(exchange_name, coin, timeframe, lookback_bars=300):
    try:
        ex = getattr(ccxt, exchange_name)({'enableRateLimit': True})
        ex.load_markets()
        symbol = f"{coin}/USDT"
        ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=lookback_bars)
        df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except:
        return None


def get_current_price(exchange_name, coin):
    try:
        ex = getattr(ccxt, exchange_name)({'enableRateLimit': True})
        ticker = ex.fetch_ticker(f"{coin}/USDT")
        return ticker['last']
    except:
        return None


# ═══════════════════════════════════════════════════════
# MONITOR LOGIC
# ═══════════════════════════════════════════════════════

def monitor_position(pos, exchange_name):
    """Полный мониторинг одной позиции."""
    c1, c2 = pos['coin1'], pos['coin2']
    tf = pos['timeframe']
    
    bars_map = {'1h': 300, '4h': 300, '1d': 120}
    n_bars = bars_map.get(tf, 300)
    
    df1 = fetch_prices(exchange_name, c1, tf, n_bars)
    df2 = fetch_prices(exchange_name, c2, tf, n_bars)
    
    if df1 is None or df2 is None:
        return None
    
    # Align timestamps
    merged = pd.merge(df1[['ts', 'c']], df2[['ts', 'c']], on='ts', suffixes=('_1', '_2'))
    if len(merged) < 50:
        return None
    
    p1 = merged['c_1'].values
    p2 = merged['c_2'].values
    ts = merged['ts'].tolist()
    
    # Kalman
    kf = kalman_hr(p1, p2)
    if kf is None:
        return None
    
    spread = kf['spread']
    hr_current = kf['hr']
    
    # Half-life → Z
    hl = calc_halflife(spread)
    hpb = {'1h': 1, '4h': 4, '1d': 24}.get(tf, 4)
    hl_bars = (hl * 24 / hpb) if hl < 999 else None
    zs, zw = calc_zscore(spread, halflife_bars=hl_bars)
    
    z_now = float(zs[~np.isnan(zs)][-1]) if any(~np.isnan(zs)) else 0
    
    # P&L
    r1 = (p1[-1] - pos['entry_price1']) / pos['entry_price1']
    r2 = (p2[-1] - pos['entry_price2']) / pos['entry_price2']
    hr = pos['entry_hr']
    if pos['direction'] == 'LONG':
        raw_pnl = r1 - hr * r2
    else:
        raw_pnl = -r1 + hr * r2
    pnl_pct = raw_pnl / (1 + abs(hr)) * 100
    
    # Exit signals
    exit_signal = None
    ez = pos.get('exit_z_target', 0.5)
    sz = pos.get('stop_z', 4.5)
    
    if pos['direction'] == 'LONG':
        if z_now >= -ez and z_now <= ez:
            exit_signal = '✅ MEAN REVERT — закрывать!'
        elif z_now > 1.0:
            exit_signal = '✅ OVERSHOOT — фиксировать прибыль!'
        elif z_now < -sz:
            exit_signal = '🛑 STOP LOSS — экстренный выход!'
    else:
        if z_now <= ez and z_now >= -ez:
            exit_signal = '✅ MEAN REVERT — закрывать!'
        elif z_now < -1.0:
            exit_signal = '✅ OVERSHOOT — фиксировать прибыль!'
        elif z_now > sz:
            exit_signal = '🛑 STOP LOSS — экстренный выход!'
    
    # Time in trade
    entry_dt = datetime.fromisoformat(pos['entry_time'])
    hours_in = (datetime.now() - entry_dt).total_seconds() / 3600
    
    return {
        'z_now': z_now,
        'z_entry': pos['entry_z'],
        'pnl_pct': pnl_pct,
        'price1_now': p1[-1],
        'price2_now': p2[-1],
        'hr_now': hr_current,
        'hr_entry': pos['entry_hr'],
        'exit_signal': exit_signal,
        'hours_in': hours_in,
        'spread': spread,
        'zscore_series': zs,
        'timestamps': ts,
        'hr_series': kf['hrs'],
        'halflife_hours': hl * 24,
        'z_window': zw,
    }


# ═══════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════

st.set_page_config(page_title="Position Monitor", page_icon="📍", layout="wide")

st.markdown("""
<style>
    .exit-signal { padding: 15px; border-radius: 10px; font-size: 1.2em; 
                   font-weight: bold; text-align: center; margin: 10px 0; }
    .signal-exit { background: #1b5e20; color: #a5d6a7; }
    .signal-stop { background: #b71c1c; color: #ef9a9a; }
</style>
""", unsafe_allow_html=True)

st.title("📍 Pairs Position Monitor")
st.caption("v1.0 | Мониторинг открытых позиций парного трейдинга")

# Sidebar
with st.sidebar:
    st.header("⚙️ Настройки")
    exchange = st.selectbox("Биржа", ['okx', 'bybit', 'binance'], index=0)
    auto_refresh = st.checkbox("Авто-обновление (2 мин)", value=False)
    
    st.divider()
    st.header("➕ Новая позиция")
    
    with st.form("add_position"):
        col1, col2 = st.columns(2)
        with col1:
            new_c1 = st.text_input("Coin 1", "ETH").upper().strip()
        with col2:
            new_c2 = st.text_input("Coin 2", "STETH").upper().strip()
        
        new_dir = st.selectbox("Направление", ["LONG", "SHORT"])
        new_tf = st.selectbox("Таймфрейм", ['1h', '4h', '1d'], index=1)
        
        col3, col4 = st.columns(2)
        with col3:
            new_z = st.number_input("Entry Z", value=2.0, step=0.1)
        with col4:
            new_hr = st.number_input("Hedge Ratio", value=1.0, step=0.01, format="%.4f")
        
        col5, col6 = st.columns(2)
        with col5:
            new_p1 = st.number_input("Цена Coin1", value=0.0, step=0.01, format="%.4f")
        with col6:
            new_p2 = st.number_input("Цена Coin2", value=0.0, step=0.01, format="%.4f")
        
        new_notes = st.text_input("Заметки", "")
        
        # Автозагрузка цен
        fetch_prices_btn = st.form_submit_button("📥 Загрузить цены + Добавить")
    
    if fetch_prices_btn and new_c1 and new_c2:
        if new_p1 == 0 or new_p2 == 0:
            with st.spinner("Загружаю текущие цены..."):
                p1_live = get_current_price(exchange, new_c1)
                p2_live = get_current_price(exchange, new_c2)
                if p1_live and p2_live:
                    new_p1 = p1_live
                    new_p2 = p2_live
                    st.info(f"💰 {new_c1}: ${p1_live:.4f} | {new_c2}: ${p2_live:.4f}")
                else:
                    st.error("Не удалось загрузить цены")
        
        if new_p1 > 0 and new_p2 > 0:
            pos = add_position(new_c1, new_c2, new_dir, new_z, new_hr,
                             new_p1, new_p2, new_tf, new_notes)
            st.success(f"✅ Позиция #{pos['id']} добавлена: {new_dir} {new_c1}/{new_c2}")
            st.rerun()

# ═══════ MAIN AREA ═══════
positions = load_positions()
open_positions = [p for p in positions if p['status'] == 'OPEN']
closed_positions = [p for p in positions if p['status'] == 'CLOSED']

# Tabs
tab1, tab2 = st.tabs([f"📍 Открытые ({len(open_positions)})", 
                       f"📋 История ({len(closed_positions)})"])

with tab1:
    if not open_positions:
        st.info("📭 Нет открытых позиций. Добавьте через боковую панель.")
    else:
        # Dashboard metrics
        total_pnl = 0
        
        for pos in open_positions:
            with st.container():
                st.markdown("---")
                
                # Header
                dir_emoji = '🟢' if pos['direction'] == 'LONG' else '🔴'
                pair_name = f"{pos['coin1']}/{pos['coin2']}"
                
                # Monitor
                with st.spinner(f"Обновляю {pair_name}..."):
                    mon = monitor_position(pos, exchange)
                
                if mon is None:
                    st.error(f"❌ Не удалось получить данные для {pair_name}")
                    continue
                
                total_pnl += mon['pnl_pct']
                
                # Exit signal banner
                if mon['exit_signal']:
                    if 'STOP' in mon['exit_signal']:
                        st.error(mon['exit_signal'])
                    else:
                        st.success(mon['exit_signal'])
                
                # Header row
                st.subheader(f"{dir_emoji} {pos['direction']} | {pair_name} | #{pos['id']}")
                
                # KPI row
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                
                pnl_color = "normal" if mon['pnl_pct'] > 0 else "inverse"
                c1.metric("P&L", f"{mon['pnl_pct']:+.2f}%", 
                         delta="profit" if mon['pnl_pct'] > 0 else "loss")
                c2.metric("Z сейчас", f"{mon['z_now']:+.2f}",
                         delta=f"вход: {mon['z_entry']:+.2f}")
                c3.metric("HR", f"{mon['hr_now']:.4f}",
                         delta=f"вход: {mon['hr_entry']:.4f}")
                c4.metric(f"{pos['coin1']}", f"${mon['price1_now']:.4f}",
                         delta=f"вход: ${pos['entry_price1']:.4f}")
                c5.metric(f"{pos['coin2']}", f"${mon['price2_now']:.4f}",
                         delta=f"вход: ${pos['entry_price2']:.4f}")
                c6.metric("В позиции", f"{mon['hours_in']:.0f}ч",
                         delta=f"HL: {mon['halflife_hours']:.0f}ч")
                
                # Chart
                with st.expander("📈 Графики", expanded=False):
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       vertical_spacing=0.08,
                                       subplot_titles=['Z-Score', 'Спред'],
                                       row_heights=[0.6, 0.4])
                    
                    ts = mon['timestamps']
                    
                    # Z-score
                    fig.add_trace(go.Scatter(
                        x=ts, y=mon['zscore_series'],
                        name='Z-Score', line=dict(color='#4fc3f7', width=2)
                    ), row=1, col=1)
                    
                    fig.add_hline(y=0, line_dash='dash', line_color='gray', 
                                 opacity=0.5, row=1, col=1)
                    fig.add_hline(y=pos.get('exit_z_target', 0.5), 
                                 line_dash='dot', line_color='#4caf50',
                                 opacity=0.5, row=1, col=1)
                    fig.add_hline(y=-pos.get('exit_z_target', 0.5), 
                                 line_dash='dot', line_color='#4caf50',
                                 opacity=0.5, row=1, col=1)
                    
                    # Entry Z marker
                    entry_dt = datetime.fromisoformat(pos['entry_time'])
                    fig.add_trace(go.Scatter(
                        x=[entry_dt], y=[pos['entry_z']],
                        mode='markers', marker=dict(size=14, color='yellow',
                                                     symbol='star'),
                        name='Entry', showlegend=True
                    ), row=1, col=1)
                    
                    # Spread
                    fig.add_trace(go.Scatter(
                        x=ts, y=mon['spread'],
                        name='Spread', line=dict(color='#ffa726', width=1.5)
                    ), row=2, col=1)
                    
                    fig.update_layout(height=400, template='plotly_dark',
                                     showlegend=False,
                                     margin=dict(l=50, r=30, t=30, b=30))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Close button
                col_close1, col_close2, col_close3 = st.columns([2, 2, 1])
                with col_close3:
                    if st.button(f"❌ Закрыть #{pos['id']}", key=f"close_{pos['id']}"):
                        close_position(
                            pos['id'], 
                            mon['price1_now'], mon['price2_now'],
                            mon['z_now'], 'MANUAL'
                        )
                        st.success(f"Позиция #{pos['id']} закрыта | P&L: {mon['pnl_pct']:+.2f}%")
                        st.rerun()
        
        # Total P&L
        st.markdown("---")
        st.metric("📊 Суммарный P&L (открытые)", f"{total_pnl:+.2f}%")

with tab2:
    if not closed_positions:
        st.info("📭 Нет закрытых позиций")
    else:
        # Summary
        pnls = [p.get('pnl_pct', 0) for p in closed_positions]
        wins = [p for p in pnls if p > 0]
        
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Сделок", len(closed_positions))
        sc2.metric("Win Rate", f"{len(wins)/len(closed_positions)*100:.0f}%" if closed_positions else "0%")
        sc3.metric("Total P&L", f"{sum(pnls):+.2f}%")
        sc4.metric("Avg P&L", f"{np.mean(pnls):+.2f}%" if pnls else "0%")
        
        # Table
        rows = []
        for p in reversed(closed_positions):
            rows.append({
                '#': p['id'],
                'Пара': f"{p['coin1']}/{p['coin2']}",
                'Dir': p['direction'],
                'TF': p['timeframe'],
                'Entry Z': f"{p['entry_z']:+.2f}",
                'Exit Z': f"{p.get('exit_z', 0):+.2f}",
                'P&L %': f"{p.get('pnl_pct', 0):+.2f}",
                'Причина': p.get('exit_reason', ''),
                'Вход': p['entry_time'][:16],
                'Выход': p.get('exit_time', '')[:16] if p.get('exit_time') else '',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# Auto refresh
if auto_refresh:
    time.sleep(120)
    st.rerun()

st.divider()
st.caption("""
**Pairs Position Monitor v1.0**

Как добавить позицию:
1. Найди SIGNAL в скринере (pairs_monitor.py)
2. Скопируй данные: Coin1, Coin2, Direction, Z, HR, цены
3. Введи в форму слева → "Загрузить цены + Добавить"
4. Монитор покажет когда закрывать

Позиции сохраняются в positions.json — не потеряются при перезапуске.
""")
