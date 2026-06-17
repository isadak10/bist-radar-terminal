import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import os

# ==========================================
# 1. SAYFA YAPILANDIRMASI & PREMİUM TEMALANDIRMA
# ==========================================
st.set_page_config(
    page_title="BIST Kısa Vade Tarama & Strateji Üssü",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Bloomberg & TradingView Koyu Tema (#0b0e11) CSS Entegrasyonu
st.markdown("""
<style>
    /* Koyu Tema Arka Planı ve Yazı Tipi */
    .stApp {
        background-color: #0b0e11;
        color: #d1d4dc;
    }
    
    h1, h2, h3, p, label, .stMarkdown {
        color: #d1d4dc !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }
    
    /* Sol Menü (Sidebar) Özelleştirmeleri */
    section[data-testid="stSidebar"] {
        background-color: #151924 !important;
        border-right: 1px solid #2a2e39;
    }
    
    /* Premium Kart Tasarımı (stCard) */
    .st-card {
        background-color: #131722;
        border: 1px solid #2a2e39;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
    }
    
    /* Trade Kurulum Kartı (Neon Yeşil & Kırmızı) */
    .setup-card {
        background: linear-gradient(135deg, #11261d 0%, #151924 100%);
        border: 2px solid #00e676;
        border-radius: 8px;
        padding: 20px;
        margin-top: 15px;
        box-shadow: 0 0 15px rgba(0, 230, 118, 0.2);
    }
    
    .setup-card-title {
        color: #00e676;
        font-size: 1.1em;
        font-weight: bold;
        text-transform: uppercase;
        margin-bottom: 12px;
        letter-spacing: 0.05em;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Rozet Tasarımları */
    .badge {
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85em;
        display: inline-block;
    }
    .badge-green {
        background-color: rgba(0, 230, 118, 0.15);
        color: #00e676;
        border: 1px solid #00e676;
    }
    .badge-red {
        background-color: rgba(255, 23, 68, 0.15);
        color: #ff1744;
        border: 1px solid #ff1744;
    }
    .badge-blue {
        background-color: rgba(41, 121, 255, 0.15);
        color: #2979ff;
        border: 1px solid #2979ff;
    }
    .badge-orange {
        background-color: rgba(255, 145, 0, 0.15);
        color: #ff9100;
        border: 1px solid #ff9100;
    }
    
    /* Tab Tasarımları */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0b0e11;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #131722;
        border-radius: 4px 4px 0px 0px;
        color: #8f96a3;
        border: 1px solid #2a2e39;
        border-bottom: none;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2a2e39 !important;
        color: #ffffff !important;
        font-weight: bold;
        border-top: 2px solid #2979ff !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. VERİ TEMİZLEME VE İNDİKATÖR MATEMATİĞİ
# ==========================================

def clean_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance'den gelen DataFrame'deki MultiIndex sütun yapılarını düzleştirir,
    isimleri küçük harfe çevirir ve eksik değerleri temizler.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.columns = [col.lower() for col in df.columns]
    
    # Mükerrer kolonları temizle
    df = df.loc[:, ~df.columns.duplicated()]
    
    required = ['open', 'high', 'low', 'close', 'volume']
    df = df[[col for col in required if col in df.columns]]
    df = df.dropna()
    return df

def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    1 saatlik veriyi 4 saatlik periyotlara dönüştürerek 4h mumlarını simüle eder.
    """
    df = clean_yf_df(df_1h)
    resampled = df.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return resampled

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Saf Pandas ve Numpy ile tüm teknik indikatörleri hesaplar.
    Python 3.14.0 üzerinde 100% kararlı çalışır.
    """
    df = df.copy()
    
    # 1. Bollinger Bantları (20, 2)
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (std * 2)
    df['bb_lower'] = df['bb_middle'] - (std * 2)
    
    # Bollinger Bant Sıkışması (Squeeze)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    bb_width_min = df['bb_width'].rolling(20).min()
    bb_width_max = df['bb_width'].rolling(20).max()
    bb_width_pct = (df['bb_width'] - bb_width_min) / (bb_width_max - bb_width_min + 1e-8)
    df['bb_squeeze'] = bb_width_pct < 0.25
    
    # 2. EMAs
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # 3. RSI (14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)
    
    # 4. MACD (12, 26, 9)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 5. VWMA (20 Periyotluk Hacim Ağırlıklı Hareketli Ortalama)
    df['vwma'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['vwma'] = df['vwma'].fillna(df['bb_middle']) # Hacim sıfır ise SMA kullan
    
    # 6. 10 Günlük Momentum
    df['momentum'] = (df['close'] / df['close'].shift(10)) * 100
    
    # 7. Hacim Katı (Hacim / 20 Günlük Hacim SMA)
    df['vol_sma'] = df['volume'].rolling(window=20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma'].replace(0, np.nan)
    df['vol_ratio'] = df['vol_ratio'].fillna(1.0)
    
    # 10 Günlük Hacim SMA ve Hacim Katı (10g Ort. Kıyasla)
    df['vol_sma_10'] = df['volume'].rolling(window=10).mean()
    df['vol_ratio_10'] = df['volume'] / df['vol_sma_10'].replace(0, np.nan)
    df['vol_ratio_10'] = df['vol_ratio_10'].fillna(1.0)
    
    # 8. SMI (Stochastic Momentum Index) (10, 3, 3)
    hh = df['high'].rolling(10).max()
    ll = df['low'].rolling(10).min()
    center = (hh + ll) / 2
    d = df['close'] - center
    r = hh - ll
    
    d_s1 = d.ewm(span=3, adjust=False).mean()
    d_s2 = d_s1.ewm(span=3, adjust=False).mean()
    
    r_s1 = r.ewm(span=3, adjust=False).mean()
    r_s2 = r_s1.ewm(span=3, adjust=False).mean()
    
    df['smi'] = 200 * (d_s2 / r_s2.replace(0, np.nan))
    df['smi'] = df['smi'].fillna(0.0)
    df['smi_ema'] = df['smi'].ewm(span=3, adjust=False).mean()
    
    # SMI_VWAP (7 periyotluk SMI hacim ağırlıklı hareketli ortalama)
    weighted_smi = df['smi'] * df['volume']
    df['smi_vwap'] = weighted_smi.rolling(window=7).sum() / df['volume'].rolling(window=7).sum()
    df['smi_vwap'] = df['smi_vwap'].fillna(df['smi_ema'])
    
    # 9. ATR (Average True Range) 14
    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    df['atr_14'] = df['atr_14'].fillna(df['close'] * 0.02)
    
    # 10. SMA 200 (Larry Connors trend filter)
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # 11. EMA 5 (Larry Connors exit target)
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    
    # 12. RSI(2) (Larry Connors trigger)
    delta_2 = df['close'].diff()
    gain_2 = delta_2.where(delta_2 > 0, 0.0)
    loss_2 = -delta_2.where(delta_2 < 0, 0.0)
    avg_gain_2 = gain_2.ewm(alpha=1/2, adjust=False).mean()
    avg_loss_2 = loss_2.ewm(alpha=1/2, adjust=False).mean()
    rs_2 = avg_gain_2 / avg_loss_2.replace(0, np.nan)
    df['rsi_2'] = (100 - (100 / (1 + rs_2))).fillna(50)
    
    # 13. ATR 20 & Keltner Channels (John Carter Squeeze)
    tr_20 = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df['atr_20'] = tr_20.rolling(window=20).mean()
    df['atr_20'] = df['atr_20'].fillna(df['close'] * 0.02)
    df['kc_upper'] = df['ema_20'] + 1.5 * df['atr_20']
    df['kc_lower'] = df['ema_20'] - 1.5 * df['atr_20']
    df['squeeze_on'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
    
    return df


# ==========================================
# 3. YEREL VERİ ÇEKİM VERİTABANI VE PARSER
# ==========================================

# Fundamental data function removed (exclusively technical terminal)

@st.cache_data(ttl=300)
def fetch_single_stock(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Belirtilen hissenin zaman dilimine göre verisini indirir ve temizler.
    """
    symbol = symbol.upper().strip()
    if not symbol.endswith(".IS") and "." not in symbol:
        yf_symbol = f"{symbol}.IS"
    else:
        yf_symbol = symbol
        
    # Zaman dilimine göre veri aralığı ve periyot ayarı
    if timeframe == "1h":
        df = yf.download(yf_symbol, period="730d", interval="1h", progress=False)
        df = clean_yf_df(df)
    elif timeframe == "4h":
        # 4 saatlik veri için saatlik çekilip resample edilir
        df_1h = yf.download(yf_symbol, period="730d", interval="1h", progress=False)
        df = resample_to_4h(df_1h)
    elif timeframe == "1d":
        df = yf.download(yf_symbol, period="2y", interval="1d", progress=False)
        df = clean_yf_df(df)
    elif timeframe == "1wk":
        df = yf.download(yf_symbol, period="5y", interval="1wk", progress=False)
        df = clean_yf_df(df)
    else:
        raise ValueError("Geçersiz zaman dilimi!")
        
    if df.empty:
        raise ValueError("Veri bulunamadı. Lütfen sembolün geçerliliğini kontrol edin.")
        
    return calculate_technical_indicators(df)


# Büyük BIST listesi (Sekme 2 ve Sekme 3 için kullanılacak)
BIST_SCAN_LIST = [
    "BINHO", "AVOD", "A1CAP", "ACSEL", "ADEL", "ADESE", "AFYON", "AGHOL", "AGESA", "AGROT", "AHGAZ", 
    "AKBNK", "AKCNS", "AKENR", "AKFGY", "AKFIS", "AKFYE", "AKSA", "AKSEN", "AKGRT", "ALARK", "ALBRK", 
    "ALCTL", "ALFAS", "ALKIM", "ALTNY", "ANSGR", "AEFES", "ANHYT", "ASUZU", "ARDYZ", "ARCLK", "ASELS", 
    "ASTOR", "ATATP", "AYDEM", "AYGAZ", "BAGFS", "BANVT", "BERA", "BIMAS", "BIOEN", "BRLSM", "BOBET", 
    "BRSAN", "BRISA", "BUCIM", "CCOLA", "CVKMD", "CWENE", "CANTE", "CATES", "CIMSA", "DEVA", "DOAS", 
    "DOHOL", "EBEBK", "ECZYT", "EGEEN", "ECILC", "EKOS", "EKGYO", "ENJSA", "ENERY", "ENKAI", "ENSRI", 
    "EREGL", "EUPWR", "FROTO", "FORTE", "GESAN", "GLYHO", "GUBRF", "SAHOL", "HATSN", "HEKTS", "INVEO", 
    "ISCTR", "ISMEN", "IZENR", "KARDMD", "KAREL", "KARSN", "KAYSE", "KRVGD", "KCAER", "KCHOL", "KONTR", 
    "KONYA", "KOZAL", "KOZAA", "MACKO", "MAVI", "MIATK", "MGROS", "MOGAN", "NATEN", "NETAS", "ODAS", 
    "ODINE", "ORGE", "OYAKC", "PETKM", "PGSUS", "QUAGR", "REEDR", "RYGYO", "RYSAS", "SASA", "SDTTR", 
    "SNGYO", "SMRTG", "SOKM", "TABGD", "TAVHL", "TKFEN", "TKNSA", "TOASO", "TCELL", "TMSN", "TUPRS", 
    "THYAO", "GARAN", "HALKB", "TSKB", "TURSG", "SISE", "VAKBN", "TTKOM", "TTRAK", "ULKER", "YKBNK", "YEOTK"
]

@st.cache_data(ttl=900)  # Toplu tarama verilerini 15 dakika önbelleğe al
def batch_scan_bist_data() -> dict:
    """
    yfinance toplu indirme özelliğini kullanarak tüm BIST listesini tek seferde indirir,
    indikatörlerini hesaplar ve bir sözlük yapısında döndürür.
    """
    tickers_formatted = [f"{s}.IS" for s in BIST_SCAN_LIST]
    
    # Toplu indirme işlemi (Maksimum 1 yıllık veri, SMA 200 hesaplamak için)
    batch_df = yf.download(tickers_formatted, period="1y", interval="1d", group_by="ticker", progress=False)
    
    scanned_data = {}
    for sym in BIST_SCAN_LIST:
        yf_sym = f"{sym}.IS"
        try:
            if yf_sym not in batch_df.columns.levels[0]:
                continue
                
            df_stock = batch_df[yf_sym].copy()
            df_stock.columns = [col.lower() for col in df_stock.columns]
            df_stock = df_stock.dropna(subset=['close'])
            
            # SMA 200 hesaplamak için en az 200 satır bulunmalı
            if len(df_stock) < 200:
                continue
                
            # İndikatörleri hesapla ve kaydet
            df_indicators = calculate_technical_indicators(df_stock)
            scanned_data[sym] = df_indicators
        except:
            continue
            
    return scanned_data


# Başlık Bölümü (Bloomberg Terminal Estetiği)
st.markdown("""
<div style="background-color: #131722; border-left: 5px solid #2979ff; padding: 15px; border-radius: 4px; margin-bottom: 25px; border: 1px solid #2a2e39;">
    <h1 style="margin:0; font-size: 2.2em; color: #ffffff; font-weight: bold; letter-spacing: -0.02em;">⚡ BIST KISA VADE TARAMA & STRATEJİ ÜSSÜ</h1>
    <p style="margin: 5px 0 0 0; color: #8f96a3; font-size: 0.95em; letter-spacing: 0.02em;">Kısa Vadeli Patlama Potansiyeli Taşıyan Hisseleri Avlama ve Analiz Terminali</p>
</div>
""", unsafe_allow_html=True)


# Sekmeleri Oluşturma
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Gelişmiş Teknik Kurulum (Tab 1)", 
    "📡 VWMA Radar & Combo Tarayıcı (Tab 2)", 
    "🎯 SMI Momentum Patlama Avcısı (Tab 3)",
    "⚡ Trader Tetik Odası (Tab 4)",
    "🚨 Connors RSI(2) Panik Avcısı (Tab 5)",
    "💣 TTM Squeeze (Tab 6)"
])


# ==========================================
# SEKME 1: GELİŞMİŞ TEKNİK KURULUM
# ==========================================
with tab1:
    st.markdown("<h3 style='color:#2979ff; margin-bottom:15px;'>🔍 Hisse Röntgen Odası (Derin Teknik Röntgen)</h3>", unsafe_allow_html=True)
    
    # Form Düzeni
    col_inp1, col_inp2, col_inp3 = st.columns([2, 2, 1])
    with col_inp1:
        stock_input = st.text_input("BIST Hisse Kodu (Örn: ASELS, THYAO):", value="ASELS").upper().strip()
    with col_inp2:
        timeframe_input = st.selectbox(
            "Zaman Dilimi Seçin:",
            options=["1h", "4h", "1d", "1wk"],
            format_func=lambda x: "1 Saatlik (1h)" if x=="1h" else ("4 Saatlik (4h - Simüle)" if x=="4h" else ("Günlük (1d)" if x=="1d" else "Haftalık (1wk)")),
            index=2
        )
    with col_inp3:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        analyze_btn = st.button("Röntgen Çek 📊", use_container_width=True)
        
    if analyze_btn or stock_input:
        try:
            with st.spinner(f"{stock_input} verileri indiriliyor ve teknik röntgen analizi yapılıyor..."):
                df = fetch_single_stock(stock_input, timeframe_input)
                
                last_row = df.iloc[-1]
                close_price = float(last_row['close'])
                prev_close = float(df.iloc[-2]['close']) if len(df) > 1 else close_price
                change_pct = ((close_price - prev_close) / prev_close) * 100
                
                # Destek ve Direnç Sıralama Matematiği
                bb_upper = float(last_row['bb_upper'])
                bb_lower = float(last_row['bb_lower'])
                high_20 = float(df.tail(20)['high'].max())
                low_20 = float(df.tail(20)['low'].min())
                
                # REVERSED FIBONACCI (Swing Trade Mantığı: Tepe = 0.0%, Dip = 100.0%)
                recent_50 = df.tail(50)
                max_50 = float(recent_50['high'].max())
                min_50 = float(recent_50['low'].min())
                diff = max_50 - min_50
                
                fib_levels = {
                    "Fib 0.0% (Zirve)": max_50,
                    "Fib 23.6%": max_50 - diff * 0.236,
                    "Fib 38.2%": max_50 - diff * 0.382,
                    "Fib 50.0%": max_50 - diff * 0.500,
                    "Fib 61.8% (Altın Oran)": max_50 - diff * 0.618,
                    "Fib 78.6%": max_50 - diff * 0.786,
                    "Fib 100.0% (Dip)": min_50
                }
                
                # Seviye adayları
                candidates = [
                    ("Bollinger Üst Bant", bb_upper),
                    ("Bollinger Alt Bant", bb_lower),
                    ("20 Günlük En Yüksek", high_20),
                    ("20 Günlük En Düşük", low_20)
                ]
                for f_name, f_val in fib_levels.items():
                    candidates.append((f_name, f_val))
                    
                resistances = []
                supports = []
                for name, val in candidates:
                    if val > close_price:
                        resistances.append((name, val))
                    elif val < close_price:
                        supports.append((name, val))
                        
                resistances.sort(key=lambda x: x[1])
                supports.sort(key=lambda x: x[1], reverse=True)
                
                r1_name, r1_val = resistances[0] if len(resistances) > 0 else ("Direnç 1", high_20)
                r2_name, r2_val = resistances[1] if len(resistances) > 1 else ("Direnç 2", bb_upper)
                s1_name, s1_val = supports[0] if len(supports) > 0 else ("Destek 1", low_20)
                s2_name, s2_val = supports[1] if len(supports) > 1 else ("Destek 2", bb_lower)
                
                # DİNAMİK TRADE SETUP HESAPLAMA (ATR & Destek/Direnç Tabanlı)
                atr_val = last_row['atr_14']
                
                # Stop-Loss: En yakın desteğin %1.5 altına veya ATR paylı altına yerleştirilir
                stop_price = s1_val * 0.985
                if stop_price >= close_price * 0.99:
                    stop_price = close_price - (1.5 * atr_val)
                stop_price = min(stop_price, close_price * 0.99) # Her durumda fiyattan küçük olmalı
                
                # Hedef (Kâr Al): En yakın direncin hafif altına yerleştirilir. 
                # Çok yakınsa R2 hedeflenir.
                target_price = r1_val * 0.995
                if target_price <= close_price * 1.025:
                    target_price = r2_val * 0.995
                if target_price <= close_price * 1.025:
                    target_price = close_price + (3.0 * atr_val)
                target_price = max(target_price, close_price * 1.01) # Her durumda fiyattan büyük olmalı
                
                rr_ratio = (target_price - close_price) / (close_price - stop_price) if (close_price - stop_price) != 0 else 1.0
                
                # TEKNİK PUANLAMA MOTORU (Dinamik 10 Üzerinden Puanlar)
                # A. Trend Yapısı Puanı
                if close_price > last_row['ema_20'] > last_row['ema_50'] > last_row['ema_200']:
                    trend_score = 10.0
                elif close_price > last_row['ema_20'] > last_row['ema_50']:
                    trend_score = 8.5
                elif close_price > last_row['ema_20']:
                    trend_score = 7.0
                elif close_price < last_row['ema_20'] and close_price > last_row['ema_50']:
                    trend_score = 5.0
                else:
                    trend_score = 3.0
                    
                # B. Momentum (RSI) Puanı
                rsi_val = last_row['rsi']
                if 50 <= rsi_val <= 65:
                    mom_score = 10.0
                elif 65 < rsi_val <= 75:
                    mom_score = 8.0
                elif rsi_val > 75:
                    mom_score = 6.0
                elif 40 <= rsi_val < 50:
                    mom_score = 7.0
                else:
                    mom_score = 4.0
                    
                # C. MACD Puanı
                if last_row['macd'] > last_row['macd_signal'] and last_row['macd_hist'] > 0:
                    macd_score = 10.0
                elif last_row['macd'] > last_row['macd_signal'] and last_row['macd_hist'] <= 0:
                    macd_score = 8.0
                elif last_row['macd'] <= last_row['macd_signal'] and last_row['macd_hist'] < 0:
                    macd_score = 4.0
                else:
                    macd_score = 5.0
                    
                # D. Destek/Direnç Puanı (Mevcut fiyatın kanaldaki yeri)
                range_width = r1_val - s1_val
                price_position = (close_price - s1_val) / range_width if range_width > 0 else 0.5
                if price_position <= 0.3:
                    sr_score = 10.0  # Desteğe yakın (Alım Bölgesi)
                elif 0.3 < price_position <= 0.7:
                    sr_score = 7.5   # Orta Kanal
                else:
                    sr_score = 5.0   # Dirence yakın (Satım Bölgesi)
                    
                # E. Risk/Getiri Oranı Puanı
                if rr_ratio >= 2.0:
                    rr_score = 10.0
                elif 1.5 <= rr_ratio < 2.0:
                    rr_score = 8.0
                elif 1.0 <= rr_ratio < 1.5:
                    rr_score = 6.0
                else:
                    rr_score = 4.0
                    
                # Genel Teknik Skor
                genel_skor = (trend_score * 0.25) + (mom_score * 0.2) + (macd_score * 0.2) + (sr_score * 0.2) + (rr_score * 0.15)
                genel_skor = round(genel_skor, 1)
                
                # Dinamik Senaryo Olasılıkları
                if genel_skor >= 7.0:
                    up_prob, side_prob, down_prob = 65, 20, 15
                elif 4.5 <= genel_skor < 7.0:
                    up_prob, side_prob, down_prob = 35, 45, 20
                else:
                    up_prob, side_prob, down_prob = 15, 30, 55
                    
                # ==========================================
                # ARAYÜZ YERLEŞİMİ - ÜST PANEL (Teknik & Puanlama)
                # ==========================================
                col_chart, col_score_section = st.columns([2, 1])
                
                with col_chart:
                    df_plot = df.tail(100)
                    fig = make_subplots(
                        rows=4, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.48, 0.14, 0.16, 0.22]
                    )
                    
                    # Row 1: Candlestick + EMAs + BB
                    fig.add_trace(go.Candlestick(
                        x=df_plot.index, open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'],
                        name="Fiyat", increasing_line_color='#00e676', decreasing_line_color='#ff1744',
                        increasing_fillcolor='rgba(0, 230, 118, 0.2)', decreasing_fillcolor='rgba(255, 23, 68, 0.2)'
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ema_20'], name="EMA20", line=dict(color='#2979ff', width=1.5)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ema_50'], name="EMA50", line=dict(color='#ff9100', width=1.5)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ema_200'], name="EMA200", line=dict(color='#9c27b0', width=1.5)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['bb_upper'], name="Bollinger Üst", line=dict(color='#455a64', width=1, dash='dash')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['bb_lower'], name="Bollinger Alt", line=dict(color='#455a64', width=1, dash='dash')), row=1, col=1)
                    
                    # Row 2: RSI
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['rsi'], name="RSI", line=dict(color='#00e676', width=1.5)), row=2, col=1)
                    fig.add_shape(type="line", x0=df_plot.index[0], x1=df_plot.index[-1], y0=70, y1=70, line=dict(color="#ff1744", width=1, dash="dot"), row=2, col=1)
                    fig.add_shape(type="line", x0=df_plot.index[0], x1=df_plot.index[-1], y0=30, y1=30, line=dict(color="#00e676", width=1, dash="dot"), row=2, col=1)
                    
                    # Row 3: MACD
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['macd'], name="MACD", line=dict(color='#2979ff', width=1.5)), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['macd_signal'], name="Sinyal", line=dict(color='#ff9100', width=1.5)), row=3, col=1)
                    macd_colors = ['#00e676' if val >= 0 else '#ff1744' for val in df_plot['macd_hist']]
                    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['macd_hist'], name="Histogram", marker_color=macd_colors), row=3, col=1)
                    
                    # Row 4: SMI
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['smi'], name="SMI", line=dict(color='#00e676', width=1.8)), row=4, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['smi_ema'], name="SMI Ema", line=dict(color='#ff9100', width=1.5, dash='dash')), row=4, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['smi_vwap'], name="SMI VWAP", line=dict(color='#2979ff', width=2)), row=4, col=1)
                    fig.add_shape(type="line", x0=df_plot.index[0], x1=df_plot.index[-1], y0=0, y1=0, line=dict(color="#ffffff", width=1), row=4, col=1)
                    
                    fig.update_layout(
                        height=540, paper_bgcolor='#0b0e11', plot_bgcolor='#0b0e11',
                        showlegend=False, xaxis_rangeslider_visible=False,
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    fig.update_xaxes(showgrid=True, gridcolor='#2a2e39', linecolor='#2a2e39')
                    fig.update_yaxes(showgrid=True, gridcolor='#2a2e39', linecolor='#2a2e39')
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col_score_section:
                    with st.container(border=True):
                        if genel_skor >= 7.0:
                            st.markdown(f"### 🏆 GENEL TEKNİK SKOR\n## :green[{genel_skor} / 10]")
                        elif 4.5 <= genel_skor < 7.0:
                            st.markdown(f"### 🏆 GENEL TEKNİK SKOR\n## :orange[{genel_skor} / 10]")
                        else:
                            st.markdown(f"### 🏆 GENEL TEKNİK SKOR\n## :red[{genel_skor} / 10]")
                            
                    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                    
                    # Native table for kategorik puanlama
                    def get_stars_str(score_val):
                        filled = int(round(score_val / 2))
                        return "★" * filled + "☆" * (5 - filled)
                        
                    puan_df = pd.DataFrame([
                        {"Kriter": "Trend Yapısı", "Skor": f"{trend_score}/10", "Derece": get_stars_str(trend_score)},
                        {"Kriter": "Momentum (RSI)", "Skor": f"{mom_score}/10", "Derece": get_stars_str(mom_score)},
                        {"Kriter": "MACD Kesişimi", "Skor": f"{macd_score}/10", "Derece": get_stars_str(macd_score)},
                        {"Kriter": "Destek/Direnç Konumu", "Skor": f"{sr_score}/10", "Derece": get_stars_str(sr_score)},
                        {"Kriter": "Risk/Getiri Durumu", "Skor": f"{rr_score}/10", "Derece": get_stars_str(rr_score)}
                    ])
                    st.table(puan_df)
                    
                # ==========================================
                # NATIVE KARTLAR (Alt Satır)
                # ==========================================
                col_fib, col_trend, col_scen = st.columns([1, 1, 1])
                
                with col_fib:
                    with st.container(border=True):
                        st.markdown("#### 📐 FIBONACCI RETRACEMENT (SWING)")
                        fib_df = pd.DataFrame([
                            {"Fib Seviyesi": k, "Fiyat": f"{v:.2f} TL"} for k, v in fib_levels.items()
                        ])
                        st.dataframe(fib_df, use_container_width=True, hide_index=True)
                        
                with col_trend:
                    with st.container(border=True):
                        st.markdown("#### 📈 TREND & MACD ANALİZİ")
                        
                        ema20_slope = ((last_row['ema_20'] - df.iloc[-5]['ema_20']) / df.iloc[-5]['ema_20']) * 100 if len(df) > 5 else 0.0
                        if ema20_slope > 0.05:
                            trend_label = "Yön Yukarı"
                            trend_color = "green"
                            trend_desc = "Fiyat EMA20 üzerinde, boğa ivmesi güçlü."
                        elif ema20_slope < -0.05:
                            trend_label = "Yön Aşağı"
                            trend_color = "red"
                            trend_desc = "Fiyat EMA20 altında, satıcı baskısı hâkim."
                        else:
                            trend_label = "Yön Yatay"
                            trend_color = "orange"
                            trend_desc = "Yatay konsolidasyon ve sıkışma hâkim."
                            
                        macd_bullish = last_row['macd'] > last_row['macd_signal']
                        macd_label = "AL Sinyali" if macd_bullish else "SAT Sinyali"
                        macd_color = "green" if macd_bullish else "red"
                        
                        st.metric(label="MA20 Trend Eğilimi", value=trend_label, delta=f"{ema20_slope:+.2f}%")
                        st.metric(label="MACD Sinyal Durumu", value=macd_label, delta="Boğa Kesişimi" if macd_bullish else "Ayı Kesişimi", delta_color="normal" if macd_bullish else "inverse")
                        
                        if trend_color == "green":
                            st.success(trend_desc)
                        elif trend_color == "red":
                            st.error(trend_desc)
                        else:
                            st.warning(trend_desc)
                            
                with col_scen:
                    with st.container(border=True):
                        st.markdown("#### 🛡️ RISK/GETİRİ VE SENARYOLAR")
                        
                        setup_df = pd.DataFrame([
                            {"Parametre": "Giriş Fiyatı", "Değer": f"{close_price:.2f} TL"},
                            {"Parametre": "🎯 Kâr Al (Target)", "Değer": f"{target_price:.2f} TL"},
                            {"Parametre": "🛑 Zarar Kes (Stop)", "Değer": f"{stop_price:.2f} TL"},
                            {"Parametre": "📊 Risk/Ödül Katsayısı", "Değer": f"{rr_ratio:.2f}"}
                        ])
                        st.dataframe(setup_df, use_container_width=True, hide_index=True)
                        
                        st.markdown(f"""
                        **Olası Senaryolar:**
                        * 🟢 **Yükseliş (%{up_prob}):** Direnç aşıldığında hedef **{target_price:.2f} TL**.
                        * 🟡 **Yatay (%{side_prob}):** {s1_val:.2f} - {r1_val:.2f} TL bandında seyir.
                        * 🔴 **Kırılım (%{down_prob}):** Desteğin kırılımıyla stop-loss tetiklenmesi (**{stop_price:.2f} TL**).
                        """)
                        
                # ==========================================
                # EN ALT KISIM - AI BAŞ ANALİST YORUMU
                # ==========================================
                analyst_summary = f"""
                **Piyasa Teknik Perspektifi:**  
                {stock_input} pay senedi üzerinde gerçekleştirdiğimiz röntgen analizi, hissenin teknik momentum yapısı ile oynaklık bantlarının yeni bir yön arayışında olduğunu göstermektedir. Fiyatın EMA20 ({last_row['ema_20']:.2f} TL) üzerindeki performansı trend yönünün tayininde kritik bir rol oynamaktadır. Son dönem ATR oynaklık katsayısı ({atr_val:.2f} TL), piyasadaki potansiyel dalgalanma boyuna dair rehberlik sunmaktadır.
                
                **Taktiksel Giriş & Risk Dağılım Planı:**  
                Kısa vadeli risk-ödül oranını maksimize etmek amacıyla kurgulanan bu kurulumda, en yakın güçlü yatay destek olan **{s1_val:.2f} TL** seviyesinin hemen altında **{stop_price:.2f} TL** bölgesi zarar kes (Stop-Loss) seviyesi olarak kurgulanmıştır. Yukarı yönlü olası bir momentum genişlemesinde ise önündeki güçlü yatay direncin hemen altında yer alan **{target_price:.2f} TL** seviyesi birincil kâr al (Target) hedefi olarak takip edilmelidir. 
                
                **Senaryolar ve Fon Yönetim Planı:**  
                Mevcut teknik skor ({genel_skor}/10) dikkate alındığında, fiyatın direnç seviyelerine doğru yönelme olasılığı %{up_prob} düzeyinde değerlendirilmektedir. Destek altı sarkmalarda sermaye koruma refleksiyle zarar kes disiplini hassasiyetle çalıştırılmalıdır. Risk/Ödül katsayısının {rr_ratio:.2f} olması, swing trade kurgusunun matematiksel olarak son derece avantajlı bir risk profili sunduğunu tescillemektedir.
                
                *Yasal Uyarı: Bu analiz, gelişmiş teknik algoritma süzgeçlerinden süzülerek üretilmiş olup yatırım danışmanlığı kapsamında değildir.*
                """
                
                st.markdown("### 🤖 HİSSE TEKNİK YORUM ÖZETİ (AI BAŞ ANALİST RAPORU)")
                st.info(analyst_summary)
                
        except Exception as e:
            st.error(f"Hata: {e}")


# ==========================================
# SEKME 2: VWMA RADAR & COMBO TARAYICI
# ==========================================
with tab2:
    st.markdown("<h3 style='color:#2979ff; margin-bottom:5px;'>📡 VWMA Radar & Hacim Combo Tarayıcı</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8f96a3; font-size:0.9em;'>Fiyatı 20 günlük Hacim Ağırlıklı Hareketli Ortalamasının (VWMA) üzerinde ve ortalamaya en fazla %2.5 yakınlıkta (veya bugün yeni yukarı kesmiş) olan potansiyel patlama adayları.</p>", unsafe_allow_html=True)
    
    start_scan_btn = st.button("🚀 BIST Listesini Taramaya Başla", use_container_width=True)
    
    # Session state kontrolleri
    if "vwma_signals" not in st.session_state:
        st.session_state.vwma_signals = []
        
    if start_scan_btn:
        st.session_state.vwma_signals = []
        with st.spinner("Tüm BIST listesi toplu indiriliyor ve analiz ediliyor..."):
            # Toplu tarama yap
            all_data = batch_scan_bist_data()
            
            for sym, df_stock in all_data.items():
                last_row = df_stock.iloc[-1]
                close = float(last_row['close'])
                prev_close = float(df_stock.iloc[-2]['close']) if len(df_stock) > 1 else close
                vwma_val = float(last_row['vwma'])
                prev_vwma = float(df_stock.iloc[-2]['vwma']) if len(df_stock) > 1 else vwma_val
                rsi_val = float(last_row['rsi'])
                mom_val = float(last_row['momentum'])
                vol_ratio_val = float(last_row['vol_ratio'])
                
                # Şart 1: Fiyat VWMA üzerinde ve yakınlığı <= %2.5 OLMALI (veya bugün yeni kesmiş olmalı)
                distance = (close - vwma_val) / vwma_val
                crossed_above = close >= vwma_val and prev_close < prev_vwma
                
                if (close > vwma_val and distance <= 0.025) or crossed_above:
                    vol_ratio_10_val = float(last_row['vol_ratio_10']) if 'vol_ratio_10' in last_row else vol_ratio_val
                    st.session_state.vwma_signals.append({
                        "sym": sym,
                        "price": close,
                        "change": ((close - prev_close) / prev_close) * 100,
                        "vwma": vwma_val,
                        "distance": distance * 100,
                        "rsi": rsi_val,
                        "momentum": mom_val,
                        "vol_ratio": vol_ratio_val,
                        "vol_ratio_10": vol_ratio_10_val
                    })
                    
        st.success(f"Tarama tamamlandı! {len(st.session_state.vwma_signals)} hisse kriterleri sağlıyor.")
        
    # Tarama sonuçları varsa tabloları oluştur
    if st.session_state.vwma_signals:
        signals_df = pd.DataFrame(st.session_state.vwma_signals)
        
        # Önbellek çakışmasına karşı koruma
        if 'vol_ratio_10' not in signals_df.columns:
            signals_df['vol_ratio_10'] = signals_df['vol_ratio'] if 'vol_ratio' in signals_df.columns else 1.0
            
        # Tabloları st.dataframe ile formatlı çizdiren yardımcı fonksiyon
        def display_signal_dataframe(df_subset: pd.DataFrame, signal_type_label: str):
            if df_subset.empty:
                st.info("Kriterleri karşılayan hiçbir hisse bulunamadı.")
                return
                
            render_df = pd.DataFrame()
            render_df["Hisse Kodu"] = df_subset["sym"]
            render_df["Sinyal Tipi"] = signal_type_label
            render_df["Fiyat (TL)"] = df_subset["price"]
            render_df["VWMA (TL)"] = df_subset["vwma"]
            render_df["VWMA Uzaklık (%)"] = df_subset["distance"]
            render_df["RSI (14)"] = df_subset["rsi"]
            render_df["Momentum"] = df_subset["momentum"]
            render_df["Hacim Katı (10g)"] = df_subset["vol_ratio_10"]
            
            st.dataframe(
                render_df,
                column_config={
                    "Hisse Kodu": st.column_config.TextColumn("Hisse Kodu"),
                    "Sinyal Tipi": st.column_config.TextColumn("Sinyal Tipi"),
                    "Fiyat (TL)": st.column_config.NumberColumn("Fiyat (TL)", format="%.2f TL"),
                    "VWMA (TL)": st.column_config.NumberColumn("VWMA (TL)", format="%.2f TL"),
                    "VWMA Uzaklık (%)": st.column_config.NumberColumn("VWMA Uzaklık (%)", format="%.2f%%"),
                    "RSI (14)": st.column_config.NumberColumn("RSI (14)", format="%.1f"),
                    "Momentum": st.column_config.NumberColumn("Momentum", format="%.1f"),
                    "Hacim Katı (10g)": st.column_config.NumberColumn("Hacim Katı (10g)", format="%.2fx")
                },
                use_container_width=True,
                hide_index=True
            )
            
        # Filtre 1: Tüm Sinyaller
        f1_df = signals_df.copy()
        
        # Filtre 2: VWMA + RSI (40-65 arası)
        f2_df = signals_df[(signals_df['rsi'] >= 40) & (signals_df['rsi'] <= 65)].copy()
        
        # Filtre 3: VWMA + Momentum (>100)
        f3_df = signals_df[signals_df['momentum'] > 100.0].copy()
        
        # Filtre 4: 5 YILDIZ KOMBO (Tüm kriterler)
        f4_df = signals_df[
            (signals_df['rsi'] >= 40) & 
            (signals_df['rsi'] <= 65) & 
            (signals_df['momentum'] > 100.0) & 
            (signals_df['vol_ratio_10'] >= 1.5)
        ].copy()
        
        # A. 5 YILDIZ KOMBO (EN ÜSTE VE EN DİKKAT ÇEKİCİ YERE)
        st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color:rgba(0, 230, 118, 0.05); border:1px solid #00e676; border-radius:6px; padding:15px; margin-bottom:10px;">
            <span class="badge badge-green" style="font-size:1.1em; padding:5px 12px;">🌟 5 YILDIZ KOMBO PATLAMA LİSTESİ</span>
            <p style="margin:8px 0 0 0; color:#8f96a3; font-size:0.9em;">RSI nötr-pozitif (40-65), Momentum > 100 ve son gün hacmi 10 günlük hacim SMA'sının en az %50 üzerinde (Hacim Katı >= 1.5) olan süper potansiyelli patlama adayları.</p>
        </div>
        """, unsafe_allow_html=True)
        display_signal_dataframe(f4_df, "🌟 5 YILDIZ KOMBO")
        
        st.markdown("---")
        
        # B. DİĞER ALT TABLOLAR
        st.markdown("<h4 style='color:#ffffff; margin-top:20px;'>📊 1. Tüm VWMA Yakın Sinyaller</h4>", unsafe_allow_html=True)
        display_signal_dataframe(f1_df, "VWMA Giriş")
        
        st.markdown("<h4 style='color:#ffffff; margin-top:20px;'>📊 2. VWMA + RSI Süzgeci (RSI 40 - 65)</h4>", unsafe_allow_html=True)
        display_signal_dataframe(f2_df, "VWMA + RSI Uyumlu")
        
        st.markdown("<h4 style='color:#ffffff; margin-top:20px;'>📊 3. VWMA + Momentum Süzgeci (Momentum > 100)</h4>", unsafe_allow_html=True)
        display_signal_dataframe(f3_df, "VWMA + Momentum Uyumlu")
        
    else:
        st.info("⚠️ Taramayı başlatmak için lütfen yukarıdaki 'BIST Listesini Taramaya Başla' butonuna basın.")


# ==========================================
# SEKME 3: SMI MOMENTUM PATLAMA AVCISI
# ==========================================
with tab3:
    st.markdown("<h3 style='color:#2979ff; margin-bottom:5px;'>🎯 SMI Momentum Patlama Avcısı</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8f96a3; font-size:0.9em;'>2-4 haftalık patlama potansiyeli taşıyan, SMI, SMI_VWAP ve Bollinger orta bandının üçünü birden sağlayan 🚨 SÜPER SİNYAL üreten hisseler.</p>", unsafe_allow_html=True)
    
    start_smi_btn = st.button("🚨 SMI Patlama Aramasını Başlat", use_container_width=True)
    
    if "smi_signals" not in st.session_state:
        st.session_state.smi_signals = []
        
    if start_smi_btn:
        st.session_state.smi_signals = []
        with st.spinner("BIST listesi SMI kırılımlarına göre taranıyor..."):
            all_data = batch_scan_bist_data()
            
            for sym, df_stock in all_data.items():
                last_row = df_stock.iloc[-1]
                close = float(last_row['close'])
                prev_close = float(df_stock.iloc[-2]['close']) if len(df_stock) > 1 else close
                
                smi = float(last_row['smi'])
                smi_ema = float(last_row['smi_ema'])
                smi_vwap = float(last_row['smi_vwap'])
                bb_middle = float(last_row['bb_middle'])
                
                # 3 Şartı Kontrol Et
                cond1 = smi >= smi_ema and smi < 0
                cond2 = smi_vwap > smi and smi_vwap > smi_ema
                cond3 = close >= bb_middle
                
                score = 0.0
                if cond1: score += 4.0
                if cond2: score += 3.0
                if cond3: score += 3.0
                
                if cond1 and cond2 and cond3:
                    rsi_val = float(last_row['rsi']) if 'rsi' in last_row else 50.0
                    
                    if 'vol_ratio_10' in last_row:
                        vol_ratio_10_val = float(last_row['vol_ratio_10'])
                    else:
                        # Önbellekte yoksa anlık hesapla
                        vol_sma_10 = df_stock['volume'].rolling(window=10).mean()
                        vol_ratio_10_series = df_stock['volume'] / vol_sma_10.replace(0, np.nan)
                        vol_ratio_10_val = float(vol_ratio_10_series.fillna(1.0).iloc[-1])
                        
                    vwma_val = float(last_row['vwma']) if 'vwma' in last_row else close
                    distance_val = ((close - vwma_val) / vwma_val) * 100 if vwma_val != 0 else 0
                    
                    # Süper Sinyal verenleri kaydet
                    st.session_state.smi_signals.append({
                        "sym": sym,
                        "price": close,
                        "change": ((close - prev_close) / prev_close) * 100,
                        "smi": smi,
                        "smi_ema": smi_ema,
                        "smi_vwap": smi_vwap,
                        "rsi": rsi_val,
                        "vol_ratio_10": vol_ratio_10_val,
                        "distance": distance_val,
                        "score": score
                    })
                    
        st.success(f"Tarama tamamlandı! {len(st.session_state.smi_signals)} hisse '🚨 SÜPER SİNYAL' üretti.")
        
    if st.session_state.smi_signals:
        # Sonuçları Kartlar halinde yan yana basmak için 3 kolonlu bir grid sistemi
        cols = st.columns(3)
        for idx, signal in enumerate(st.session_state.smi_signals):
            target = signal['price'] * 1.10
            stop = signal['price'] * 0.97
            col_target = cols[idx % 3]
            
            with col_target:
                with st.container(border=True):
                    # Header Row (Symbol and Badge)
                    col_h1, col_h2 = st.columns([2, 2])
                    with col_h1:
                        st.markdown(f"### ⚡ {signal['sym']}")
                    with col_h2:
                        st.markdown('<div style="text-align:right;"><span class="badge badge-green">🚨 SÜPER SİNYAL</span></div>', unsafe_allow_html=True)
                    
                    # Current Price Metric
                    st.metric(
                        label="Güncel Fiyat", 
                        value=f"{signal['price']:.2f} TL", 
                        delta=f"{signal['change']:+.2f}%"
                    )
                    
                    st.markdown("---")
                    
                    # Indicators Grid
                    st.markdown("**🔍 Teknik Değerler**")
                    ind_col1, ind_col2 = st.columns(2)
                    with ind_col1:
                        st.metric("SMI Değeri", f"{signal['smi']:.2f}")
                        st.metric("Hacim Katı (10g)", f"{signal['vol_ratio_10']:.2f}x")
                    with ind_col2:
                        st.metric("RSI (14)", f"{signal['rsi']:.1f}")
                        st.metric("VWMA Uzaklık", f"{signal['distance']:.2f}%")
                        
                    st.markdown("---")
                    
                    # Realistic Trade Setup Box
                    st.markdown("**📈 Gerçekçi Setup (Hedef & Stop)**")
                    st.markdown(f"""
                    | Seviye Tipi | Fiyat Seviyesi |
                    | :--- | :---: |
                    | **Giriş Fiyatı** | {signal['price']:.2f} TL |
                    | **🎯 Kâr Al (+10% Target)** | <span style="color:#00e676; font-weight:bold;">{target:.2f} TL</span> |
                    | **🛑 Zarar Kes (-3% Stop)** | <span style="color:#ff1744; font-weight:bold;">{stop:.2f} TL</span> |
                    """, unsafe_allow_html=True)
    else:
        st.info("⚠️ Taramayı başlatmak için lütfen yukarıdaki 'SMI Patlama Aramasını Başlat' butonuna basın.")


# ==========================================
# SEKME 4: TRADER TETİK ODASI (TRIGGER ROOM)
# ==========================================
with tab4:
    st.markdown("<h3 style='color:#2979ff; margin-bottom:5px;'>⚡ Trader Tetik Odası (Trigger Room)</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8f96a3; font-size:0.9em;'>Price Action (Pin Bar), Volatilite Sıkışması ve BIST100 Endeksine göre Güçlü Duruş sergileyen Giriş Zamanlaması tespiti.</p>", unsafe_allow_html=True)
    
    start_trigger_btn = st.button("⚡ Tetik Odası Taramasını Başlat", width="stretch")
    
    if "trigger_signals" not in st.session_state:
        st.session_state.trigger_signals = []
        
    if start_trigger_btn:
        st.session_state.trigger_signals = []
        with st.spinner("BIST100 ve hisse verileri indirilip Price Action / Sıkışma taraması yapılıyor..."):
            # 1. BIST100 Endeks Getirisini Hesapla
            try:
                index_df = yf.download("XU100.IS", period="15d", interval="1d", progress=False)
                index_df = clean_yf_df(index_df)
                index_close = index_df['close']
                index_5d_ret = ((index_close.iloc[-1] - index_close.iloc[-6]) / index_close.iloc[-6]) * 100
            except:
                index_5d_ret = 0.0
                
            # 2. Hisse Verilerini Toplu Çek
            all_data = batch_scan_bist_data()
            
            for sym, df_stock in all_data.items():
                if len(df_stock) < 10:
                    continue
                    
                last_row = df_stock.iloc[-1]
                close = float(last_row['close'])
                prev_close = float(df_stock.iloc[-2]['close']) if len(df_stock) > 1 else close
                open_last = float(last_row['open'])
                high = float(last_row['high'])
                low = float(last_row['low'])
                
                # A. Endeksüstü Güç
                stock_5d_ret = ((df_stock['close'].iloc[-1] - df_stock['close'].iloc[-6]) / df_stock['close'].iloc[-6]) * 100
                rel_strength = stock_5d_ret - index_5d_ret
                is_index_strong = rel_strength > 0
                
                # B. Price Action - Pin Bar (Alıcı Baskısı)
                body = abs(open_last - close)
                lower_wick = min(open_last, close) - low
                is_pin_bar = (body > 0 and lower_wick / body >= 1.5) or (body == 0 and lower_wick > 0)
                
                # C. Volatilite Sıkışması (Bollinger Squeeze)
                bb_width = df_stock['bb_width']
                is_squeeze = bb_width.iloc[-1] == bb_width.tail(10).min()
                
                # Sıkışma Oranı (Bandwidth)
                squeeze_ratio = float(last_row['bb_width']) * 100
                
                # Tetik Durumu Kararı
                if is_index_strong and is_pin_bar and is_squeeze:
                    trigger_status = "🔥 KUSURSUZ ZAMANLAMA (TETİĞE BAS)"
                else:
                    parts = []
                    if is_index_strong: parts.append("Endeksüstü Güç")
                    if is_pin_bar: parts.append("Alıcı Baskısı")
                    if is_squeeze: parts.append("Sıkışma")
                    trigger_status = " + ".join(parts) if parts else "NÖTR"
                    
                st.session_state.trigger_signals.append({
                    "sym": sym,
                    "price": close,
                    "rel_strength": rel_strength,
                    "mum_yapisi": "⚡ ALICI BASKISI (GİRİŞE UYGUN)" if is_pin_bar else "NORMAL",
                    "squeeze_ratio": squeeze_ratio,
                    "status": trigger_status,
                    "is_perfect": is_index_strong and is_pin_bar and is_squeeze
                })
                
        st.success(f"Tetik Odası taraması tamamlandı! {len(st.session_state.trigger_signals)} hisse analiz edildi.")
        
    if st.session_state.trigger_signals:
        trigger_df = pd.DataFrame(st.session_state.trigger_signals)
        
        # VIP Hisseler (Kusursuz 3'te 3 yapanlar)
        perfect_df = trigger_df[trigger_df['is_perfect'] == True].copy()
        
        # Potansiyel Hisseler (En az 1 veya 2 şartı sağlayanlar, yani NÖTR olmayan ve VIP olmayanlar)
        potential_df = trigger_df[(trigger_df['status'] != "NÖTR") & (trigger_df['is_perfect'] == False)].copy()
        
        # Tabloları çizen yardımcı fonksiyon
        def draw_trigger_table(df_subset: pd.DataFrame):
            df_subset = df_subset.sort_values(by='rel_strength', ascending=False)
            st.dataframe(
                df_subset.drop(columns=['is_perfect']),
                column_config={
                    "sym": st.column_config.TextColumn("Hisse Kodu"),
                    "price": st.column_config.NumberColumn("Son Fiyat", format="%.2f TL"),
                    "rel_strength": st.column_config.NumberColumn("Endeks Gücü (%)", format="%+.2f%%"),
                    "mum_yapisi": st.column_config.TextColumn("Mum Yapısı"),
                    "squeeze_ratio": st.column_config.NumberColumn("Sıkışma Oranı (Bant)", format="%.2f%%"),
                    "status": st.column_config.TextColumn("TETİK DURUMU")
                },
                use_container_width=True,
                hide_index=True
            )
            
        # 1. EN TEPEYE: VIP TETİKTEKİLER
        st.markdown("<h4 style='color:#00e676; margin-top:20px;'>🔥 VIP TETİKTEKİLER (KUSURSUZ ALIM FIRSATI)</h4>", unsafe_allow_html=True)
        
        if not perfect_df.empty:
            col_vip_left, col_vip_right = st.columns([2, 1])
            
            with col_vip_left:
                draw_trigger_table(perfect_df)
                
            with col_vip_right:
                # VIP hisseler için hedef kartlar
                for _, row in perfect_df.iterrows():
                    target = row['price'] * 1.10
                    stop = row['price'] * 0.97
                    with st.container(border=True):
                        st.markdown(f"**🎯 Giriş Tetiği Aktif: {row['sym']}**")
                        st.markdown(f"**Giriş Fiyatı:** {row['price']:.2f} TL")
                        st.write("🎯 **Kâr Al (+10% Target):** :green[" + f"{target:.2f} TL" + "]")
                        st.write("🛑 **Zarar Kes (-3% Stop):** :red[" + f"{stop:.2f} TL" + "]")
        else:
            st.markdown("""
            <div class="st-card" style="border-left: 4px solid #ff9100; background-color: #1a231f; padding: 15px;">
                <span style="color:#ff9100; font-weight:bold; font-size:1.0em;">ℹ️ VIP Sinyal Bildirimi</span>
                <p style="margin:5px 0 0 0; color:#8f96a3; font-size:0.9em;">Şu an 3'te 3 yapan kusursuz hisse yok, aşağıdaki potansiyel listeyi inceleyin.</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # 2. ALT KISMA: POTANSİYEL TETİK LİSTESİ
        st.markdown("<h4 style='color:#ffffff; margin-top:20px;'>📋 Gelişmekte Olan Potansiyel Tetik Listesi</h4>", unsafe_allow_html=True)
        if not potential_df.empty:
            draw_trigger_table(potential_df)
        else:
            st.info("Kriterleri karşılayan potansiyel bir hisse bulunamadı.")
                        
    else:
        st.info("⚠️ Taramayı başlatmak için lütfen yukarıdaki 'Tetik Odası Taramasını Başlat' butonuna basın.")


# ==========================================
# SEKME 5: CONNORS RSI(2) PANİK AVCISI
# ==========================================
with tab5:
    st.markdown("<h3 style='color:#2979ff; margin-bottom:5px;'>🚨 Connors RSI(2) Panik Avcısı (Dipten Dönüş)</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8f96a3; font-size:0.9em;'>Larry Connors'ın yüksek kazanma oranlı Mean Reversion stratejisi. Fiyatı SMA 200 üzerinde (yükseliş trendinde) olan ve kısa vadeli RSI(2) değeri 10'un altına düşerek aşırı panik satışı yiyen kelepir hisseleri bulur. Çıkış hedefi 5 günlük EMA ortalamasıdır.</p>", unsafe_allow_html=True)
    
    start_connors_btn = st.button("🚨 Connors RSI(2) Taramasını Başlat", use_container_width=True)
    
    if "connors_signals" not in st.session_state:
        st.session_state.connors_signals = []
        
    if start_connors_btn:
        st.session_state.connors_signals = []
        with st.spinner("Tüm BIST listesi Connors RSI(2) kurallarına göre taranıyor (1 Yıllık veriler indiriliyor)..."):
            all_data = batch_scan_bist_data()
            
            for sym, df_stock in all_data.items():
                if len(df_stock) < 200:
                    continue
                    
                last_row = df_stock.iloc[-1]
                close = float(last_row['close'])
                sma_200 = float(last_row['sma_200'])
                rsi_2 = float(last_row['rsi_2'])
                ema_5 = float(last_row['ema_5'])
                
                # Larry Connors Kuralları:
                # 1. Uzun vadeli trend boğa: close > sma_200
                # 2. Tetikleyici aşırı satım: rsi_2 < 10
                if close > sma_200 and rsi_2 < 10.0:
                    st.session_state.connors_signals.append({
                        "sym": sym,
                        "price": close,
                        "sma_200": sma_200,
                        "rsi_2": rsi_2,
                        "ema_5": ema_5
                    })
                    
        st.success(f"Tarama tamamlandı! {len(st.session_state.connors_signals)} hisse aşırı satım (panik) bölgesinde tespit edildi.")
        
    if st.session_state.connors_signals:
        st.markdown("#### 🚨 CİNNET BÖLGESİNDEKİ KELEPİRLER")
        
        connors_df = pd.DataFrame(st.session_state.connors_signals)
        render_df = pd.DataFrame()
        render_df["Hisse Kodu"] = connors_df["sym"]
        render_df["Son Fiyat"] = connors_df["price"]
        render_df["SMA 200"] = connors_df["sma_200"]
        render_df["RSI(2) Değeri"] = connors_df["rsi_2"]
        render_df["🎯 Kâr Al Hedefi (EMA 5)"] = connors_df["ema_5"]
        
        st.dataframe(
            render_df,
            column_config={
                "Hisse Kodu": st.column_config.TextColumn("Hisse Kodu"),
                "Son Fiyat": st.column_config.NumberColumn("Son Fiyat", format="%.2f TL"),
                "SMA 200": st.column_config.NumberColumn("SMA 200", format="%.2f TL"),
                "RSI(2) Değeri": st.column_config.NumberColumn("RSI(2) Değeri", format="%.2f"),
                "🎯 Kâr Al Hedefi (EMA 5)": st.column_config.NumberColumn("🎯 Kâr Al Hedefi (EMA 5)", format="%.2f TL")
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        if start_connors_btn:
            st.info("Kriterleri karşılayan aşırı satım bölgesinde kelepir hisse bulunamadı.")
        else:
            st.info("⚠️ Taramayı başlatmak için lütfen yukarıdaki 'Connors RSI(2) Taramasını Başlat' butonuna basın.")


# ==========================================
# SEKME 6: TTM SQUEEZE
# ==========================================
with tab6:
    st.markdown("<h3 style='color:#2979ff; margin-bottom:5px;'>💣 TTM Squeeze (Volatilite Patlama Avcısı)</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8f96a3; font-size:0.9em;'>John Carter'ın fırtına öncesi sessizliği ve sert patlamaları yakalayan TTM Squeeze stratejisi. Bollinger Bantları (20, 2.0) Keltner Kanallarının (20, 1.5) içine girdiğinde enerji sıkışmaya başlar. Bantlar dışarı taştığı ilk gün momentum yönü yukarıysa patlama tetiklenir.</p>", unsafe_allow_html=True)
    
    start_squeeze_btn = st.button("💣 TTM Squeeze Taramasını Başlat", use_container_width=True)
    
    if "squeeze_active" not in st.session_state:
        st.session_state.squeeze_active = []
    if "squeeze_breakout" not in st.session_state:
        st.session_state.squeeze_breakout = []
        
    if start_squeeze_btn:
        st.session_state.squeeze_active = []
        st.session_state.squeeze_breakout = []
        
        with st.spinner("Tüm BIST listesi TTM Squeeze kriterlerine göre taranıyor..."):
            all_data = batch_scan_bist_data()
            
            for sym, df_stock in all_data.items():
                if len(df_stock) < 30:
                    continue
                    
                last_row = df_stock.iloc[-1]
                prev_row = df_stock.iloc[-2]
                
                close = float(last_row['close'])
                squeeze_on_today = bool(last_row['squeeze_on'])
                squeeze_on_prev = bool(prev_row['squeeze_on'])
                
                # Momentum yönü: MACD Hist eğimi
                macd_hist_today = float(last_row['macd_hist'])
                macd_hist_prev = float(prev_row['macd_hist'])
                mom_dir = "🚀 YUKARI" if macd_hist_today > macd_hist_prev else "🔻 AŞAĞI"
                
                if squeeze_on_today:
                    # Sıkışma aktif: Gün sayısını hesapla
                    squeeze_days = 0
                    for i in range(len(df_stock) - 1, -1, -1):
                        if df_stock['squeeze_on'].iloc[i]:
                            squeeze_days += 1
                        else:
                            break
                    st.session_state.squeeze_active.append({
                        "sym": sym,
                        "price": close,
                        "days": squeeze_days,
                        "mom_dir": mom_dir
                    })
                elif squeeze_on_prev and not squeeze_on_today:
                    # Bugün patlama tetiklendi (Sıkışma bitti) ve momentum yukarı
                    if macd_hist_today > macd_hist_prev:
                        st.session_state.squeeze_breakout.append({
                            "sym": sym,
                            "price": close,
                            "mom_dir": mom_dir
                        })
                        
        st.success(f"Tarama tamamlandı! {len(st.session_state.squeeze_active)} sıkışan, {len(st.session_state.squeeze_breakout)} patlama veren hisse tespit edildi.")
        
    # Tablo 1: ENERJİSİ SIKIŞANLAR
    st.markdown("#### 💣 BOMBA HAZIR: ENERJİSİ SIKIŞANLAR")
    if st.session_state.squeeze_active:
        active_df = pd.DataFrame(st.session_state.squeeze_active)
        active_df = active_df.sort_values(by='days', ascending=False)
        st.dataframe(
            active_df,
            column_config={
                "sym": st.column_config.TextColumn("Hisse Kodu"),
                "price": st.column_config.NumberColumn("Son Fiyat", format="%.2f TL"),
                "days": st.column_config.NumberColumn("Sıkışma Gün Sayısı", format="%d Gün"),
                "mom_dir": st.column_config.TextColumn("Momentum Yönü")
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        if start_squeeze_btn:
            st.info("Kriterleri karşılayan sıkışan hisse bulunamadı.")
        else:
            st.info("⚠️ Taramayı başlatmak için lütfen yukarıdaki 'TTM Squeeze Taramasını Başlat' butonuna basın.")
            
    st.markdown("---")
    
    # Tablo 2: PATLAMA BAŞLADI
    st.markdown("#### 🚀 PATLAMA BAŞLADI (TETİK ALDI)")
    if st.session_state.squeeze_breakout:
        breakout_df = pd.DataFrame(st.session_state.squeeze_breakout)
        st.dataframe(
            breakout_df,
            column_config={
                "sym": st.column_config.TextColumn("Hisse Kodu"),
                "price": st.column_config.NumberColumn("Son Fiyat", format="%.2f TL"),
                "mom_dir": st.column_config.TextColumn("Momentum Yönü")
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        if start_squeeze_btn:
            st.info("Bugün sıkışması bitip yukarı yönlü patlama tetikleyen hisse bulunamadı.")


# ==========================================
# 4. YARDIMCI YORUM VE METİN MOTORU
# ==========================================

def generate_dynamic_commentary(symbol: str, last_row, s1_score, s2_score, s2_status, fib_levels, support_resistance) -> str:
    """
    Raporlar için finansal yorum üretir.
    """
    close = last_row['close']
    rsi = last_row['rsi']
    ema20 = last_row['ema_20']
    ema50 = last_row['ema_50']
    
    if close > ema20 > ema50:
        trend_desc = "kısa ve orta vadeli hareketli ortalamaların (EMA20 ve EMA50) üzerinde kalarak pozitif bir yükseliş trendi yapısını sürdürmektedir."
    elif close < ema20 and close < ema50:
        trend_desc = "EMA20 ve EMA50 ortalamalarının altında kalarak kısa vadeli satıcılı bir düzeltme eğilimindedir."
    else:
        trend_desc = "hareketli ortalamaların çevresinde yatay bir sıkışma ve konsolidasyon aşamasında hareket etmektedir."
        
    comment = f"{symbol} hissesi, son kapanış fiyatı olan {close:.2f} TL itibarıyla {trend_desc} "
    comment += f"Momentum tarafında RSI ({rsi:.1f}) {('aşırı alım bölgesine yakın olup güçlü bir iştahı' if rsi > 60 else ('aşırı satım bölgesine yakın bir tepki potansiyelini' if rsi < 35 else 'nötr-dengeli bir duruşu'))} işaret etmektedir. "
    comment += f"Kısa vadeli risk kontrolü için destek seviyeleri yakından izlenmeli, pozitif kırılımlarda direnç hedefleri takip edilmelidir."
    
    return comment
