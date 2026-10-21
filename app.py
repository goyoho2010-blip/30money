from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
WATCHLIST_PATH = APP_DIR / "watchlist.json"
MAX_PRICE = 30_000
LOW_PRICE_CEILING = 10_000

st.set_page_config(page_title="수급 흔적 레이더", page_icon="🔎", layout="wide")


def load_watchlist() -> dict[str, str]:
    if WATCHLIST_PATH.exists():
        try:
            data = json.loads(WATCHLIST_PATH.read_text(encoding="utf-8"))
            return {str(k).zfill(6): str(v) for k, v in data.items()}
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def save_watchlist(data: dict[str, str]) -> None:
    WATCHLIST_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_market_frame(frame: pd.DataFrame, market: str, day: str) -> pd.DataFrame:
    """Normalize one pykrx all-ticker daily frame."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy().reset_index()
    rename = {
        "티커": "ticker", "종목코드": "ticker", "index": "ticker",
        "시가": "open", "고가": "high", "저가": "low", "종가": "close",
        "거래량": "volume", "거래대금": "value", "등락률": "change_pct",
    }
    out = out.rename(columns=rename)
    if "ticker" not in out:
        out = out.rename(columns={out.columns[0]: "ticker"})
    out["ticker"] = out["ticker"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    out["date"] = pd.to_datetime(day)
    out["market"] = market
    for c in ["open", "high", "low", "close", "volume", "value", "change_pct"]:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "value" not in out:
        out["value"] = out["close"] * out["volume"]
    if "change_pct" not in out:
        out["change_pct"] = pd.NA
    return out[["date", "ticker", "market", "open", "high", "low", "close", "volume", "value", "change_pct"]]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_krx_history(days: int, api_key: str) -> tuple[pd.DataFrame, str]:
    """Fetch official KRX market-wide daily trade data. KRX requires an approved API key."""
    rows: list[pd.DataFrame] = []
    now_seoul = datetime.now(ZoneInfo("Asia/Seoul"))
    cursor = now_seoul.date()
    # Before the Korean market close, do not treat today's partial bar as a completed daily candle.
    if (now_seoul.hour, now_seoul.minute) < (15, 45):
        cursor -= timedelta(days=1)
    fetched_sessions = 0
    latest_day = ""
    calendar_days = 0
    with st.spinner("최근 시장 데이터를 가져오는 중입니다. 첫 조회는 시간이 걸릴 수 있습니다."):
        while fetched_sessions < days and calendar_days < 115:
            if cursor.weekday() < 5:
                day_str = cursor.strftime("%Y%m%d")
                got_data = False
                for market, api_id in (("KOSPI", "stk_bydd_trd"), ("KOSDAQ", "ksq_bydd_trd")):
                    try:
                        url = f"https://data-dbg.krx.co.kr/svc/apis/sto/{api_id}"
                        response = requests.get(
                            url,
                            params={"basDd": day_str},
                            headers={"AUTH_KEY": api_key},
                            timeout=20,
                        )
                        if response.status_code in {401, 403}:
                            raise RuntimeError("KRX API 인증키가 유효하지 않거나 해당 API 사용 승인이 완료되지 않았습니다.")
                        response.raise_for_status()
                        payload = response.json()
                        if payload.get("respCode"):
                            if str(payload.get("respCode")) in {"401", "403"}:
                                raise RuntimeError("KRX API 인증키가 승인되지 않았거나 해당 API 사용 승인이 완료되지 않았습니다.")
                            raise RuntimeError(str(payload.get("respMsg", "KRX API 응답 오류")))
                        records = payload.get("OutBlock_1", [])
                        if not records:
                            continue
                        raw = pd.DataFrame(records).rename(columns={
                            "ISU_CD": "ticker", "ISU_NM": "name", "MKT_NM": "market",
                            "TDD_CLSPRC": "close", "TDD_OPNPRC": "open",
                            "TDD_HGPRC": "high", "TDD_LWPRC": "low",
                            "ACC_TRDVOL": "volume", "ACC_TRDVAL": "value",
                            "FLUC_RT": "change_pct",
                        })
                        normalized = raw.copy()
                        normalized["date"] = pd.to_datetime(day_str)
                        normalized["ticker"] = normalized["ticker"].astype(str).str.zfill(6)
                        for c in ["open", "high", "low", "close", "volume", "value", "change_pct"]:
                            normalized[c] = pd.to_numeric(normalized[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
                        normalized["market"] = market
                        normalized = normalized[["date", "ticker", "name", "market", "open", "high", "low", "close", "volume", "value", "change_pct"]]
                        if not normalized.empty:
                            rows.append(normalized)
                            got_data = True
                    except RuntimeError:
                        raise
                    except requests.RequestException as exc:
                        raise RuntimeError(f"KRX API 통신 오류: {exc}") from exc
                    except Exception:
                        continue
                if got_data:
                    fetched_sessions += 1
                    if not latest_day:
                        latest_day = day_str
            cursor -= timedelta(days=1)
            calendar_days += 1
            # Limit request bursts; cached results prevent repeat calls during normal use.
            time.sleep(0.08)
    if not rows:
        raise RuntimeError("시장 데이터를 가져오지 못했습니다. API 키 승인 상태와 네트워크를 확인하거나 CSV 업로드를 이용해 주세요.")
    data = pd.concat(rows, ignore_index=True).drop_duplicates(["date", "ticker", "market"])
    return data, latest_day


def normalize_upload(upload: pd.DataFrame) -> pd.DataFrame:
    """Accept Korean or English OHLCV headings from an uploaded CSV/XLSX."""
    aliases = {
        "날짜": "date", "일자": "date", "Date": "date", "일시": "date",
        "종목코드": "ticker", "티커": "ticker", "코드": "ticker", "종목번호": "ticker",
        "종목명": "name", "Name": "name", "시장": "market",
        "시가": "open", "고가": "high", "저가": "low", "종가": "close",
        "현재가": "close", "거래량": "volume", "거래대금": "value", "등락률": "change_pct",
    }
    df = upload.rename(columns={c: aliases.get(str(c).strip(), str(c).strip().lower()) for c in upload.columns}).copy()
    needed = {"date", "ticker", "open", "high", "low", "close", "volume"}
    if not needed.issubset(df.columns):
        raise ValueError("필수 열은 날짜, 종목코드, 시가, 고가, 저가, 종가, 거래량입니다. 종목명·시장·거래대금은 선택입니다.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    if "market" not in df:
        df["market"] = "업로드"
    if "name" not in df:
        df["name"] = df["ticker"]
    for c in ["open", "high", "low", "close", "volume", "value", "change_pct"]:
        if c in df:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if "value" not in df:
        df["value"] = df["close"] * df["volume"]
    df = df.dropna(subset=["date", "ticker", "close", "volume"]).sort_values(["ticker", "date"])
    return df


def add_signals(df: pd.DataFrame, volume_multiple: float, min_value: int) -> pd.DataFrame:
    """Compute transparent volume/price anomaly flags; not an investor identity detector."""
    df = df.sort_values(["ticker", "date"]).copy()
    df["avg_volume_20"] = df.groupby("ticker")["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    df["volume_ratio"] = df["volume"] / df["avg_volume_20"].replace(0, pd.NA)
    df["ret_5d"] = df.groupby("ticker")["close"].pct_change(5) * 100
    df["ret_20d"] = df.groupby("ticker")["close"].pct_change(20) * 100
    df["ma_5"] = df.groupby("ticker")["close"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    df["ma_20"] = df.groupby("ticker")["close"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    price_direction = df.groupby("ticker")["close"].diff().fillna(0).apply(np.sign)
    df["obv"] = (price_direction * df["volume"]).groupby(df["ticker"]).cumsum()
    df["obv_change_5"] = df.groupby("ticker")["obv"].diff(5)
    df["vol_signal"] = df["volume_ratio"] >= volume_multiple
    df["value_signal"] = df["value"] >= min_value
    df["close_firm"] = df["close"] >= df["open"]
    df["score"] = (
        df["vol_signal"].fillna(False).astype(int) * 2
        + df["value_signal"].fillna(False).astype(int)
        + df["close_firm"].fillna(False).astype(int)
        + (df["ret_5d"].fillna(-999) >= 0).astype(int)
    )
    df["signal"] = df["vol_signal"] & df["value_signal"]
    # Transparent end-of-day review flags. These are review prompts, not sell orders.
    df["below_ma5"] = df["close"] < df["ma_5"]
    df["weak_close_spike"] = (df["volume_ratio"] >= 1.5) & (df["close"] < df["open"])
    df["weak_close_range"] = df["close_position"] <= 0.30
    df["obv_falling_5d"] = df["obv_change_5"] < 0
    review_cols = ["below_ma5", "weak_close_spike", "weak_close_range", "obv_falling_5d"]
    df["review_count"] = df[review_cols].fillna(False).astype(int).sum(axis=1)
    df["review_level"] = np.select(
        [df["review_count"] >= 2, df["review_count"] == 1],
        ["매도 검토 항목 2개 이상", "매도 검토 항목 1개"],
        default="기술 점검 항목 없음",
    )
    return df


def normalize_investor_upload(upload: pd.DataFrame) -> pd.DataFrame:
    """Optional KRX investor-flow CSV; join by ticker and, when available, date."""
    aliases = {
        "날짜": "date", "일자": "date", "기준일자": "date", "date": "date",
        "종목코드": "ticker", "단축코드": "ticker", "티커": "ticker", "ticker": "ticker",
        "기관순매수": "institution_net", "기관계순매수": "institution_net", "기관합계순매수": "institution_net",
        "외국인순매수": "foreign_net", "외국인순매수량": "foreign_net",
        "외국인합계순매수": "foreign_net", "외국인계순매수": "foreign_net",
    }
    frame = upload.rename(columns={c: aliases.get(str(c).strip().replace(" ", ""), str(c).strip().lower()) for c in upload.columns}).copy()
    if "ticker" not in frame or not ({"institution_net", "foreign_net"} & set(frame.columns)):
        raise ValueError("필수 열은 종목코드와 기관순매수 또는 외국인순매수입니다. 날짜 열은 선택입니다.")
    frame["ticker"] = frame["ticker"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    for column in ["institution_net", "foreign_net"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def fmt_int(x) -> str:
    try:
        return f"{int(float(x)):,}"
    except (ValueError, TypeError):
        return "-"


st.title("🔎 수급 흔적 레이더")
st.caption("거래량·거래대금·주가 흐름에서 이상 징후 후보를 찾습니다. 실제 매수 주체나 매집 여부를 확정하는 기능은 아닙니다.")

with st.sidebar:
    st.header("검색 조건")
    source = st.radio("데이터 입력", ["시장 자동 조회", "CSV/XLSX 업로드"], help="자동 조회는 한국거래소 KRX Open API의 승인된 인증키를 사용합니다.")
    lookback = st.slider("차트·검색 기간(거래일)", 30, 90, 60, 5)
    volume_multiple = st.slider("거래량 급증 기준(20일 평균 대비)", 1.5, 5.0, 2.0, 0.5)
    min_value_eok = st.number_input("최소 거래대금(억 원)", min_value=0, max_value=1000, value=10, step=5)
    max_price = st.number_input("기본 검색 상한(원)", min_value=1000, max_value=1_000_000, value=MAX_PRICE, step=1000)
    try:
        saved_api_key = st.secrets.get("KRX_API_KEY", "")
    except Exception:
        saved_api_key = ""
    api_key = st.text_input("KRX API 인증키", value=saved_api_key, type="password", help="키는 이 앱 입력창에서만 사용합니다. Streamlit 배포 시에는 앱 설정의 Secrets에 KRX_API_KEY로 저장할 수 있습니다.") if source == "시장 자동 조회" else ""
    uploaded = None
    if source == "CSV/XLSX 업로드":
        uploaded = st.file_uploader("일봉 데이터 파일", type=["csv", "xlsx"])
    if st.button("🔄 데이터 새로고침", use_container_width=True):
        fetch_krx_history.clear()
        st.rerun()

st.subheader("관심 종목")
watchlist = load_watchlist()
add_col1, add_col2, add_col3 = st.columns([2, 3, 1])
with add_col1:
    new_ticker = st.text_input("종목코드", placeholder="예: 020150", max_chars=6)
with add_col2:
    new_name = st.text_input("종목명(선택)", placeholder="예: 롯데에너지머티리얼즈")
with add_col3:
    st.write("")
    st.write("")
    if st.button("추가", use_container_width=True):
        code = "".join(ch for ch in new_ticker if ch.isdigit()).zfill(6)
        if len(code) == 6 and code != "000000":
            watchlist[code] = new_name.strip() or code
            save_watchlist(watchlist)
            st.success(f"{watchlist[code]}를 관심 종목에 추가했습니다.")
            st.rerun()
        else:
            st.error("종목코드 6자리를 입력해 주세요.")
if watchlist:
    remove = st.multiselect("관심 종목 삭제", [f"{code} · {name}" for code, name in watchlist.items()])
    if remove and st.button("선택 종목 삭제"):
        for label in remove:
            del watchlist[label.split(" · ")[0]]
        save_watchlist(watchlist)
        st.rerun()
else:
    st.info("종목코드를 등록하면 가격 구간 밖의 종목도 함께 추적할 수 있습니다. 관심 종목은 이 프로그램 폴더의 watchlist.json에 저장됩니다.")

try:
    if source == "시장 자동 조회":
        if not api_key:
            st.warning("자동 전 종목 조회에는 KRX Open API 인증키가 필요합니다. KRX Data Marketplace에서 인증키와 ‘유가증권 일별매매정보’, ‘코스닥 일별매매정보’ API 사용 승인을 먼저 받아 주세요. 신청 전에도 CSV/XLSX 업로드 분석은 사용할 수 있습니다.")
            st.stop()
        raw_data, latest_label = fetch_krx_history(lookback, api_key)
        data = raw_data
        st.caption(f"데이터 기준: {latest_label} · KOSPI/KOSDAQ · KRX Open API")
    elif uploaded is None:
        st.info("CSV 또는 XLSX 파일을 올려 주세요. 업로드 파일은 자동 분석용 샘플로 제공됩니다.")
        st.stop()
    else:
        if uploaded.name.lower().endswith(".xlsx"):
            data = normalize_upload(pd.read_excel(uploaded))
        else:
            data = normalize_upload(pd.read_csv(uploaded))
        latest_label = data["date"].max().strftime("%Y-%m-%d")
        st.caption(f"데이터 기준: {latest_label} · 업로드 파일: {uploaded.name}")

    if data.empty:
        st.warning("표시할 데이터가 없습니다.")
        st.stop()

    data = add_signals(data, float(volume_multiple), int(min_value_eok * 100_000_000))
    last_date = data["date"].max()
    latest = data[data["date"] == last_date].copy()
    if "name" not in latest:
        latest["name"] = latest["ticker"].map(watchlist).fillna(latest["ticker"])
    else:
        latest["name"] = latest["name"].fillna(latest["ticker"])
        latest["name"] = latest.apply(lambda r: watchlist.get(r["ticker"], r["name"]), axis=1)
    # Price universe: up to selected ceiling. Watchlist additions are always included.
    held_codes = {p["ticker"] for p in st.session_state.get("eod_positions", [])}
    latest["is_watch"] = latest["ticker"].isin(set(watchlist) | held_codes)
    latest = latest[(latest["close"] <= max_price) | latest["is_watch"]]
    latest["price_band"] = latest["close"].apply(lambda x: "1만 원 이하" if x <= LOW_PRICE_CEILING else "1만~3만 원")
    if "market" not in latest:
        latest["market"] = "업로드"

    tab1, tab2, tab3 = st.tabs(["종목 레이더", "종목별 차트", "장마감 매도 검토"])
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("가격 조건 종목", f"{len(latest):,}개")
        c2.metric("1만 원 이하", f"{(latest['close'] <= LOW_PRICE_CEILING).sum():,}개")
        c3.metric("1만~3만 원", f"{((latest['close'] > LOW_PRICE_CEILING) & (latest['close'] <= max_price)).sum():,}개")
        c4.metric("거래량 이상 후보", f"{int(latest['signal'].sum()):,}개")
        band = st.selectbox("가격 구간", ["전체", "1만 원 이하", "1만~3만 원", "관심 종목"])
        view = latest.copy()
        if band == "1만 원 이하":
            view = view[view["close"] <= LOW_PRICE_CEILING]
        elif band == "1만~3만 원":
            view = view[(view["close"] > LOW_PRICE_CEILING) & (view["close"] <= max_price)]
        elif band == "관심 종목":
            view = view[view["is_watch"]]
        view = view.sort_values(["signal", "score", "volume_ratio"], ascending=[False, False, False])
        cols = ["ticker", "name", "market", "close", "change_pct", "volume", "avg_volume_20", "volume_ratio", "value", "ret_5d", "ret_20d", "score", "signal", "is_watch"]
        labels = {"ticker":"종목코드", "name":"종목명", "market":"시장", "close":"종가", "change_pct":"당일 등락률(%)", "volume":"거래량", "avg_volume_20":"20일 평균 거래량", "volume_ratio":"거래량 배수", "value":"거래대금(원)", "ret_5d":"5일 등락률(%)", "ret_20d":"20일 등락률(%)", "score":"징후 점수(0~5)", "signal":"급증 조건 충족", "is_watch":"관심 종목"}
        table = view[[c for c in cols if c in view]].rename(columns=labels)
        st.dataframe(table, use_container_width=True, hide_index=True, column_config={
            "종가": st.column_config.NumberColumn(format="%d원"),
            "당일 등락률(%)": st.column_config.NumberColumn(format="%.2f%%"),
            "거래량 배수": st.column_config.NumberColumn(format="%.2fx"),
            "5일 등락률(%)": st.column_config.NumberColumn(format="%.2f%%"),
            "20일 등락률(%)": st.column_config.NumberColumn(format="%.2f%%"),
            "거래대금(원)": st.column_config.NumberColumn(format="%d"),
            "급증 조건 충족": st.column_config.CheckboxColumn(),
            "관심 종목": st.column_config.CheckboxColumn(),
        })
        st.download_button("현재 목록 CSV 저장", table.to_csv(index=False).encode("utf-8-sig"), "수급흔적레이더_검색결과.csv", "text/csv")
        with st.expander("점수와 표시 기준"):
            st.markdown(f"""
            - **거래량 급증:** 당일 거래량이 직전 20거래일 평균의 **{volume_multiple:.1f}배 이상**
            - **급증 조건 충족 표시:** 거래량 급증과 최소 거래대금 **{min_value_eok}억 원**을 함께 충족
            - **징후 점수(0~5):** 거래량 배수, 거래대금, 종가가 시가 이상인지, 최근 5일 수익률을 합산한 참고 점수
            - 이 점수는 매수 주체를 판별하지 않으며, 투자 판단이나 수익을 보장하지 않습니다.
            """)
    with tab2:
        if latest.empty:
            st.info("조건에 맞는 종목이 없습니다.")
        else:
            options = latest.sort_values("name").apply(lambda r: f"{r['ticker']} · {r['name']}", axis=1).tolist()
            selected = st.selectbox("종목 선택", options)
            code = selected.split(" · ")[0]
            history = data[data["ticker"] == code].sort_values("date").tail(lookback).copy()
            if history.empty:
                st.info("차트 데이터가 없습니다.")
            else:
                name = selected.split(" · ", 1)[1]
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=history["date"], open=history["open"], high=history["high"], low=history["low"], close=history["close"], name="주가"))
                flagged = history[history["vol_signal"]]
                if not flagged.empty:
                    fig.add_trace(go.Scatter(x=flagged["date"], y=flagged["high"] * 1.025, mode="markers", name="거래량 급증", marker=dict(symbol="diamond", size=11, color="#7c3aed"), hovertemplate="%{x|%Y-%m-%d}<br>거래량 급증<extra></extra>"))
                fig.update_layout(title=f"{name} ({code}) · 최근 {lookback}거래일", height=520, xaxis_rangeslider_visible=False, yaxis_title="주가(원)", legend_orientation="h")
                st.plotly_chart(fig, use_container_width=True)
                history_view = history[["date", "close", "volume", "avg_volume_20", "volume_ratio", "value", "score", "vol_signal"]].rename(columns={"date":"날짜", "close":"종가", "volume":"거래량", "avg_volume_20":"20일 평균 거래량", "volume_ratio":"거래량 배수", "value":"거래대금(원)", "score":"징후 점수", "vol_signal":"거래량 급증"})
                st.dataframe(history_view.sort_values("날짜", ascending=False), use_container_width=True, hide_index=True)
    with tab3:
        st.subheader("장 마감 후 보유 종목 점검")
        st.caption("종가 기준 데이터를 모아 점검 항목을 보여드립니다. 매도 여부와 주문은 천명 선생님이 직접 판단합니다.")

        positions_key = "eod_positions"
        if positions_key not in st.session_state:
            st.session_state[positions_key] = []
        with st.expander("보유 종목 입력 / 수정", expanded=not st.session_state[positions_key]):
            with st.form("add_eod_position", clear_on_submit=True):
                p1, p2, p3, p4 = st.columns([1.5, 2.2, 1.3, 1.5])
                position_code = p1.text_input("종목코드", max_chars=6, placeholder="예: 020150")
                position_name = p2.text_input("종목명", placeholder="예: 롯데에너지머티리얼즈")
                position_qty = p3.number_input("보유 수량", min_value=0, step=1)
                position_cost = p4.number_input("평균 매입가(원)", min_value=0, step=100)
                submitted = st.form_submit_button("보유 종목 추가")
            if submitted:
                clean_code = "".join(ch for ch in position_code if ch.isdigit()).zfill(6)
                if len(clean_code) == 6 and clean_code != "000000" and position_qty > 0 and position_cost > 0:
                    resolved_name = position_name.strip() or watchlist.get(clean_code, clean_code)
                    updated = [p for p in st.session_state[positions_key] if p["ticker"] != clean_code]
                    updated.append({"ticker": clean_code, "name": resolved_name, "quantity": int(position_qty), "avg_cost": int(position_cost)})
                    st.session_state[positions_key] = updated
                    st.success(f"{resolved_name} 보유 정보를 반영했습니다.")
                    st.rerun()
                else:
                    st.error("종목코드 6자리, 보유 수량, 평균 매입가를 확인해 주세요.")
            if st.session_state[positions_key]:
                position_options = [f"{p['ticker']} · {p['name']}" for p in st.session_state[positions_key]]
                delete_positions = st.multiselect("보유 목록에서 삭제", position_options, key="delete_eod_positions")
                if delete_positions and st.button("선택한 보유 종목 삭제", key="delete_eod_positions_button"):
                    remove_codes = {x.split(" · ")[0] for x in delete_positions}
                    st.session_state[positions_key] = [p for p in st.session_state[positions_key] if p["ticker"] not in remove_codes]
                    st.rerun()

        # Optional investor-flow file: this is kept separate because the approved daily-trade APIs
        # used for OHLCV do not provide investor-by-investor net buying fields.
        flow_file = st.file_uploader(
            "기관·외국인 순매수 자료 (선택, CSV/XLSX)",
            type=["csv", "xlsx"],
            key="eod_investor_flow_upload",
            help="종목코드, 기관순매수, 외국인순매수 열이 필요합니다. 날짜 열을 넣으면 기준일도 확인합니다.",
        )
        flow_data = pd.DataFrame()
        if flow_file is not None:
            try:
                flow_raw = pd.read_excel(flow_file) if flow_file.name.lower().endswith(".xlsx") else pd.read_csv(flow_file)
                flow_data = normalize_investor_upload(flow_raw)
                if "date" in flow_data and flow_data["date"].notna().any():
                    flow_data = flow_data[flow_data["date"].dt.normalize() == pd.Timestamp(last_date).normalize()]
                if flow_data.empty:
                    st.warning("업로드 자료에서 기준일 데이터가 없습니다. 날짜 열이 있다면 최신 종가일과 같은 날짜인지 확인해 주세요.")
            except Exception as flow_exc:
                st.error(f"수급 자료를 읽지 못했습니다: {flow_exc}")
                flow_data = pd.DataFrame()

        # Review both held names and the existing watchlist. Other candidates remain available
        # through the all-market table and can be added to the watchlist.
        review_codes = {p["ticker"] for p in st.session_state[positions_key]} | set(watchlist)
        review = latest[latest["ticker"].isin(review_codes)].copy()
        held_by_code = {p["ticker"]: p for p in st.session_state[positions_key]}
        if not review.empty:
            for column in ["institution_net", "foreign_net"]:
                review[column] = np.nan
            if not flow_data.empty:
                for _, flow_row in flow_data.iterrows():
                    mask = review["ticker"] == flow_row["ticker"]
                    for column in ["institution_net", "foreign_net"]:
                        if column in flow_data and mask.any():
                            review.loc[mask, column] = flow_row[column]
            review["매입단가"] = review["ticker"].map({code: p["avg_cost"] for code, p in held_by_code.items()})
            review["보유수량"] = review["ticker"].map({code: p["quantity"] for code, p in held_by_code.items()})
            review["평가손익"] = (review["close"] - review["매입단가"]) * review["보유수량"]
            review["수익률(%)"] = (review["close"] / review["매입단가"] - 1) * 100
            review["장마감 점검 항목"] = review["review_level"]
            review["기관 순매수"] = review["institution_net"]
            review["외국인 순매수"] = review["foreign_net"]
            review["내 판단"] = review["ticker"].map(st.session_state.get("eod_decisions", {})).fillna("미판단")
            review["메모"] = review["ticker"].map(st.session_state.get("eod_notes", {})).fillna("")
            view_columns = ["ticker", "name", "close", "change_pct", "volume_ratio", "value", "ma_5", "close_position", "obv_change_5", "institution_net", "foreign_net", "매입단가", "보유수량", "평가손익", "수익률(%)", "review_count", "review_level", "내 판단", "메모"]
            table = review[[c for c in view_columns if c in review]].rename(columns={
                "ticker":"종목코드", "name":"종목명", "close":"종가", "change_pct":"당일 등락률(%)",
                "volume_ratio":"20일 평균 대비 거래량", "value":"거래대금(원)", "ma_5":"5일 평균 종가",
                "close_position":"고가-저가 내 종가 위치(0~1)", "obv_change_5":"OBV 5일 변화",
                "institution_net":"기관 순매수", "foreign_net":"외국인 순매수",
                "review_count":"점검 항목 수", "review_level":"기술 점검 결과",
            })
            edited = st.data_editor(
                table,
                use_container_width=True,
                hide_index=True,
                disabled=[c for c in table.columns if c not in ["내 판단", "메모"]],
                column_config={
                    "종가": st.column_config.NumberColumn(format="%d원"),
                    "당일 등락률(%)": st.column_config.NumberColumn(format="%.2f%%"),
                    "20일 평균 대비 거래량": st.column_config.NumberColumn(format="%.2fx"),
                    "거래대금(원)": st.column_config.NumberColumn(format="%d"),
                    "5일 평균 종가": st.column_config.NumberColumn(format="%d원"),
                    "고가-저가 내 종가 위치(0~1)": st.column_config.NumberColumn(format="%.2f"),
                    "평가손익": st.column_config.NumberColumn(format="%d원"),
                    "수익률(%)": st.column_config.NumberColumn(format="%.2f%%"),
                    "내 판단": st.column_config.SelectboxColumn(options=["미판단", "보유", "일부 매도 검토", "매도 검토", "추가 관찰"]),
                    "메모": st.column_config.TextColumn(),
                },
                key="eod_review_editor",
            )
            st.session_state["eod_decisions"] = dict(zip(review["ticker"], edited["내 판단"]))
            st.session_state["eod_notes"] = dict(zip(review["ticker"], edited["메모"]))
            st.download_button("장마감 점검표 CSV 저장", edited.to_csv(index=False).encode("utf-8-sig"), "장마감_매도검토.csv", "text/csv", key="download_eod_review")
            st.markdown("**기술 점검 기준**")
            st.caption("5일 평균 종가 아래 · 거래량 1.5배 이상이면서 음봉 · 당일 범위의 하단 30%에서 마감 · OBV 5일 변화 하락. 해당 항목 개수만 표시하며 매도 신호나 주문 권유가 아닙니다.")
            st.markdown("**참고 지표**")
            st.caption("OBV는 종가와 거래량으로 산출한 누적 지표입니다. CVD는 체결별 매수·매도 주도 데이터가 필요해 일봉 API만으로 정확히 계산하지 않습니다. 기관·외국인 순매수는 승인된 현재 일봉 API에서 제공되지 않아 별도 자료 업로드가 있을 때만 표시합니다.")
        else:
            st.info("장마감 점검 대상이 없습니다. 관심 종목을 추가하거나 보유 종목을 입력해 주세요.")
except Exception as exc:
    st.error(f"데이터 처리 중 문제가 생겼습니다: {exc}")
    st.info("자동 조회가 막히면 CSV/XLSX 업로드 탭으로 분석할 수 있습니다.")

st.divider()
st.caption("자료 기준: 일봉 종가. 자동 조회는 KOSPI·KOSDAQ의 최근 거래 데이터를 이용합니다. 시장 데이터 제공처의 응답 제한이나 변경에 따라 자동 조회가 일시적으로 실패할 수 있습니다.")
