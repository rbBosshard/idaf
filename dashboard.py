from typing import Optional, Tuple, Dict, List, Any

import pandas as pd
import streamlit as st
import re

from dashboard_helper import get_data_from_excel

# Optional: Altair für flexible Diagramme
try:
    import altair as alt
    ALT_AVAILABLE = True
except Exception:
    ALT_AVAILABLE = False

# ---------- Feste Spaltennamen aus der Excel-Vorlage ----------
COL_DEVICE = "Gerät / Aktivität"
COL_KWH = "Täglicher Verbrauch [kWh]"
COL_COST = "Tägliche Kosten [CHF]"
COL_EMISSIONS = "Tägliche CO₂-Emissionen [g CO₂eq]"


def _normalize_label(val: Any) -> str:
    """Normalisiert Label-Texte: trimmt, ersetzt NBSP mit Space, komprimiert Whitespace, lower-case.
    Dient zum robusten Abgleich von Zeilenlabels (z. B. 'Gerät / Aktivität').
    """
    if pd.isna(val):
        return ""
    s = str(val)
    # ersetze NBSP und andere Whitespace zu Space
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_metric_table(df: pd.DataFrame) -> bool:
    """Erkennt die Vorlagenform, bei der die erste Spalte Zeilenlabels (z. B. 'Nr.') enthält
    und u. a. eine Zeile 'Gerät / Aktivität' die Gerätespalten-Überschriften liefert.
    """
    if df.empty:
        return False
    label_col = df.columns[0]
    col_values = df[label_col].apply(_normalize_label)
    return col_values.eq(_normalize_label(COL_DEVICE)).any()


def _from_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    """Wandelt die Vorlagen-Tabelle (Zeilenlabels in erster Spalte) in eine flache Geräte-Tabelle um.
    Erwartet folgende Zeilenlabels: COL_DEVICE, COL_KWH, COL_COST (optional), COL_EMISSIONS (optional).
    Rückgabe hat die Spalten [COL_DEVICE, COL_KWH, (COL_COST), (COL_EMISSIONS)].
    """
    df0 = df.copy()
    label_col = df0.columns[0]
    # Index auf Zeilenlabel setzen für leichten Zugriff
    df0 = df0.set_index(label_col)
    # Geräte-Bezeichnungen stehen in der Zeile 'Gerät / Aktivität' (als Spaltenwerte)
    if _normalize_label(COL_DEVICE) not in map(_normalize_label, df0.index):
        # Index normalisieren
        df0.index = df0.index.map(_normalize_label)
    if _normalize_label(COL_DEVICE) not in df0.index:
        raise KeyError(f"Zeile '{COL_DEVICE}' nicht gefunden")

    devices_row = df0.loc[_normalize_label(COL_DEVICE)]
    # Erlaubte Gerätespalten anhand der Werte in der Gerätezeile bestimmen:
    # nur echte Strings != (leer|nan|total|tottal|summe|gesamt)
    device_cols: List[str] = []
    device_names: List[str] = []

    def _is_total_column_name(name: Any) -> bool:
        n = _normalize_label(name).lower()
        return any(t in n for t in ["total", "summe", "gesamt"])

    for idx, (col, val) in enumerate(devices_row.items()):
        # Überspringe Labelspalte
        if col == label_col:
            continue
        # Überspringe Spalten, deren Name selbst auf Total/Summe hindeutet
        if _is_total_column_name(col):
            continue
        val_norm = _normalize_label(val)
        # Gerätebezeichnung muss Text mit Buchstaben enthalten
        if not val_norm or not re.search(r"[A-Za-zÄÖÜäöü]", val_norm):
            continue
        if val_norm.lower() in {"total", "tottal", "summe", "gesamt"}:
            continue
        device_cols.append(col)
        device_names.append(str(val).strip())

    # Hilfsfunktion: hole Wertezeile robust (mit/ohne Space vor [)
    def get_row(label: str) -> Optional[pd.Series]:
        # direkter Treffer
        if label in df0.index:
            return df0.loc[label]
        # Normalisiert suchen (z. B. NBSP vs Space)
        label_norm = _normalize_label(label)
        if label_norm in df0.index:
            return df0.loc[label_norm]
        # Variante ohne/mit Space vor [ suchen
        alt = label.replace(" [", "[") if " [" in label else label.replace("[", " [")
        alt_norm = _normalize_label(alt)
        if alt_norm in df0.index:
            return df0.loc[alt_norm]
        return None

    data: Dict[str, List[Any]] = {COL_DEVICE: device_names}
    # Verbrauch
    kwh_row = get_row(COL_KWH)
    if kwh_row is not None:
        data[COL_KWH] = pd.to_numeric(kwh_row[device_cols], errors="coerce").tolist()
    else:
        data[COL_KWH] = [pd.NA] * len(device_cols)
    # Kosten optional
    cost_row = get_row(COL_COST)
    if cost_row is not None:
        data[COL_COST] = pd.to_numeric(cost_row[device_cols], errors="coerce").tolist()
    # Emissionen optional
    em_row = get_row(COL_EMISSIONS)
    if em_row is not None:
        data[COL_EMISSIONS] = pd.to_numeric(em_row[device_cols], errors="coerce").tolist()

    out = pd.DataFrame(data)
    return out


_TS_LABEL_PATTERN = re.compile(r"^Verbrauch\s+(\d{2}\.\d{2}\.\d{4}),\s*(\d{2}:\d{2})\s*\[kWh\]$")


def _extract_time_series(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Extrahiert Zeitreihenzeilen der Form 'Verbrauch dd.mm.yyyy, HH:MM [kWh]' und gibt
    ein Long-DataFrame mit Spalten ['timestamp', COL_DEVICE, 'kWh'] zurück.
    Erwartet die gleiche Vorlagenstruktur (erste Spalte: Zeilenlabels, Geräte in der Zeile COL_DEVICE).
    """
    if df.empty:
        return None
    label_col = df.columns[0]
    df0 = df.copy()
    df0[label_col] = df0[label_col].map(_normalize_label)
    # Gerätezeile finden
    mask_dev = df0[label_col].eq(_normalize_label(COL_DEVICE))
    if not mask_dev.any():
        return None
    devices_row = df0.loc[mask_dev].iloc[0]
    # Gerätespalten bestimmen (wie in _from_metric_table)
    device_cols: List[str] = []
    device_names: List[str] = []

    def _is_total_column_name(name: Any) -> bool:
        n = _normalize_label(name).lower()
        return any(t in n for t in ["total", "tottal", "summe", "gesamt"])

    for col, val in devices_row.items():
        if col == label_col:
            continue
        if _is_total_column_name(col):
            continue
        val_norm = _normalize_label(val)
        if not val_norm or not re.search(r"[A-Za-zÄÖÜäöü]", val_norm):
            continue
        if val_norm.lower() in {"total", "tottal", "summe", "gesamt"}:
            continue
        device_cols.append(col)
        device_names.append(str(val).strip())

    # Mapping von Spaltennamen zu Gerätenamen
    col2device = dict(zip(device_cols, device_names))

    # Zeitreihenzeilen selektieren
    ts_rows = df0[label_col].apply(lambda s: bool(_TS_LABEL_PATTERN.match(str(s))))
    if not ts_rows.any():
        return None

    records: List[Dict[str, Any]] = []
    for _, row in df0.loc[ts_rows].iterrows():
        m = _TS_LABEL_PATTERN.match(str(row[label_col]))
        if not m:
            continue
        date_s, time_s = m.group(1), m.group(2)
        ts = pd.to_datetime(f"{date_s} {time_s}", dayfirst=True, format="%d.%m.%Y %H:%M", errors="coerce")
        if pd.isna(ts):
            continue
        for col in device_cols:
            val = pd.to_numeric(row.get(col), errors="coerce")
            if pd.isna(val):
                continue
            records.append({
                "timestamp": ts,
                COL_DEVICE: col2device[col],
                "kWh": float(val),
            })

    if not records:
        return None
    ts_df = pd.DataFrame.from_records(records)
    return ts_df

def _sidebar_controls(df: pd.DataFrame, has_ts: bool) -> Tuple[pd.DataFrame, Dict]:
    st.sidebar.markdown("### Filter & Optionen")

    # Gerätefilter
    devices = sorted(df[COL_DEVICE].dropna().unique().tolist())
    chosen = st.sidebar.multiselect("Geräte", options=devices, default=devices)
    if chosen:
        df = df[df[COL_DEVICE].isin(chosen)]

    # Szenario: Sparmassnahmen
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Szenario")
    savings_devices = st.sidebar.multiselect("Sparmassnahmen für Geräte", options=chosen or devices, help="Reduktion des Verbrauchs für ausgewählte Geräte")
    reduction_pct = st.sidebar.slider("Reduktion (%)", 0, 100, 0, step=5)

    # Kosten & CO2: Berechnung
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Kosten & Emissionen")
    price_mode = st.sidebar.radio("Kostenquelle", ["Aus Datei (falls vorhanden)", "Berechne mit Tarif"], index=1)
    price_per_kwh = st.sidebar.number_input("Tarif (CHF/kWh)", min_value=0.0, value=0.27, step=0.01, format="%.3f")
    co2_mode = st.sidebar.radio("CO₂-Quelle", ["Aus Datei (falls vorhanden)", "Berechne mit Intensität"], index=1)
    co2_intensity = st.sidebar.number_input("CO₂-Intensität (g/kWh)", min_value=0.0, value=120.0, step=5.0, format="%.1f")

    # Zeitvariabler Tarif/CO2 (falls Zeitreihe vorhanden)
    tou_enabled = False
    peak_start = None
    peak_end = None
    peak_mult = 1.0
    co2_peak_mult = 1.0
    if has_ts:
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### Zeitvariabler Tarif (bei Zeitreihe)")
        tou_enabled = st.sidebar.checkbox("Zeitvariable Preise/CO₂ aktivieren", value=False)
        if tou_enabled:
            peak_start = st.sidebar.time_input("Spitze beginnt", value=pd.Timestamp("1970-01-01 07:00").time())
            peak_end = st.sidebar.time_input("Spitze endet", value=pd.Timestamp("1970-01-01 10:00").time())
            peak_mult = st.sidebar.number_input("Preisfaktor Spitze", min_value=0.1, value=1.5, step=0.1, format="%.1f")
            co2_peak_mult = st.sidebar.number_input("CO₂-Faktor Spitze", min_value=0.1, value=1.2, step=0.1, format="%.1f")

    # Aggregationsebene (hier nur Summe vorgesehen)
    period = "Summe"
    return df, {
        "devices": chosen,
        "period": period,
        "savings_devices": savings_devices,
        "reduction_pct": reduction_pct,
        "price_mode": price_mode,
        "price_per_kwh": float(price_per_kwh),
        "co2_mode": co2_mode,
        "co2_intensity": float(co2_intensity),
        "tou_enabled": tou_enabled,
        "peak_start": peak_start,
        "peak_end": peak_end,
        "peak_mult": float(peak_mult),
        "co2_peak_mult": float(co2_peak_mult),
    }


def _aggregate(df: pd.DataFrame, period: str) -> pd.DataFrame:
    df = df.copy()
    if period == "Summe":
        df["bucket"] = "Gesamt"

    # Emissionen (g) -> kg umrechnen, falls vorhanden
    if COL_EMISSIONS in df.columns:
        df["emissions_kg"] = pd.to_numeric(df[COL_EMISSIONS], errors="coerce") / 1000.0

    # Sicherstellen, dass numerische Spalten numerisch sind
    if COL_KWH in df.columns:
        df[COL_KWH] = pd.to_numeric(df[COL_KWH], errors="coerce")
    if COL_COST in df.columns:
        df[COL_COST] = pd.to_numeric(df[COL_COST], errors="coerce")

    # Einfache Aggregation über vorhandene Spalten und anschließendes Umbenennen
    agg_dict: Dict[str, str] = {}
    if COL_KWH in df.columns:
        agg_dict[COL_KWH] = "sum"
    if COL_COST in df.columns:
        agg_dict[COL_COST] = "sum"
    if "emissions_kg" in df.columns:
        agg_dict["emissions_kg"] = "sum"

    if not agg_dict:
        # Keine aggregierbaren Spalten vorhanden -> leeres Ergebnis mit Schlüsselfeldern zurückgeben
        return df[["bucket", COL_DEVICE]].drop_duplicates().assign(**{"energy_kwh": pd.NA})

    agg = (
        df.groupby(["bucket", COL_DEVICE], dropna=False)
          .agg(agg_dict)
          .reset_index()
    )

    rename_map = {}
    if COL_KWH in agg.columns:
        rename_map[COL_KWH] = "energy_kwh"
    if COL_COST in agg.columns:
        rename_map[COL_COST] = "cost"
    # emissions_kg bleibt gleich
    agg = agg.rename(columns=rename_map)
    return agg


def _apply_reduction(df: pd.DataFrame, ts_df: Optional[pd.DataFrame], devices: List[str], reduction_pct: int) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Wendet eine prozentuale Reduktion auf ausgewählte Geräte an (sowohl auf df[COL_KWH] als auch ts_df['kWh'])."""
    if not devices or reduction_pct <= 0:
        return df.copy(), ts_df.copy() if ts_df is not None else None
    factor = max(0.0, 1.0 - (reduction_pct / 100.0))
    df2 = df.copy()
    mask = df2[COL_DEVICE].isin(devices)
    if COL_KWH in df2.columns:
        df2.loc[mask, COL_KWH] = pd.to_numeric(df2.loc[mask, COL_KWH], errors="coerce") * factor
    ts2 = None
    if ts_df is not None:
        ts2 = ts_df.copy()
        ts2.loc[ts2[COL_DEVICE].isin(devices), "kWh"] = pd.to_numeric(ts2.loc[ts2[COL_DEVICE].isin(devices), "kWh"], errors="coerce") * factor
    return df2, ts2


def _is_peak_time(ts: pd.Timestamp, start: Any, end: Any) -> bool:
    """Prüft, ob Zeitstempel innerhalb des Spitzenzeitfensters liegt. Unterstützt auch über Mitternacht."""
    if start is None or end is None:
        return False
    t = ts.time()
    if start <= end:
        return start <= t < end
    # über Mitternacht
    return t >= start or t < end


def _compute_cost_emissions(
    df: pd.DataFrame,
    ts_df: Optional[pd.DataFrame],
    *,
    price_mode: str,
    price_per_kwh: float,
    co2_mode: str,
    co2_intensity: float,
    tou_enabled: bool,
    peak_start: Any,
    peak_end: Any,
    peak_mult: float,
    co2_peak_mult: float,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Erweitert df um Kosten/Emissionen gemäß gewählter Logik. Nutzt Zeitreihe für ToU, falls vorhanden."""
    df2 = df.copy()
    ts2 = ts_df.copy() if ts_df is not None else None

    # Sicherstellen, dass kWh numerisch ist
    if COL_KWH in df2.columns:
        df2[COL_KWH] = pd.to_numeric(df2[COL_KWH], errors="coerce")

    # Kosten
    use_file_cost = (price_mode == "Aus Datei (falls vorhanden)") and (COL_COST in df2.columns)
    if not use_file_cost:
        if ts2 is not None and tou_enabled:
            ts_tmp = ts2.copy()
            ts_tmp["_price"] = price_per_kwh
            ts_tmp.loc[ts_tmp["timestamp"].apply(lambda x: _is_peak_time(x, peak_start, peak_end)), "_price"] *= peak_mult
            ts_tmp["_cost"] = pd.to_numeric(ts_tmp["kWh"], errors="coerce") * ts_tmp["_price"]
            by_dev = ts_tmp.groupby(COL_DEVICE)["_cost"].sum()
            df2[COL_COST] = df2[COL_DEVICE].map(by_dev).fillna(0.0)
            ts2 = ts_tmp
        else:
            df2[COL_COST] = df2[COL_KWH] * price_per_kwh

    # Emissionen (Datei liefert g; unsere Berechnung auch in g)
    use_file_em = (co2_mode == "Aus Datei (falls vorhanden)") and (COL_EMISSIONS in df2.columns)
    if not use_file_em:
        if ts2 is not None and tou_enabled:
            ts_tmp = ts2.copy()
            ts_tmp["_intensity_g_per_kwh"] = co2_intensity
            ts_tmp.loc[ts_tmp["timestamp"].apply(lambda x: _is_peak_time(x, peak_start, peak_end)), "_intensity_g_per_kwh"] *= co2_peak_mult
            ts_tmp["_emissions_g"] = pd.to_numeric(ts_tmp["kWh"], errors="coerce") * ts_tmp["_intensity_g_per_kwh"]
            by_dev = ts_tmp.groupby(COL_DEVICE)["_emissions_g"].sum()
            df2[COL_EMISSIONS] = df2[COL_DEVICE].map(by_dev).fillna(0.0)
            ts2 = ts_tmp
        else:
            df2[COL_EMISSIONS] = df2[COL_KWH] * co2_intensity

    return df2, ts2


def _kpis_overview(df: pd.DataFrame):
    st.subheader("Übersicht")
    total_energy = pd.to_numeric(df[COL_KWH], errors="coerce").sum()
    total_cost = pd.to_numeric(df.get(COL_COST), errors="coerce").sum() if COL_COST in df.columns else float("nan")
    # Emissionen in kg darstellen (Vorlage liefert g)
    total_em = (pd.to_numeric(df.get(COL_EMISSIONS), errors="coerce").sum() / 1000.0) if COL_EMISSIONS in df.columns else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Gesamtverbrauch (kWh)", f"{total_energy:,.0f}".replace(",", " "))
    with c2:
        if pd.notna(total_cost):
            st.metric("Gesamtkosten", f"{total_cost:,.2f}".replace(",", " "))
        else:
            st.metric("Gesamtkosten", "–")
    with c3:
        if pd.notna(total_em):
            st.metric("Emissionen gesamt (kg CO₂e)", f"{total_em:,.1f}".replace(",", " "))
        else:
            st.metric("Emissionen gesamt", "–")
    with c4:
        if pd.notna(total_cost) and total_energy > 0:
            st.metric("Durchschnittspreis (pro kWh)", f"{(total_cost/total_energy):.4f}")
        else:
            st.metric("Durchschnittspreis (pro kWh)", "–")

    # Hinweis: Kein Datum in der Vorlage -> kein Peak-Tag


def _energy_charts(agg: pd.DataFrame, period: str):
    st.subheader("Energieverbrauch")
    # Zeitreihen nur, wenn mehrere Zeit-Buckets vorhanden und temporaler Typ
    pivot = agg.pivot_table(index="bucket", columns=COL_DEVICE, values="energy_kwh", aggfunc="sum").fillna(0)
    is_temporal = pd.api.types.is_datetime64_any_dtype(pivot.index)
    has_many_buckets = pivot.shape[0] > 1
    if is_temporal and has_many_buckets:
        if ALT_AVAILABLE:
            long = pivot.reset_index().melt("bucket", var_name="device", value_name="kWh")
            chart = alt.Chart(long).mark_area().encode(
                x=alt.X("bucket:T", title=period),
                y=alt.Y("kWh:Q", title="kWh"),
                color=alt.Color("device:N", legend=alt.Legend(title="Gerät"))
            ).properties(height=400)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.area_chart(pivot)
    else:
        # Fallback: Balken nach Gerät (Gesamtsumme)
        dev = agg.groupby(COL_DEVICE, as_index=False)["energy_kwh"].sum().sort_values("energy_kwh", ascending=False)
        if ALT_AVAILABLE:
            chart2 = alt.Chart(dev).mark_bar().encode(
                x=alt.X(f"{COL_DEVICE}:N", sort='-y', title="Gerät"), y=alt.Y("energy_kwh:Q", title="kWh")
            )
            st.altair_chart(chart2, use_container_width=True)
        else:
            st.bar_chart(dev.set_index(COL_DEVICE)["energy_kwh"])

    # Verbrauch nach Gerät (Top-N)
    dev = agg.groupby(COL_DEVICE, as_index=False)["energy_kwh"].sum().sort_values("energy_kwh", ascending=False)
    st.markdown("Top-Geräte nach Verbrauch")
    if ALT_AVAILABLE:
        chart2 = alt.Chart(dev).mark_bar().encode(x=alt.X(f"{COL_DEVICE}:N", sort='-y', title="Gerät"), y=alt.Y("energy_kwh:Q", title="kWh"))
        st.altair_chart(chart2, use_container_width=True)
    else:
        st.bar_chart(dev.set_index(COL_DEVICE)["energy_kwh"])


def _emission_charts(agg: pd.DataFrame, period: str):
    st.subheader("Emissionen")
    if "emissions_kg" not in agg.columns or agg["emissions_kg"].isna().all():
        st.info("Keine Emissionsdaten verfügbar (Spalte 'Emissionen' oder Emissionsfaktor/Tarif fehlt).")
        return
    pivot = agg.pivot_table(index="bucket", columns=COL_DEVICE, values="emissions_kg", aggfunc="sum").fillna(0)
    is_temporal = pd.api.types.is_datetime64_any_dtype(pivot.index)
    has_many_buckets = pivot.shape[0] > 1
    if is_temporal and has_many_buckets:
        if ALT_AVAILABLE:
            long = pivot.reset_index().melt("bucket", var_name="device", value_name="kgCO2e")
            chart = alt.Chart(long).mark_area().encode(
                x=alt.X("bucket:T", title=period),
                y=alt.Y("kgCO2e:Q", title="kg CO₂e"),
                color=alt.Color("device:N", legend=alt.Legend(title="Gerät"))
            ).properties(height=400)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.area_chart(pivot)
    # Balken nach Gerät (immer anzeigen)
    by_dev = agg.groupby(COL_DEVICE, as_index=False)["emissions_kg"].sum().sort_values("emissions_kg", ascending=False)
    st.markdown("Emissionen nach Gerät")
    if ALT_AVAILABLE:
        chart2 = alt.Chart(by_dev).mark_bar().encode(
            x=alt.X(f"{COL_DEVICE}:N", sort='-y', title="Gerät"), y=alt.Y("emissions_kg:Q", title="kg CO₂e")
        )
        st.altair_chart(chart2, use_container_width=True)
    else:
        st.bar_chart(by_dev.set_index(COL_DEVICE)["emissions_kg"])


def _cost_charts(agg: pd.DataFrame, period: str):
    st.subheader("Kosten")
    if "cost" not in agg.columns or agg["cost"].isna().all():
        st.info("Keine Kostendaten verfügbar (Spalte 'Kosten' oder Preis pro kWh fehlt).")
        return
    pivot = agg.pivot_table(index="bucket", columns=COL_DEVICE, values="cost", aggfunc="sum").fillna(0)
    is_temporal = pd.api.types.is_datetime64_any_dtype(pivot.index)
    has_many_buckets = pivot.shape[0] > 1
    if is_temporal and has_many_buckets:
        if ALT_AVAILABLE:
            long = pivot.reset_index().melt("bucket", var_name="device", value_name="Kosten")
            chart = alt.Chart(long).mark_area().encode(
                x=alt.X("bucket:T", title=period),
                y=alt.Y("Kosten:Q", title="Kosten"),
                color=alt.Color("device:N", legend=alt.Legend(title="Gerät"))
            ).properties(height=400)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.area_chart(pivot)

    by_dev = agg.groupby(COL_DEVICE, as_index=False)["cost"].sum().sort_values("cost", ascending=False)
    st.markdown("Kosten nach Gerät")
    if ALT_AVAILABLE:
        chart2 = alt.Chart(by_dev).mark_bar().encode(x=alt.X(f"{COL_DEVICE}:N", sort='-y', title="Gerät"), y=alt.Y("cost:Q", title="Kosten"))
        st.altair_chart(chart2, use_container_width=True)
    else:
        st.bar_chart(by_dev.set_index(COL_DEVICE)["cost"])


def _share_charts(agg: pd.DataFrame):
    """Zeigt Anteils- und Pareto-Darstellung für Energieverbrauch."""
    if "energy_kwh" not in agg.columns:
        return
    by_dev = agg.groupby(COL_DEVICE, as_index=False)["energy_kwh"].sum().sort_values("energy_kwh", ascending=False)
    total = by_dev["energy_kwh"].sum()
    if total <= 0:
        return
    by_dev["share"] = by_dev["energy_kwh"] / total
    by_dev["cum_share"] = by_dev["share"].cumsum()
    st.markdown("Anteile & Pareto")
    if ALT_AVAILABLE:
        import altair as alt  # local import for type checkers
        bar = alt.Chart(by_dev).mark_bar().encode(
            x=alt.X(f"{COL_DEVICE}:N", sort='-y', title="Gerät"),
            y=alt.Y("energy_kwh:Q", title="kWh")
        )
        line = alt.Chart(by_dev).mark_line(color="#d62728").encode(
            x=alt.X(f"{COL_DEVICE}:N", sort='-y', title="Gerät"),
            y=alt.Y("cum_share:Q", axis=alt.Axis(format='%'), title="Kumulierte Anteile")
        )
        rule = alt.Chart(pd.DataFrame({"y": [0.8]})).mark_rule(color="#d62728", strokeDash=[4,4]).encode(y="y")
        st.altair_chart((bar & (line + rule)).resolve_scale(y='independent'), use_container_width=True)
    else:
        st.bar_chart(by_dev.set_index(COL_DEVICE)["energy_kwh"])
        st.dataframe(by_dev[[COL_DEVICE, "share", "cum_share"]])


def _savings_charts(agg_base: pd.DataFrame, agg_scn: pd.DataFrame):
    """Vergleicht Baseline vs. Szenario und zeigt Einsparungen."""
    cols = []
    if "energy_kwh" in agg_base.columns and "energy_kwh" in agg_scn.columns:
        cols.append("energy_kwh")
    if "cost" in agg_base.columns and "cost" in agg_scn.columns:
        cols.append("cost")
    if "emissions_kg" in agg_base.columns and "emissions_kg" in agg_scn.columns:
        cols.append("emissions_kg")
    if not cols:
        st.info("Keine vergleichbaren Kennzahlen gefunden.")
        return

    merged = pd.merge(
        agg_base[[COL_DEVICE] + cols],
        agg_scn[[COL_DEVICE] + cols],
        on=COL_DEVICE,
        how="outer",
        suffixes=("_base", "_scn")
    ).fillna(0)

    st.markdown("Einsparungen nach Gerät")
    for c in cols:
        merged[c + "_saved"] = merged[c + "_base"] - merged[c + "_scn"]
    show_cols = [COL_DEVICE] + [c + "_saved" for c in cols]
    show = merged[show_cols]

    if ALT_AVAILABLE:
        import altair as alt
        long = show.melt(COL_DEVICE, var_name="metric", value_name="value")
        long["metric"] = long["metric"].map({
            "energy_kwh_saved": "kWh",
            "cost_saved": "Kosten",
            "emissions_kg_saved": "kg CO₂e",
        }).fillna(long["metric"])
        chart = alt.Chart(long).mark_bar().encode(
            x=alt.X(f"{COL_DEVICE}:N", sort='-y', title="Gerät"),
            y=alt.Y("value:Q", title="Einsparung"),
            color="metric:N"
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.dataframe(show)

    total_kwh = show.get("energy_kwh_saved").sum() if "energy_kwh_saved" in show.columns else float("nan")
    total_cost = show.get("cost_saved").sum() if "cost_saved" in show.columns else float("nan")
    total_em = show.get("emissions_kg_saved").sum() if "emissions_kg_saved" in show.columns else float("nan")
    c1, c2, c3 = st.columns(3)
    with c1:
        if pd.notna(total_kwh):
            st.metric("kWh gespart", f"{total_kwh:,.0f}".replace(",", " "))
    with c2:
        if pd.notna(total_cost):
            st.metric("Kosten gespart", f"{total_cost:,.2f}".replace(",", " "))
    with c3:
        if pd.notna(total_em):
            st.metric("CO₂ gespart (kg)", f"{total_em:,.1f}".replace(",", " "))


def _energy_time_series_charts(ts_df: pd.DataFrame, devices: List[str]):
    st.subheader("Zeitreihe: Verbrauch nach Zeit")
    if ts_df.empty:
        st.info("Keine Zeitreihendaten gefunden.")
        return
    df = ts_df.copy()
    if devices:
        df = df[df[COL_DEVICE].isin(devices)]
    # Summiert pro Zeitstempel und Gerät
    if ALT_AVAILABLE:
        chart = alt.Chart(df).mark_line().encode(
            x=alt.X("timestamp:T", title="Zeit"),
            y=alt.Y("kWh:Q", title="kWh"),
            color=alt.Color(f"{COL_DEVICE}:N", legend=alt.Legend(title="Gerät"))
        ).properties(height=420)
        st.altair_chart(chart, use_container_width=True)
    else:
        pivot = df.pivot_table(index="timestamp", columns=COL_DEVICE, values="kWh", aggfunc="sum").fillna(0)
        st.line_chart(pivot)


def _data_section(df: pd.DataFrame):
    st.subheader("Daten (standardisiert)")
    st.dataframe(df, use_container_width=True, height=460)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Gefilterte Daten als CSV", csv, file_name="energie_dashboard_daten.csv", mime="text/csv")


def main():
    st.set_page_config(page_title="Energie-Dashboard", layout="wide")

    st.title("⚡ Energie-Dashboard")
    st.caption("Spezifische Auswertung für Verbrauch (kWh), Emissionen (kg CO₂e) und Kosten (CHF) basierend auf einer Excel-Vorlage.")

    DEFAULT_PATH = "Energieanalyse_Vorlage_update.xlsx"

    data, sheet_selected = get_data_from_excel(DEFAULT_PATH)

    if data is None:
        st.info("Keine Daten geladen. Bitte Datei in der Seitenleiste auswählen/hochladen.")
        return

    # Falls mehrere Sheets, eines auswählen
    if isinstance(data, dict):
        sheets = list(data.keys())
        if sheet_selected not in sheets:
            sheet_selected = st.sidebar.selectbox("Arbeitsblatt", sheets, index=0)
        df_raw = data[sheet_selected]
    else:
        df_raw = data

    # Wenn die Excel-Tabelle im Vorlagenformat (Zeilenlabels in erster Spalte) vorliegt,
    # wandle sie in eine flache Geräte-Tabelle um und extrahiere optional eine Zeitreihe.
    ts_df: Optional[pd.DataFrame] = None
    try:
        if _is_metric_table(df_raw):
            df_fixed = _from_metric_table(df_raw)
            ts_df = _extract_time_series(df_raw)
        else:
            df_fixed = df_raw
    except Exception as e:
        st.error(f"Konnte Vorlage nicht lesen: {e}")
        df_fixed = df_raw

    # Filter & Optionen
    df_filtered, opts = _sidebar_controls(df_fixed, has_ts=(ts_df is not None))
    # Zeitreihe filtern
    ts_filtered = None
    if ts_df is not None:
        if opts.get("devices"):
            ts_filtered = ts_df[ts_df[COL_DEVICE].isin(opts["devices"])]
        else:
            ts_filtered = ts_df

    # Baseline (keine Reduktion) und Szenario (mit Reduktion)
    base_df, _ = _apply_reduction(df_filtered, ts_filtered, devices=[], reduction_pct=0)
    scn_df, scn_ts = _apply_reduction(df_filtered, ts_filtered, devices=opts.get("savings_devices", []), reduction_pct=opts.get("reduction_pct", 0))

    # Kosten/Emissionen berechnen (einheitlich für Baseline & Szenario)
    base_df, _ = _compute_cost_emissions(
        base_df, ts_filtered,
        price_mode=opts["price_mode"], price_per_kwh=opts["price_per_kwh"],
        co2_mode=opts["co2_mode"], co2_intensity=opts["co2_intensity"],
        tou_enabled=opts.get("tou_enabled", False), peak_start=opts.get("peak_start"), peak_end=opts.get("peak_end"),
        peak_mult=opts.get("peak_mult", 1.0), co2_peak_mult=opts.get("co2_peak_mult", 1.0)
    )
    scn_df, scn_ts = _compute_cost_emissions(
        scn_df, scn_ts,
        price_mode=opts["price_mode"], price_per_kwh=opts["price_per_kwh"],
        co2_mode=opts["co2_mode"], co2_intensity=opts["co2_intensity"],
        tou_enabled=opts.get("tou_enabled", False), peak_start=opts.get("peak_start"), peak_end=opts.get("peak_end"),
        peak_mult=opts.get("peak_mult", 1.0), co2_peak_mult=opts.get("co2_peak_mult", 1.0)
    )

    # KPIs zeigen (basierend auf Szenario)
    _kpis_overview(scn_df)

    # Aggregation
    agg_base = _aggregate(base_df, opts["period"])
    agg_scn = _aggregate(scn_df, opts["period"])

    tabs = ["Energie", "Emissionen", "Kosten", "Einsparungen"] + (["Zeitreihe"] if scn_ts is not None else [])
    selected = st.tabs(tabs)
    with selected[0]:
        _energy_charts(agg_scn, opts["period"])
        _share_charts(agg_scn)
    with selected[1]:
        _emission_charts(agg_scn, opts["period"])
    with selected[2]:
        _cost_charts(agg_scn, opts["period"])
    with selected[3]:
        _savings_charts(agg_base, agg_scn)
    if scn_ts is not None and len(selected) > 4:
        with selected[4]:
            _energy_time_series_charts(scn_ts, opts.get("devices", []))

    with st.expander("Daten ansehen/Export"):
        st.markdown("Szenario-Daten (gefiltert)")
        _data_section(scn_df)
        if scn_ts is not None:
            st.markdown("Zeitreihe (Szenario)")
            st.dataframe(scn_ts, use_container_width=True, height=320)
            csv_ts = scn_ts.to_csv(index=False).encode("utf-8")
            st.download_button("Zeitreihe als CSV", csv_ts, file_name="energie_dashboard_zeitreihe.csv", mime="text/csv")


main()
