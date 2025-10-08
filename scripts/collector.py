# scripts/collector.py
# Runs in GitHub Actions. Produces one JSON snapshot of SPX gamma metrics.

import argparse, json, os, datetime as dt
import numpy as np
import pandas as pd

# Reuse your existing logic from gamma_reporter.py
from gamma_reporter import (
    CONFIG,
    fetch_options_data,
    parse_option_data,
    validate_and_clean_data,
    process_gamma,
    calculate_gamma_profile,
    find_gamma_flip_points,
)

def build_0dte_df(options_df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """
    Build a cleaned 0DTE/1DTE dataframe with the columns expected by process_gamma().
    """
    today = dt.date.today()
    tomorrow = today + dt.timedelta(days=1)

    # Use both today's and tomorrow's expiries near the money (±2% by default)
    otm_lower = spot * (1 - CONFIG['OTM_FILTER_PERCENT'])
    otm_upper = spot * (1 + CONFIG['OTM_FILTER_PERCENT'])

    df = options_df[
        (options_df['ExpirationDate'].dt.date.isin([today, tomorrow])) &
        (options_df['Strike'].between(otm_lower, otm_upper))
    ].copy()

    # Split into calls/puts then merge on (ExpirationDate, Strike)
    calls = df[df['CallPut'] == 'C'][['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
    puts  = df[df['CallPut'] == 'P'][['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
    merged = calls.merge(puts, on=['ExpirationDate','Strike'], suffixes=('_call','_put'))
    merged.columns = [
        'ExpirationDate','StrikePrice','CallIV','CallGamma','CallOpenInt',
        'PutIV','PutGamma','PutOpenInt'
    ]

    # Add 16:00 “close” time to expiry for cleaner DTE math used elsewhere in your code
    merged['ExpirationDate'] = merged['ExpirationDate'] + dt.timedelta(hours=16)

    # Clean & filter
    merged = validate_and_clean_data(merged)
    if merged.empty:
        return merged

    # Compute per‑strike $ gamma and TotalGamma ($B / 1% move)
    merged = process_gamma(merged, spot)
    return merged


def compute_gamma_ratio_and_near_density(df_0dte: pd.DataFrame, spot: float):
    """
    GammaRatio (CallGamma$ / PutGamma$) and near-spot density (|CallGEX|+|PutGEX|
    within ±1% of spot), both in $ billions where relevant.
    """
    if df_0dte.empty:
        return None, None

    # Aggregate to match your plotting logic
    grouped = df_0dte.groupby('StrikePrice', as_index=True).sum(numeric_only=True)
    total_call_gamma = grouped['CallGEX'].sum()       # dollars
    total_put_gamma  = abs(grouped['PutGEX'].sum())   # dollars (abs)
    gamma_ratio = (total_call_gamma / total_put_gamma) if total_put_gamma != 0 else None

    # Near-spot density: sum |CallGEX| + |PutGEX| within ±1% of spot, reported in $B
    window = 0.01 * spot
    mask = (grouped.index >= spot - window) & (grouped.index <= spot + window)
    near_dollars = (grouped.loc[mask, 'CallGEX'].abs().sum() +
                    grouped.loc[mask, 'PutGEX'].abs().sum())
    near_density_B = near_dollars / CONFIG['BILLION']

    return gamma_ratio, near_density_B


def compute_zero_gamma(options_df: pd.DataFrame, spot: float) -> float | None:
    """
    Build the 0&1‑DTE profile and find the first gamma flip (zero‑gamma) near spot.
    Returns the flip level (price) or None.
    """
    expiries = sorted(options_df['ExpirationDate'].unique())
    if len(expiries) < 2:
        return None

    target = expiries[:2]
    prof_calls = options_df[options_df['CallPut'] == 'C'].copy()
    prof_puts  = options_df[options_df['CallPut'] == 'P'].copy()
    prof_calls = prof_calls[prof_calls['ExpirationDate'].isin(target)][['ExpirationDate','Strike','iv','open_interest']]
    prof_puts  = prof_puts [prof_puts ['ExpirationDate'].isin(target)][['ExpirationDate','Strike','iv','open_interest']]

    df_profile = prof_calls.merge(prof_puts, on=['ExpirationDate','Strike'], suffixes=('_call','_put'))
    df_profile.columns = ['ExpirationDate','StrikePrice','CallIV','CallOpenInt','PutIV','PutOpenInt']
    df_profile['ExpirationDate'] = df_profile['ExpirationDate'] + dt.timedelta(hours=16)
    df_profile = validate_and_clean_data(df_profile, for_profile=True)
    if df_profile.empty:
        return None

    from_strike = CONFIG['STRIKE_RANGE_LOWER'] * spot
    to_strike   = CONFIG['STRIKE_RANGE_UPPER'] * spot
    levels, total_gamma = calculate_gamma_profile(df_profile, from_strike, to_strike)
    flips = find_gamma_flip_points(levels, total_gamma)
    if not flips:
        return None

    # Choose the flip closest to spot
    flips = np.array(flips)
    return float(flips[np.argmin(np.abs(flips - spot))])


def make_snapshot(index: str = 'SPX') -> dict:
    """
    Fetch chain once and compute a single-row snapshot with:
    - spot, net_gex (B per 1% move), gamma_ratio, zero_gamma, near_density (B)
    """
    # 1) Fetch CBOE JSON once (delayed quotes)
    options_raw = fetch_options_data(index)  # uses your existing function
    spot = float(options_raw["data"]["close"])
    options_df = pd.DataFrame(options_raw["data"]["options"])
    options_df = parse_option_data(options_df)  # adds CallPut, ExpirationDate, Strike

    # 2) Build 0&1‑DTE dataframe and compute metrics
    df_0dte = build_0dte_df(options_df, spot)
    if df_0dte.empty:
        net_gex_B = None
        gamma_ratio = None
        near_density_B = None
    else:
        net_gex_B = float(df_0dte['TotalGamma'].sum())  # already in $B / 1% from process_gamma
        gamma_ratio, near_density_B = compute_gamma_ratio_and_near_density(df_0dte, spot)

    # 3) Zero‑gamma (flip) from the 0&1‑DTE profile
    zero_gamma = compute_zero_gamma(options_df, spot)

    snapshot = {
        "index": index,
        "timestamp_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "spot": spot,
        "net_gex": net_gex_B,             # $ billions per 1% move
        "gamma_ratio": gamma_ratio,       # >1 means more call gamma than put gamma
        "zero_gamma": zero_gamma,         # flip level (price)
        "near_density": near_density_B,   # |gamma| near spot in $B
    }
    return snapshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Path to write snapshot JSON")
    ap.add_argument("--index", default=os.getenv("INDEX", "SPX"))
    args = ap.parse_args()

    snap = make_snapshot(args.index)
    with open(args.out, "w") as f:
        json.dump(snap, f, separators=(",", ":"), ensure_ascii=False)
    print(f"Saved snapshot to {args.out}: {snap}")

if __name__ == "__main__":
    main()
