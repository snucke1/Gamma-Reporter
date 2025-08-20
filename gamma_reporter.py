# Save this file as gamma_reporter_combined.py

import pandas as pd
import numpy as np
from scipy.stats import norm # <-- Added from your first script
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import requests
import sys
import logging
import warnings
import os # For environment variables
import smtplib # For sending email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION (Merged from both scripts) ---
CONFIG = {
    'STRIKE_RANGE_WIDTH': 500, 'OTM_FILTER_PERCENT': 0.01,
    'STRIKE_RANGE_LOWER': 0.8, 'STRIKE_RANGE_UPPER': 1.2,
    'CONTRACT_SIZE': 100, 'PERCENT_MOVE': 0.01, 'MONTHS_TO_INCLUDE': 6,
    'BILLION': 10**9, 'TRADING_DAYS_PER_YEAR': 252,
    'RISK_FREE_RATE': 0.05, 'DIVIDEND_YIELD': 0.02,
    'GR_HIGH_THRESHOLD': 1.25, 'GR_LOW_THRESHOLD': 0.75
}
pd.options.display.float_format = '{:,.4f}'.format

# --- DATA PROCESSING & CALCULATION FUNCTIONS (Merged from both scripts) ---

# --- NEW: Black-Scholes Gamma Calculation for the Profile Plot ---
def calcGammaEx(S, K, vol, T, r, q, OI):
    """Calculate Black-Scholes gamma exposure for options."""
    if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
        return 0
    try:
        dp = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        gamma = np.exp(-q * T) * norm.pdf(dp) / (S * vol * np.sqrt(T))
        return OI * CONFIG['CONTRACT_SIZE'] * S * S * CONFIG['PERCENT_MOVE'] * gamma
    except (ValueError, ZeroDivisionError):
        return 0

# --- NEW: Function to Calculate the Full Gamma Profile ---
def calculate_gamma_profile(df, from_strike, to_strike):
    """Calculate gamma exposure profile across different spot levels."""
    levels = np.linspace(from_strike, to_strike, 100)
    today_date = date.today()
    df['daysTillExp'] = df['ExpirationDate'].apply(
        lambda x: max(1 / CONFIG['TRADING_DAYS_PER_YEAR'], 
                     np.busday_count(today_date, x.date()) / CONFIG['TRADING_DAYS_PER_YEAR'])
    )
    total_gamma = []
    for level in levels:
        call_gamma = df.apply(
            lambda row: calcGammaEx(level, row['StrikePrice'], row['CallIV'], 
                                  row['daysTillExp'], CONFIG['RISK_FREE_RATE'], 
                                  CONFIG['DIVIDEND_YIELD'], row['CallOpenInt']), 
            axis=1
        )
        put_gamma = df.apply(
            lambda row: calcGammaEx(level, row['StrikePrice'], row['PutIV'], 
                                  row['daysTillExp'], CONFIG['RISK_FREE_RATE'], 
                                  CONFIG['DIVIDEND_YIELD'], row['PutOpenInt']), 
            axis=1
        )
        total = (call_gamma.sum() - put_gamma.sum()) / CONFIG['BILLION']
        total_gamma.append(total)
    return levels, total_gamma

# --- NEW: Function to find zero-crossing points ---
def find_gamma_flip_points(levels, total_gamma):
    """Find points where gamma crosses zero using linear interpolation."""
    flip_points = []
    zero_crossings = np.where(np.diff(np.sign(total_gamma)))[0]
    for idx in zero_crossings:
        if idx < len(levels) - 1:
            x1, x2 = levels[idx], levels[idx + 1]
            y1, y2 = total_gamma[idx], total_gamma[idx + 1]
            if y2 != y1:
                zero_point = x1 - y1 * (x2 - x1) / (y2 - y1)
                flip_points.append(zero_point)
    return flip_points

# --- Helper functions from the GitHub script (unchanged) ---
def parse_option_data(data_df):
    data_df['CallPut'] = data_df['option'].str.slice(start=-9, stop=-8)
    data_df['ExpirationDate'] = pd.to_datetime(data_df['option'].str.slice(start=-15, stop=-9), format='%y%m%d', errors='coerce').dropna()
    data_df['Strike'] = pd.to_numeric(data_df['option'].str.slice(start=-8, stop=-3), errors='coerce').dropna()
    return data_df.dropna(subset=['ExpirationDate', 'Strike'])

def validate_and_clean_data(df):
    numeric_columns = ['CallIV', 'PutIV', 'CallGamma', 'PutGamma', 'CallOpenInt', 'PutOpenInt', 'StrikePrice']
    for col in numeric_columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=numeric_columns)
    df = df[df['StrikePrice'] > 0]
    df = df[(df['CallIV'] > 0) & (df['PutIV'] > 0) & (df['CallIV'] < 5) & (df['PutIV'] < 5)]
    df = df[(df['CallOpenInt'] >= 0) & (df['PutOpenInt'] >= 0)]
    return df

def fetch_options_data(index):
    try:
        url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{index}.json"
        logging.info(f"Fetching data from: {url}")
        response = requests.get(url, timeout=30); response.raise_for_status()
        options = response.json()
        if 'data' not in options or 'options' not in options['data']: raise ValueError("Unexpected data format")
        return options
    except Exception as e:
        logging.error(f"Error fetching data: {e}"); sys.exit(1)

def process_gamma_for_bars(df, spot_price):
    df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * CONFIG['CONTRACT_SIZE'] * spot_price * spot_price * CONFIG['PERCENT_MOVE'] * 0.5
    df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * CONFIG['CONTRACT_SIZE'] * spot_price * spot_price * CONFIG['PERCENT_MOVE'] * -1 * 0.5
    df['TotalGamma'] = (df['CallGEX'] + df['PutGEX']) / CONFIG['BILLION']
    return df

# --- PLOTTING FUNCTION for Bar Charts (Unchanged) ---
def plot_gamma_bars(df, spot_price, from_strike, to_strike, index, title_prefix, ax1, ax2, bin_width=None, add_guidance_box=False):
    # This function remains the same as in the original GitHub script
    if bin_width is not None:
        df = df.copy(); df['StrikeBin'] = (df['StrikePrice'] // bin_width) * bin_width
        group_col = 'StrikeBin'
    else: group_col = 'StrikePrice'
    df_agg = df.groupby([group_col]).sum(numeric_only=True)
    strikes = df_agg.index.values
    if len(strikes) > 1:
        if bin_width is not None: bar_width = bin_width
        else:
            strike_diffs = np.diff(sorted(strikes)); bar_width = np.median(strike_diffs) * 0.8
            max_width = np.min(strike_diffs) * 0.9 if len(strike_diffs) > 0 else 10
            bar_width = min(bar_width, max_width, 10)
    else: bar_width = bin_width if bin_width is not None else 5
    total_call_gamma = df_agg['CallGEX'].sum()
    total_put_gamma = abs(df_agg['PutGEX'].sum())
    gamma_ratio = total_call_gamma / total_put_gamma if total_put_gamma != 0 else np.nan
    ax1.grid(True, alpha=0.3)
    ax1.bar(strikes, df_agg['TotalGamma'].values, width=bar_width, linewidth=1.5 if bin_width else 0.5, edgecolor='k', color='steelblue', alpha=0.7, label="Net Gamma")
    ax1.set_xlim([from_strike, to_strike])
    ax1.set_title(f"{title_prefix} Total Gamma: ${df['TotalGamma'].sum():.2f} Bn per 1% {index} Move", fontweight="bold", fontsize=14)
    ax1.set_xlabel('Strike Price', fontweight="bold"); ax1.set_ylabel('Spot Gamma Exposure ($B/1% move)', fontweight="bold")
    ax1.axvline(x=spot_price, color='red', lw=2, linestyle='--', label=f"{index} Spot: ${spot_price:,.0f}")
    ax1.legend()
    # ... (rest of the function is identical to the original)
    if add_guidance_box and not np.isnan(gamma_ratio):
        # Guidance box logic remains here...
        high_gr_text = (f"GR > {CONFIG['GR_HIGH_THRESHOLD']} (Currently: {gamma_ratio:.2f}) - High Positive Gamma\nExpect: Tight ranges, dip-buying, upward drift.")
        low_gr_text = (f"GR < {CONFIG['GR_LOW_THRESHOLD']} (Currently: {gamma_ratio:.2f}) - Low/Negative Gamma\nExpect: Widening ranges, rally-selling, downside bias.")
        neutral_gr_text = (f"{CONFIG['GR_LOW_THRESHOLD']} ≤ GR ≤ {CONFIG['GR_HIGH_THRESHOLD']} (Currently: {gamma_ratio:.2f}) - Balanced Gamma\nExpect: Choppy or range-bound market.")
        if gamma_ratio > CONFIG['GR_HIGH_THRESHOLD']: guidance_text = high_gr_text
        elif gamma_ratio < CONFIG['GR_LOW_THRESHOLD']: guidance_text = low_gr_text
        else: guidance_text = neutral_gr_text
        ax1.text(0.02, 0.98, guidance_text, transform=ax1.transAxes, fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.4), verticalalignment='top')
    ax2.grid(True, alpha=0.3)
    ax2.bar(strikes, df_agg['CallGEX'].values / CONFIG['BILLION'], width=bar_width, linewidth=1.5 if bin_width else 0.5, edgecolor='k', color='green', alpha=0.7, label="Call Gamma")
    ax2.bar(strikes, df_agg['PutGEX'].values / CONFIG['BILLION'], width=bar_width, linewidth=1.5 if bin_width else 0.5, edgecolor='k', color='red', alpha=0.7, label="Put Gamma")
    ax2.set_xlim([from_strike, to_strike])
    ax2.set_title(f"{title_prefix} Gamma Exposure by Option Type", fontweight="bold", fontsize=14)
    ax2.legend()

# --- MODIFIED: Email Function to handle multiple attachments ---
def send_email_with_charts(image_paths, index, spot_price):
    sender_email = os.environ.get('SENDER_EMAIL')
    receiver_email = os.environ.get('RECEIVER_EMAIL')
    email_password = os.environ.get('EMAIL_APP_PASSWORD')

    if not all([sender_email, receiver_email, email_password]):
        logging.error("Email credentials not found in environment variables.")
        return

    msg = MIMEMultipart()
    today_str = date.today().strftime("%Y-%m-%d")
    msg['Subject'] = f"Gamma Report for {index} - {today_str}"
    msg['From'] = sender_email
    msg['To'] = receiver_email

    body = f"Attached are the gamma charts for {index}.\n\nCurrent Spot Price: ${spot_price:,.2f}"
    msg.attach(MIMEText(body, 'plain'))

    # Loop through the list of image paths and attach each one
    for image_path in image_paths:
        with open(image_path, 'rb') as fp:
            img = MIMEImage(fp.read())
            img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
            msg.attach(img)

    try:
        logging.info(f"Sending email with {len(image_paths)} charts to {receiver_email}...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, email_password)
            server.send_message(msg)
        logging.info("Email sent successfully!")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

# --- MAIN EXECUTION BLOCK ---
def main():
    if len(sys.argv) != 2:
        print("Usage: python gamma_reporter_combined.py <INDEX>")
        sys.exit(1)
    index = sys.argv[1].upper()
    options = fetch_options_data(index)
    spot_price = float(options["data"]["close"])
    data_df = pd.DataFrame(options["data"]["options"])
    data_df = parse_option_data(data_df)

    # --- Data Preparation for 0-1 DTE ---
    today_date = date.today()
    tomorrow_date = today_date + timedelta(days=1)
    data_0dte_raw = data_df[(data_df['ExpirationDate'].dt.date == today_date) | 
                           (data_df['ExpirationDate'].dt.date == tomorrow_date)]
    calls_0dte = data_0dte_raw[data_0dte_raw['CallPut'] == "C"][['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
    puts_0dte = data_0dte_raw[data_0dte_raw['CallPut'] == "P"][['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
    df_0dte = calls_0dte.merge(puts_0dte, on=['ExpirationDate', 'Strike'], suffixes=('_call', '_put'))
    df_0dte.columns = ['ExpirationDate', 'StrikePrice', 'CallIV', 'CallGamma', 'CallOpenInt', 'PutIV', 'PutGamma', 'PutOpenInt']
    df_0dte['ExpirationDate'] += timedelta(hours=16)
    df_0dte = validate_and_clean_data(df_0dte)

    # --- Data Preparation for 6 Months ---
    six_months_later = today_date + relativedelta(months=CONFIG['MONTHS_TO_INCLUDE'])
    data_6m_raw = data_df[(data_df['ExpirationDate'].dt.date > today_date) & 
                          (data_df['ExpirationDate'].dt.date <= six_months_later)]
    calls_6m = data_6m_raw[data_6m_raw['CallPut'] == "C"][['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
    puts_6m = data_6m_raw[data_6m_raw['CallPut'] == "P"][['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
    df_6m = calls_6m.merge(puts_6m, on=['ExpirationDate', 'Strike'], suffixes=('_call', '_put'))
    df_6m.columns = ['ExpirationDate', 'StrikePrice', 'CallIV', 'CallGamma', 'CallOpenInt', 'PutIV', 'PutGamma', 'PutOpenInt']
    df_6m['ExpirationDate'] += timedelta(hours=16)
    df_6m = validate_and_clean_data(df_6m)

    # --- PLOT 1: Bar Charts (from GitHub script) ---
    logging.info("Generating bar charts...")
    fig_bars, axs = plt.subplots(2, 2, figsize=(20, 14))
    
    # 0-1DTE Bars
    if not df_0dte.empty:
        df_0dte_processed = process_gamma_for_bars(df_0dte.copy(), spot_price)
        otm_lower = spot_price * (1 - CONFIG['OTM_FILTER_PERCENT'])
        otm_upper = spot_price * (1 + CONFIG['OTM_FILTER_PERCENT'])
        plot_gamma_bars(df_0dte_processed, spot_price, otm_lower, otm_upper, index, "0-1DTE", axs[0,0], axs[1,0])
    else:
        axs[0,0].set_title("No valid 0-1DTE data"); axs[1,0].set_title("No valid 0-1DTE data")

    # 6M Bars
    if not df_6m.empty:
        df_6m_processed = process_gamma_for_bars(df_6m.copy(), spot_price)
        from_strike_6m, to_strike_6m = CONFIG['STRIKE_RANGE_LOWER'] * spot_price, CONFIG['STRIKE_RANGE_UPPER'] * spot_price
        plot_gamma_bars(df_6m_processed, spot_price, from_strike_6m, to_strike_6m, index, "6M (excl. today)", axs[0,1], axs[1,1], bin_width=20, add_guidance_box=True)
    else:
        axs[0,1].set_title("No valid 6M data"); axs[1,1].set_title("No valid 6M data")

    plt.tight_layout()
    bar_chart_filename = f"gamma_bars_{index}_{date.today()}.png"
    plt.savefig(bar_chart_filename)
    plt.close(fig_bars) # Close the figure to free up memory
    logging.info(f"Bar chart saved as {bar_chart_filename}")

    # --- PLOT 2: Gamma Profile Line Chart (from your script) ---
    logging.info("Generating gamma profile chart...")
    profile_chart_filename = f"gamma_profile_{index}_{date.today()}.png"
    if not df_0dte.empty:
        from_strike, to_strike = CONFIG['STRIKE_RANGE_LOWER'] * spot_price, CONFIG['STRIKE_RANGE_UPPER'] * spot_price
        levels, total_gamma = calculate_gamma_profile(df_0dte.copy(), from_strike, to_strike)
        flip_points = find_gamma_flip_points(levels, total_gamma)

        fig_profile, ax_profile = plt.subplots(figsize=(12, 8))
        ax_profile.grid(True, alpha=0.3)
        ax_profile.plot(levels, total_gamma, 'b-', linewidth=2, label="Total Gamma (0 & 1 DTE)")
        ax_profile.set_title(f"Gamma Profile (0-DTE & 1-DTE) - {index} - {today_date.strftime('%d %b %Y')}", fontweight="bold", fontsize=16)
        ax_profile.set_xlabel('Index Price', fontweight="bold")
        ax_profile.set_ylabel('Gamma Exposure ($ billions / 1% move)', fontweight="bold")
        ax_profile.axhline(y=0, color='black', lw=1)
        ax_profile.axvline(x=spot_price, color='red', lw=2, linestyle='--', label=f"{index} Spot: ${spot_price:,.0f}")

        for i, flip in enumerate(flip_points):
            if from_strike <= flip <= to_strike:
                ax_profile.axvline(x=flip, color='green', lw=1.5, linestyle=':', label=f"Gamma Flip: ${flip:,.0f}" if i == 0 else "")
        
        ax_profile.fill_between(levels, total_gamma, 0, where=np.array(total_gamma) > 0, facecolor='green', alpha=0.1)
        ax_profile.fill_between(levels, total_gamma, 0, where=np.array(total_gamma) < 0, facecolor='red', alpha=0.1)
        ax_profile.set_xlim([from_strike, to_strike])
        ax_profile.legend(loc='best')
        plt.tight_layout()
        plt.savefig(profile_chart_filename)
        plt.close(fig_profile)
        logging.info(f"Profile chart saved as {profile_chart_filename}")
    else:
        logging.warning("Skipping profile chart generation due to no 0-1DTE data.")
        profile_chart_filename = None # Set to None if not created

    # --- Send the email with both charts ---
    chart_files_to_send = [f for f in [bar_chart_filename, profile_chart_filename] if f is not None]
    if chart_files_to_send:
        send_email_with_charts(chart_files_to_send, index, spot_price)
    else:
        logging.error("No charts were generated to email.")

if __name__ == "__main__":
    main()
