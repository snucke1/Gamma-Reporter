# Save this file as gamma_reporter.py

import pandas as pd
import numpy as np
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

# --- CONFIGURATION (No changes here) ---
CONFIG = {
    'STRIKE_RANGE_WIDTH': 500, 'OTM_FILTER_PERCENT': 0.01,
    'STRIKE_RANGE_LOWER': 0.8, 'STRIKE_RANGE_UPPER': 1.2,
    'CONTRACT_SIZE': 100, 'PERCENT_MOVE': 0.01, 'MONTHS_TO_INCLUDE': 6,
    'BILLION': 10**9, 'TRADING_DAYS_PER_YEAR': 252,
    'RISK_FREE_RATE': 0.05, 'DIVIDEND_YIELD': 0.02,
    'GR_HIGH_THRESHOLD': 1.25, 'GR_LOW_THRESHOLD': 0.75
}
pd.options.display.float_format = '{:,.4f}'.format

# --- DATA PROCESSING FUNCTIONS (No changes here) ---
def parse_option_data(data_df):
    data_df['CallPut'] = data_df['option'].str.slice(start=-9, stop=-8)
    data_df['ExpirationDate'] = data_df['option'].str.slice(start=-15, stop=-9)
    data_df['ExpirationDate'] = pd.to_datetime(data_df['ExpirationDate'], format='%y%m%d', errors='coerce')
    data_df = data_df[data_df['ExpirationDate'].notna()]
    data_df['Strike'] = data_df['option'].str.slice(start=-8, stop=-3)
    data_df['Strike'] = pd.to_numeric(data_df['Strike'].str.lstrip('0'), errors='coerce')
    data_df = data_df[data_df['Strike'].notna()]
    return data_df

def validate_and_clean_data(df):
    numeric_columns = ['CallIV', 'PutIV', 'CallGamma', 'PutGamma', 'CallOpenInt', 'PutOpenInt', 'StrikePrice']
    for col in numeric_columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[df['StrikePrice'] > 0]
    df = df[(df['CallIV'] > 0) & (df['PutIV'] > 0)]
    df = df[(df['CallOpenInt'] >= 0) & (df['PutOpenInt'] >= 0)]
    df = df[(df['CallIV'] < 5) & (df['PutIV'] < 5)]
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

def process_gamma(df, spot_price):
    df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * CONFIG['CONTRACT_SIZE'] * spot_price * spot_price * CONFIG['PERCENT_MOVE'] * 0.5
    df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * CONFIG['CONTRACT_SIZE'] * spot_price * spot_price * CONFIG['PERCENT_MOVE'] * -1 * 0.5
    df['TotalGamma'] = (df['CallGEX'] + df['PutGEX']) / CONFIG['BILLION']
    return df

# --- PLOTTING FUNCTION (No changes here) ---
def plot_gamma(df, spot_price, from_strike, to_strike, index, title_prefix, ax1, ax2, bin_width=None, add_guidance_box=False):
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
    total_call_oi = df_agg['CallOpenInt'].sum(); total_put_oi = df_agg['PutOpenInt'].sum()
    put_call_ratio = total_put_oi / total_call_oi if total_call_oi != 0 else np.nan
    ax1.grid(True, alpha=0.3)
    ax1.bar(strikes, df_agg['TotalGamma'].values, width=bar_width, linewidth=1.5 if bin_width else 0.5, edgecolor='k', color='steelblue', alpha=0.7, label="Net Gamma")
    ax1.set_xlim([from_strike, to_strike])
    ax1.set_title(f"{title_prefix} Total Gamma: ${df['TotalGamma'].sum():.2f} Bn per 1% {index} Move", fontweight="bold", fontsize=14)
    ax1.set_xlabel('Strike Price', fontweight="bold"); ax1.set_ylabel('Spot Gamma Exposure ($B/1% move)', fontweight="bold")
    ax1.axvline(x=spot_price, color='red', lw=2, linestyle='--', label=f"{index} Spot: ${spot_price:,.0f}")
    ax1.legend()
    if add_guidance_box and not np.isnan(gamma_ratio):
        high_gr_text = (f"GR > {CONFIG['GR_HIGH_THRESHOLD']} (Currently: {gamma_ratio:.2f}) - High Positive Gamma\n-----------------------------------------------------------\nExpect: Tight ranges, dip-buying, upward drift\n(especially inside 2 weeks to OPEX).\n\nTrade Ideas:\n • If mkt opens down & momentum turns up:\n   Buy ATM/SOTM calls/call spreads or sell ATM put spreads.\n • If mkt opens higher: Tougher trade, low R/R.\n   Could sell OTM puts/spreads.\n")
        low_gr_text = (f"GR < {CONFIG['GR_LOW_THRESHOLD']} (Currently: {gamma_ratio:.2f}) - Low/Negative Gamma\n-----------------------------------------------------------\nExpect: Widening ranges, rally-selling, downside bias.\n\nTrade Ideas:\n • If mkt opens high & momentum turns down:\n   Sell calls/spreads or buy puts/spreads.\n • If mkt opens down & momentum turns up:\n   Consider opposite trades.\n")
        neutral_gr_text = (f"{CONFIG['GR_LOW_THRESHOLD']} ≤ GR ≤ {CONFIG['GR_HIGH_THRESHOLD']} (Currently: {gamma_ratio:.2f}) - Balanced Gamma\n-----------------------------------------------------------\nExpect: Choppy or range-bound market. No strong directional\npressure from dealer hedging.\n")
        always_on_text = ("\nGeneral Principles:\n-----------------------------------------------------------\n4. Tight Risk: If a trade thesis is falsified, exit promptly.\n5. Position Mgt: On winning trades, consider selling a portion\n   to create a 'free trade' and let the rest run.")
        if gamma_ratio > CONFIG['GR_HIGH_THRESHOLD']: guidance_text = high_gr_text + always_on_text
        elif gamma_ratio < CONFIG['GR_LOW_THRESHOLD']: guidance_text = low_gr_text + always_on_text
        else: guidance_text = neutral_gr_text + always_on_text
        ax1.text(0.02, 0.98, guidance_text, transform=ax1.transAxes, fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.6), verticalalignment='top')
    ax2.grid(True, alpha=0.3)
    ax2.bar(strikes, df_agg['CallGEX'].values / CONFIG['BILLION'], width=bar_width, linewidth=1.5 if bin_width else 0.5, edgecolor='k', color='green', alpha=0.7, label="Call Gamma")
    ax2.bar(strikes, df_agg['PutGEX'].values / CONFIG['BILLION'], width=bar_width, linewidth=1.5 if bin_width else 0.5, edgecolor='k', color='red', alpha=0.7, label="Put Gamma")
    ax2.set_xlim([from_strike, to_strike])
    ax2.set_title(f"{title_prefix} Gamma Exposure by Option Type", fontweight="bold", fontsize=14)
    ax2.set_xlabel('Strike Price', fontweight="bold"); ax2.set_ylabel('Spot Gamma Exposure ($B/1% move)', fontweight="bold")
    ax2.axvline(x=spot_price, color='red', lw=2, linestyle='--', label=f"{index} Spot: ${spot_price:,.0f}")
    metrics_text = (f'Gamma Ratio (Call/Put): {gamma_ratio:.2f}\nPut/Call OI Ratio: {put_call_ratio:.2f}\nTotal Call Gamma: ${total_call_gamma/CONFIG["BILLION"]:.2f}B\nTotal Put Gamma: ${total_put_gamma/CONFIG["BILLION"]:.2f}B')
    ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes, fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8), verticalalignment='top')
    ax2.legend()

# --- NEW: Email Function ---
def send_email_with_chart(image_path, index, spot_price):
    # Get credentials from environment variables
    sender_email = os.environ.get('SENDER_EMAIL')
    receiver_email = os.environ.get('RECEIVER_EMAIL')
    email_password = os.environ.get('EMAIL_APP_PASSWORD') # Use an App Password

    if not all([sender_email, receiver_email, email_password]):
        logging.error("Email credentials not found in environment variables.")
        return

    # Create the email
    msg = MIMEMultipart()
    today_str = date.today().strftime("%Y-%m-%d")
    msg['Subject'] = f"Gamma Exposure Report for {index} - {today_str}"
    msg['From'] = sender_email
    msg['To'] = receiver_email

    body = f"Attached is the gamma exposure chart for {index}.\n\nCurrent Spot Price: ${spot_price:,.2f}"
    msg.attach(MIMEText(body, 'plain'))

    # Attach the image
    with open(image_path, 'rb') as fp:
        img = MIMEImage(fp.read())
        img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
        msg.attach(img)

    # Send the email via Gmail's SMTP server
    try:
        logging.info(f"Sending email to {receiver_email}...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, email_password)
            server.send_message(msg)
        logging.info("Email sent successfully!")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

# --- MODIFIED: Main Execution Block ---
def main():
    if len(sys.argv) != 2:
        print("Usage: python gamma_reporter.py <INDEX>")
        sys.exit(1)
    index = sys.argv[1].upper()
    options = fetch_options_data(index)
    spot_price = float(options["data"]["close"])
    data_df = pd.DataFrame(options["data"]["options"])
    data_df = parse_option_data(data_df)

    # --- Data processing for 0DTE and 6M (No changes here) ---
    today_date = date.today()
    half_range = CONFIG['STRIKE_RANGE_WIDTH'] / 2
    from_strike_0dte, to_strike_0dte = spot_price - half_range, spot_price + half_range
    otm_lower_bound = spot_price * (1 - CONFIG['OTM_FILTER_PERCENT'])
    otm_upper_bound = spot_price * (1 + CONFIG['OTM_FILTER_PERCENT'])
    data_0dte = data_df[data_df['ExpirationDate'].dt.date == today_date]
    data_0dte = data_0dte[(data_0dte['Strike'] >= otm_lower_bound) & (data_0dte['Strike'] <= otm_upper_bound)]
    calls_0dte = data_0dte[data_0dte['CallPut'] == "C"].reset_index(drop=True)[['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
    puts_0dte = data_0dte[data_0dte['CallPut'] == "P"].reset_index(drop=True)[['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
    df_0dte = calls_0dte.merge(puts_0dte, on=['ExpirationDate', 'Strike'], suffixes=('_call', '_put'))
    df_0dte.columns = ['ExpirationDate', 'StrikePrice', 'CallIV', 'CallGamma', 'CallOpenInt', 'PutIV', 'PutGamma', 'PutOpenInt']
    df_0dte['ExpirationDate'] += timedelta(hours=16)
    df_0dte = validate_and_clean_data(df_0dte)
    if len(df_0dte) == 0: logging.warning("No valid 0-DTE data after cleaning")
    else: df_0dte = process_gamma(df_0dte, spot_price)

    six_months_later = today_date + relativedelta(months=CONFIG['MONTHS_TO_INCLUDE'])
    from_strike_6m, to_strike_6m = CONFIG['STRIKE_RANGE_LOWER'] * spot_price, CONFIG['STRIKE_RANGE_UPPER'] * spot_price
    data_6m = data_df[data_df['ExpirationDate'].dt.date <= six_months_later]
    calls_6m = data_6m[data_6m['CallPut'] == "C"].reset_index(drop=True)[['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
    puts_6m = data_6m[data_6m['CallPut'] == "P"].reset_index(drop=True)[['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
    df_6m = calls_6m.merge(puts_6m, on=['ExpirationDate', 'Strike'], suffixes=('_call', '_put'))
    df_6m.columns = ['ExpirationDate', 'StrikePrice', 'CallIV', 'CallGamma', 'CallOpenInt', 'PutIV', 'PutGamma', 'PutOpenInt']
    df_6m['ExpirationDate'] += timedelta(hours=16)
    df_6m = validate_and_clean_data(df_6m)
    if len(df_6m) == 0: logging.warning("No valid 6M data after cleaning")
    else: df_6m = process_gamma(df_6m, spot_price)

    # --- MODIFIED: Plotting ---
    fig, axs = plt.subplots(2, 2, figsize=(20, 14))
    if len(df_0dte) > 0: plot_gamma(df_0dte, spot_price, from_strike_0dte, to_strike_0dte, index, "0-DTE", axs[0,0], axs[1,0])
    else: axs[0,0].set_title("No valid 0-DTE data"); axs[1,0].set_title("No valid 0-DTE data")
    if len(df_6m) > 0: plot_gamma(df_6m, spot_price, from_strike_6m, to_strike_6m, index, "6M", axs[0,1], axs[1,1], bin_width=20, add_guidance_box=True)
    else: axs[0,1].set_title("No valid 6M data"); axs[1,1].set_title("No valid 6M data")

    plt.tight_layout()
    chart_filename = f"gamma_report_{index}_{date.today()}.png"
    plt.savefig(chart_filename) # Save the plot instead of showing it
    logging.info(f"Chart saved as {chart_filename}")
    
    # Send the email
    send_email_with_chart(chart_filename, index, spot_price)


if __name__ == "__main__":
    main()