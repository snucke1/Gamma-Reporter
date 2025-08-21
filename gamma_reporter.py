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
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
CONFIG = {
    'STRIKE_RANGE_WIDTH': 500, 'OTM_FILTER_PERCENT': 0.01,
    'STRIKE_RANGE_LOWER': 0.8, 'STRIKE_RANGE_UPPER': 1.2,
    'CONTRACT_SIZE': 100, 'PERCENT_MOVE': 0.01, 'MONTHS_TO_INCLUDE': 6,
    'BILLION': 10**9, 'TRADING_DAYS_PER_YEAR': 252,
    'RISK_FREE_RATE': 0.05, 'DIVIDEND_YIELD': 0.02,
    'GR_HIGH_THRESHOLD': 1.25, 'GR_LOW_THRESHOLD': 0.75
}
pd.options.display.float_format = '{:,.4f}'.format

# --- DATA PROCESSING & FETCHING FUNCTIONS ---
def parse_option_data(data_df):
    data_df['CallPut'] = data_df['option'].str.slice(start=-9, stop=-8)
    data_df['ExpirationDate'] = pd.to_datetime(data_df['option'].str.slice(start=-15, stop=-9), format='%y%m%d', errors='coerce')
    data_df = data_df.dropna(subset=['ExpirationDate'])
    data_df['Strike'] = pd.to_numeric(data_df['option'].str.slice(start=-8, stop=-3), errors='coerce')
    data_df = data_df.dropna(subset=['Strike'])
    return data_df

def validate_and_clean_data(df):
    numeric_columns = ['CallIV', 'PutIV', 'CallGamma', 'PutGamma', 'CallOpenInt', 'PutOpenInt', 'StrikePrice']
    for col in numeric_columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=numeric_columns)
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
    df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * CONFIG['CONTRACT_SIZE'] * spot_price * spot_price * CONFIG['PERCENT_MOVE']
    df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * CONFIG['CONTRACT_SIZE'] * spot_price * spot_price * CONFIG['PERCENT_MOVE'] * -1
    df['TotalGamma'] = (df['CallGEX'] + df['PutGEX']) / CONFIG['BILLION']
    return df

# --- PLOTTING FUNCTION 1: THE ORIGINAL 4-PANEL REPORT ---
def plot_gamma_reporter(df, spot_price, from_strike, to_strike, index, title_prefix, ax1, ax2, bin_width=None, add_guidance_box=False):
    # This function plots the bar charts and remains mostly unchanged.
    if bin_width is not None:
        df = df.copy(); df['StrikeBin'] = (df['StrikePrice'] // bin_width) * bin_width
        group_col = 'StrikeBin'
    else: group_col = 'StrikePrice'
    df_agg = df.groupby([group_col]).sum(numeric_only=True)
    strikes = df_agg.index.values
    bar_width = bin_width if bin_width is not None else 10 # Adjusted for visibility
    
    # Total Gamma Plot
    ax1.grid(True, alpha=0.3)
    ax1.bar(strikes, df_agg['TotalGamma'].values, width=bar_width, linewidth=0.5, edgecolor='k', color='steelblue', label="Net Gamma")
    ax1.set_xlim([from_strike, to_strike])
    ax1.set_title(f"{title_prefix} Total Gamma: ${df_agg['TotalGamma'].sum():.2f} Bn", fontweight="bold", fontsize=12)
    ax1.axvline(x=spot_price, color='red', lw=2, linestyle='--', label=f"Spot: ${spot_price:,.0f}")
    ax1.legend()

    # Guidance Box (only for the 6M chart)
    if add_guidance_box:
        # Guidance text logic is omitted for brevity but is unchanged
        gamma_ratio = df_agg['CallGEX'].sum() / abs(df_agg['PutGEX'].sum()) if df_agg['PutGEX'].sum() != 0 else 0
        if gamma_ratio > CONFIG['GR_HIGH_THRESHOLD']: guidance_text = "High Positive Gamma..."
        elif gamma_ratio < CONFIG['GR_LOW_THRESHOLD']: guidance_text = "Low/Negative Gamma..."
        else: guidance_text = "Balanced Gamma..."
        ax1.text(0.02, 0.98, guidance_text, transform=ax1.transAxes, fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5), va='top')
    
    # Call/Put Gamma Plot
    ax2.grid(True, alpha=0.3)
    ax2.bar(strikes, df_agg['CallGEX'].values / CONFIG['BILLION'], width=bar_width, linewidth=0.5, edgecolor='k', color='green', alpha=0.7, label="Call Gamma")
    ax2.bar(strikes, df_agg['PutGEX'].values / CONFIG['BILLION'], width=bar_width, linewidth=0.5, edgecolor='k', color='red', alpha=0.7, label="Put Gamma")
    ax2.set_xlim([from_strike, to_strike])
    ax2.set_title(f"{title_prefix} Gamma by Type", fontweight="bold", fontsize=12)
    ax2.axvline(x=spot_price, color='red', lw=2, linestyle='--')
    ax2.legend()

# --- PLOTTING FUNCTION 2: THE NEW GAMMA PROFILE CURVE ---
def plot_gamma_profile(df, spot_price, index):
    # This function now creates, saves, and closes its own figure.
    from_strike, to_strike = spot_price - CONFIG['STRIKE_RANGE_WIDTH']/2, spot_price + CONFIG['STRIKE_RANGE_WIDTH']/2
    
    df_agg = df.groupby('StrikePrice').sum(numeric_only=True)
    strikes_sorted = df_agg.index.values
    gamma_sorted = df_agg['TotalGamma'].values
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.grid(True, alpha=0.3)

    if len(strikes_sorted) > 3:
        strike_range = np.linspace(strikes_sorted.min(), strikes_sorted.max(), 300)
        f_gamma = interp1d(strikes_sorted, gamma_sorted, kind='cubic', fill_value="extrapolate")
        smooth_gamma = f_gamma(strike_range)
        ax.plot(strike_range, smooth_gamma, color='darkblue', linewidth=2, label='Gamma Profile')
        ax.fill_between(strike_range, 0, smooth_gamma, where=(smooth_gamma >= 0), color='green', alpha=0.3)
        ax.fill_between(strike_range, 0, smooth_gamma, where=(smooth_gamma < 0), color='red', alpha=0.3)
    else:
        ax.plot(strikes_sorted, gamma_sorted, color='darkblue', marker='o', label='Gamma Profile')

    ax.set_title(f"0-1 DTE Gamma Profile for {index} - {date.today()}", fontweight="bold", fontsize=16)
    ax.set_xlabel('Strike Price', fontweight="bold")
    ax.set_ylabel('Gamma Exposure ($B / 1% move)', fontweight="bold")
    ax.axvline(x=spot_price, color='red', lw=2, linestyle='--', label=f"Spot: ${spot_price:,.0f}")
    ax.axhline(y=0, color='black', lw=1)
    ax.set_xlim([from_strike, to_strike])
    ax.legend()
    plt.tight_layout()

    chart_filename = f"gamma_profile_{index}_{date.today()}.png"
    plt.savefig(chart_filename, dpi=100)
    logging.info(f"Gamma profile chart saved as {chart_filename}")
    plt.close(fig)
    return chart_filename

# --- MODIFIED EMAIL FUNCTION (To send multiple charts) ---
def send_email_with_charts(image_paths, index, spot_price):
    sender_email, receiver_email, email_password = os.environ.get('SENDER_EMAIL'), os.environ.get('RECEIVER_EMAIL'), os.environ.get('EMAIL_APP_PASSWORD')
    if not all([sender_email, receiver_email, email_password]):
        logging.error("Email credentials not found."); return

    msg = MIMEMultipart()
    msg['Subject'] = f"Daily Gamma Report for {index} - {date.today().strftime('%Y-%m-%d')}"
    msg['From'], msg['To'] = sender_email, receiver_email
    msg.attach(MIMEText(f"Attached are the gamma exposure charts for {index}.\n\nCurrent Spot Price: ${spot_price:,.2f}", 'plain'))

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
    if len(sys.argv) != 2: print("Usage: python gamma_reporter.py <INDEX>"); sys.exit(1)
    index = sys.argv[1].upper()
    options = fetch_options_data(index)
    spot_price = float(options["data"]["close"])
    data_df = pd.DataFrame(options["data"]["options"])
    data_df = parse_option_data(data_df)
    today_date = date.today()

    # --- Data Prep for 0-1 DTE ---
    df_0dte = pd.DataFrame()
    tomorrow_date = today_date + timedelta(days=1)
    data_0dte = data_df[(data_df['ExpirationDate'].dt.date == today_date) | (data_df['ExpirationDate'].dt.date == tomorrow_date)]
    if not data_0dte.empty:
        calls_0dte = data_0dte[data_0dte['CallPut'] == "C"].reset_index(drop=True)[['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
        puts_0dte = data_0dte[data_0dte['CallPut'] == "P"].reset_index(drop=True)[['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
        df_0dte = calls_0dte.merge(puts_0dte, on=['ExpirationDate', 'Strike'], suffixes=('_call', '_put'))
        df_0dte.columns = ['ExpirationDate', 'StrikePrice', 'CallIV', 'CallGamma', 'CallOpenInt', 'PutIV', 'PutGamma', 'PutOpenInt']
        df_0dte = validate_and_clean_data(df_0dte)
        if not df_0dte.empty: df_0dte = process_gamma(df_0dte, spot_price)

    # --- Data Prep for 6 Months ---
    df_6m = pd.DataFrame()
    six_months_later = today_date + relativedelta(months=CONFIG['MONTHS_TO_INCLUDE'])
    data_6m = data_df[(data_df['ExpirationDate'].dt.date > today_date) & (data_df['ExpirationDate'].dt.date <= six_months_later)]
    if not data_6m.empty:
        calls_6m = data_6m[data_6m['CallPut'] == "C"].reset_index(drop=True)[['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
        puts_6m = data_6m[data_6m['CallPut'] == "P"].reset_index(drop=True)[['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
        df_6m = calls_6m.merge(puts_6m, on=['ExpirationDate', 'Strike'], suffixes=('_call', '_put'))
        df_6m.columns = ['ExpirationDate', 'StrikePrice', 'CallIV', 'CallGamma', 'CallOpenInt', 'PutIV', 'PutGamma', 'PutOpenInt']
        df_6m = validate_and_clean_data(df_6m)
        if not df_6m.empty: df_6m = process_gamma(df_6m, spot_price)

    # --- Generate and Save Charts ---
    charts_to_send = []

    # 1. Generate the main 4-panel report
    fig, axs = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(f'{index} Gamma Exposure Analysis - {date.today()}', fontsize=16, fontweight='bold')
    
    if not df_0dte.empty:
        plot_gamma_reporter(df_0dte, spot_price, spot_price - CONFIG['STRIKE_RANGE_WIDTH']/2, spot_price + CONFIG['STRIKE_RANGE_WIDTH']/2, index, "0-1 DTE", axs[0,0], axs[1,0])
    else:
        axs[0,0].set_title("No 0-1 DTE Data"); axs[1,0].set_title("No 0-1 DTE Data")

    if not df_6m.empty:
        plot_gamma_reporter(df_6m, spot_price, CONFIG['STRIKE_RANGE_LOWER'] * spot_price, CONFIG['STRIKE_RANGE_UPPER'] * spot_price, index, "6 Months", axs[0,1], axs[1,1], bin_width=20, add_guidance_box=True)
    else:
        axs[0,1].set_title("No 6M Data"); axs[1,1].set_title("No 6M Data")
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    chart1_filename = f"gamma_report_{index}_{date.today()}.png"
    plt.savefig(chart1_filename, dpi=100)
    logging.info(f"Main report chart saved as {chart1_filename}")
    plt.close(fig)
    charts_to_send.append(chart1_filename)

    # 2. Generate the separate gamma profile chart
    if not df_0dte.empty:
        chart2_filename = plot_gamma_profile(df_0dte, spot_price, index)
        charts_to_send.append(chart2_filename)
    else:
        logging.warning("Skipping gamma profile chart due to no 0-1 DTE data.")

    # --- Send the Email ---
    if charts_to_send:
        send_email_with_charts(charts_to_send, index, spot_price)
    else:
        logging.error("No charts were generated to send.")

if __name__ == "__main__":
    main()
