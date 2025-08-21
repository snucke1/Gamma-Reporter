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
from scipy.interpolate import interp1d  # Add this for smooth interpolation

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

# --- NEW FUNCTION: Plot Gamma Profile (for 0-1DTE only) ---
def plot_gamma_profile(df, spot_price, from_strike, to_strike, index, title_prefix, ax, bin_width=None):
    """
    Plots a gamma profile showing the distribution and cumulative gamma exposure
    Specifically designed for 0-1DTE options
    """
    if bin_width is not None:
        df = df.copy()
        df['StrikeBin'] = (df['StrikePrice'] // bin_width) * bin_width
        group_col = 'StrikeBin'
    else:
        group_col = 'StrikePrice'
    
    df_agg = df.groupby([group_col]).sum(numeric_only=True)
    strikes = df_agg.index.values
    
    # Calculate total gamma at each strike
    total_gamma = df_agg['TotalGamma'].values
    
    # Sort strikes and gamma for proper plotting
    sorted_indices = np.argsort(strikes)
    strikes_sorted = strikes[sorted_indices]
    gamma_sorted = total_gamma[sorted_indices]
    
    # Calculate cumulative gamma
    cumulative_gamma = np.cumsum(gamma_sorted)
    
    # Create smooth interpolation for the gamma profile
    if len(strikes_sorted) > 3:
        # Create a finer grid for smooth plotting
        strike_range = np.linspace(strikes_sorted.min(), strikes_sorted.max(), 200)
        
        # Interpolate gamma values
        f_gamma = interp1d(strikes_sorted, gamma_sorted, kind='cubic', fill_value='extrapolate')
        smooth_gamma = f_gamma(strike_range)
        
        # Plot the smooth gamma profile
        ax.fill_between(strike_range, 0, smooth_gamma, 
                        where=(smooth_gamma >= 0), 
                        color='green', alpha=0.3, label='Positive Gamma')
        ax.fill_between(strike_range, 0, smooth_gamma, 
                        where=(smooth_gamma < 0), 
                        color='red', alpha=0.3, label='Negative Gamma')
        ax.plot(strike_range, smooth_gamma, color='darkblue', linewidth=2.5, label='Gamma Profile')
    else:
        # If not enough points for smooth interpolation, use simple plot
        ax.fill_between(strikes_sorted, 0, gamma_sorted, 
                        where=(gamma_sorted >= 0), 
                        color='green', alpha=0.3, label='Positive Gamma')
        ax.fill_between(strikes_sorted, 0, gamma_sorted, 
                        where=(gamma_sorted < 0), 
                        color='red', alpha=0.3, label='Negative Gamma')
        ax.plot(strikes_sorted, gamma_sorted, color='darkblue', linewidth=2.5, label='Gamma Profile')
    
    # Add cumulative gamma on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(strikes_sorted, cumulative_gamma, color='orange', linewidth=2, 
             linestyle='--', label='Cumulative Gamma', alpha=0.7)
    ax2.set_ylabel('Cumulative Gamma ($B)', fontweight="bold", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # Formatting
    ax.grid(True, alpha=0.3)
    ax.set_xlim([from_strike, to_strike])
    ax.set_title(f"{title_prefix} Gamma Profile (Near-Term Focus)", fontweight="bold", fontsize=14)
    ax.set_xlabel('Strike Price', fontweight="bold")
    ax.set_ylabel('Gamma Exposure ($B/1% move)', fontweight="bold")
    
    # Add spot price line
    ax.axvline(x=spot_price, color='red', lw=2, linestyle='--', label=f"{index} Spot: ${spot_price:,.0f}")
    
    # Add zero line
    ax.axhline(y=0, color='black', lw=1, linestyle='-', alpha=0.3)
    
    # Find and mark key gamma levels
    if len(gamma_sorted) > 0:
        # Maximum absolute gamma
        max_gamma_idx = np.argmax(np.abs(gamma_sorted))
        max_gamma_strike = strikes_sorted[max_gamma_idx]
        max_gamma_value = gamma_sorted[max_gamma_idx]
        ax.plot(max_gamma_strike, max_gamma_value, 'r*', markersize=15, 
                label=f'Max Gamma: ${max_gamma_value:.2f}B @ ${max_gamma_strike:.0f}')
        
        # Find gamma flip point (where cumulative gamma crosses zero)
        if len(cumulative_gamma) > 1:
            zero_crossings = np.where(np.diff(np.sign(cumulative_gamma)))[0]
            if len(zero_crossings) > 0:
                flip_idx = zero_crossings[0]
                flip_strike = strikes_sorted[flip_idx]
                ax.axvline(x=flip_strike, color='purple', lw=1.5, linestyle=':', 
                          alpha=0.7, label=f'Gamma Flip: ${flip_strike:.0f}')
    
    # Add key statistics box
    total_gamma_sum = df['TotalGamma'].sum()
    gamma_above_spot = df[df['StrikePrice'] > spot_price]['TotalGamma'].sum()
    gamma_below_spot = df[df['StrikePrice'] < spot_price]['TotalGamma'].sum()
    
    stats_text = (f'0-1DTE Gamma Stats:\n'
                  f'─────────────────\n'
                  f'Total: ${total_gamma_sum:.2f}B\n'
                  f'Above Spot: ${gamma_above_spot:.2f}B\n'
                  f'Below Spot: ${gamma_below_spot:.2f}B')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, 
            fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', 
            facecolor='lightyellow', alpha=0.8), verticalalignment='top')
    
    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

# --- EXISTING PLOTTING FUNCTION (No changes) ---
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
        ax1.text(0.02, 0.98, guidance_text, transform=ax1.transAxes, fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.4), verticalalignment='top')
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

# --- Email Function (No changes here) ---
def send_email_with_chart(image_path, index, spot_price):
    sender_email = os.environ.get('SENDER_EMAIL')
    receiver_email = os.environ.get('RECEIVER_EMAIL')
    email_password = os.environ.get('EMAIL_APP_PASSWORD')

    if not all([sender_email, receiver_email, email_password]):
        logging.error("Email credentials not found in environment variables.")
        return

    msg = MIMEMultipart()
    today_str = date.today().strftime("%Y-%m-%d")
    msg['Subject'] = f"Gamma Exposure Report for {index} - {today_str}"
    msg['From'] = sender_email
    msg['To'] = receiver_email

    body = f"Attached is the gamma exposure chart for {index}.\n\nCurrent Spot Price: ${spot_price:,.2f}"
    msg.attach(MIMEText(body, 'plain'))

    with open(image_path, 'rb') as fp:
        img = MIMEImage(fp.read())
        img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
        msg.attach(img)

    try:
        logging.info(f"Sending email to {receiver_email}...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, email_password)
            server.send_message(msg)
        logging.info("Email sent successfully!")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

# --- MODIFIED: Main Execution Block with Gamma Profile for 0-1DTE only ---
def main():
    if len(sys.argv) != 2:
        print("Usage: python gamma_reporter.py <INDEX>")
        sys.exit(1)
    index = sys.argv[1].upper()
    options = fetch_options_data(index)
    spot_price = float(options["data"]["close"])
    data_df = pd.DataFrame(options["data"]["options"])
    data_df = parse_option_data(data_df)

    # --- MODIFIED: 0DTE and 1DTE (0-1 DTE) ---
    today_date = date.today()
    tomorrow_date = today_date + timedelta(days=1)  # Add tomorrow's date
    
    half_range = CONFIG['STRIKE_RANGE_WIDTH'] / 2
    from_strike_0dte, to_strike_0dte = spot_price - half_range, spot_price + half_range
    otm_lower_bound = spot_price * (1 - CONFIG['OTM_FILTER_PERCENT'])
    otm_upper_bound = spot_price * (1 + CONFIG['OTM_FILTER_PERCENT'])
    
    # Filter for both today AND tomorrow expiring options
    data_0dte = data_df[(data_df['ExpirationDate'].dt.date == today_date) | 
                        (data_df['ExpirationDate'].dt.date == tomorrow_date)]
    
    # Log the number of options found for each expiry
    today_count = len(data_df[data_df['ExpirationDate'].dt.date == today_date])
    tomorrow_count = len(data_df[data_df['ExpirationDate'].dt.date == tomorrow_date])
    logging.info(f"Found {today_count} options expiring today and {tomorrow_count} options expiring tomorrow")
    
    data_0dte = data_0dte[(data_0dte['Strike'] >= otm_lower_bound) & (data_0dte['Strike'] <= otm_upper_bound)]
    calls_0dte = data_0dte[data_0dte['CallPut'] == "C"].reset_index(drop=True)[['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
    puts_0dte = data_0dte[data_0dte['CallPut'] == "P"].reset_index(drop=True)[['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
    df_0dte = calls_0dte.merge(puts_0dte, on=['ExpirationDate', 'Strike'], suffixes=('_call', '_put'))
    df_0dte.columns = ['ExpirationDate', 'StrikePrice', 'CallIV', 'CallGamma', 'CallOpenInt', 'PutIV', 'PutGamma', 'PutOpenInt']
    df_0dte['ExpirationDate'] += timedelta(hours=16)
    df_0dte = validate_and_clean_data(df_0dte)
    if len(df_0dte) == 0: logging.warning("No valid 0-1DTE options data after cleaning")
    else: df_0dte = process_gamma(df_0dte, spot_price)

    # --- MODIFIED: 6 Months (excluding today's options) ---
    six_months_later = today_date + relativedelta(months=CONFIG['MONTHS_TO_INCLUDE'])
    from_strike_6m, to_strike_6m = CONFIG['STRIKE_RANGE_LOWER'] * spot_price, CONFIG['STRIKE_RANGE_UPPER'] * spot_price
    
    # Exclude today's expiring options, start from tomorrow
    data_6m = data_df[(data_df['ExpirationDate'].dt.date > today_date) & 
                      (data_df['ExpirationDate'].dt.date <= six_months_later)]
    
    # Log the date range for 6M options
    if len(data_6m) > 0:
        earliest_expiry = data_6m['ExpirationDate'].dt.date.min()
        latest_expiry = data_6m['ExpirationDate'].dt.date.max()
        logging.info(f"6M options date range: {earliest_expiry} to {latest_expiry} (excluding today)")
    
    calls_6m = data_6m[data_6m['CallPut'] == "C"].reset_index(drop=True)[['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
    puts_6m = data_6m[data_6m['CallPut'] == "P"].reset_index(drop=True)[['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
    df_6m = calls_6m.merge(puts_6m, on=['ExpirationDate', 'Strike'], suffixes=('_call', '_put'))
    df_6m.columns = ['ExpirationDate', 'StrikePrice', 'CallIV', 'CallGamma', 'CallOpenInt', 'PutIV', 'PutGamma', 'PutOpenInt']
    df_6m['ExpirationDate'] += timedelta(hours=16)
    df_6m = validate_and_clean_data(df_6m)
    if len(df_6m) == 0: logging.warning("No valid 6M options data after cleaning")
    else: df_6m = process_gamma(df_6m, spot_price)

    # --- MODIFIED: Create custom layout with GridSpec for better control ---
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(20, 18))
    gs = GridSpec(5, 2, height_ratios=[1, 1, 0.8, 1, 1], hspace=0.3)
    
    # Create axes for plots
    ax1 = fig.add_subplot(gs[0, 0])  # 0-1DTE Total Gamma
    ax2 = fig.add_subplot(gs[1, 0])  # 0-1DTE by Option Type
    ax3 = fig.add_subplot(gs[2, :])  # Gamma Profile (spans both columns)
    ax4 = fig.add_subplot(gs[3, 1])  # 6M Total Gamma
    ax5 = fig.add_subplot(gs[4, 1])  # 6M by Option Type
    
    # 0-1DTE plots (left column, rows 1-2)
    if len(df_0dte) > 0: 
        plot_gamma(df_0dte, spot_price, from_strike_0dte, to_strike_0dte, index, "0-1DTE", ax1, ax2)
        plot_gamma_profile(df_0dte, spot_price, from_strike_0dte, to_strike_0dte, index, "0-1DTE", ax3)
    else: 
        ax1.set_title("No valid 0-1DTE data")
        ax2.set_title("No valid 0-1DTE data")
        ax3.set_title("No valid 0-1DTE gamma profile data")
    
    # 6M plots (right column, rows 4-5)
    if len(df_6m) > 0: 
        plot_gamma(df_6m, spot_price, from_strike_6m, to_strike_6m, index, "6M (excl. today)", ax4, ax5, bin_width=20, add_guidance_box=True)
    else: 
        ax4.set_title("No valid 6M data")
        ax5.set_title("No valid 6M data")
    
    plt.suptitle(f'{index} Gamma Exposure Analysis - {date.today()}', fontsize=16, fontweight='bold', y=0.995)
    
    chart_filename = f"gamma_report_{index}_{date.today()}.png"
    plt.savefig(chart_filename, dpi=100, bbox_inches='tight')  # Added bbox_inches for better layout
    logging.info(f"Chart saved as {chart_filename}")
    
    # Send the email
    send_email_with_chart(chart_filename, index, spot_price)

if __name__ == "__main__":
    main()
