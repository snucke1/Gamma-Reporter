# Save this file as gamma_reporter.py

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
from dateutil.relativedelta import relativedelta
import requests
import sys
import logging
import warnings
import os  # For environment variables
import smtplib  # For sending email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration constants
CONFIG = {
    'STRIKE_RANGE_WIDTH': 500,  # For 0DTE
    'OTM_FILTER_PERCENT': 0.02,  # For 0DTE
    'STRIKE_RANGE_LOWER': 0.8,   # For 6M and profile
    'STRIKE_RANGE_UPPER': 1.2,   # For 6M and profile
    'CONTRACT_SIZE': 100,
    'PERCENT_MOVE': 0.01,
    'MONTHS_TO_INCLUDE': 6,
    'BILLION': 10**9,
    'TRADING_DAYS_PER_YEAR': 252,
    'RISK_FREE_RATE': 0.05,
    'DIVIDEND_YIELD': 0.02,
    'GR_HIGH_THRESHOLD': 1.25,
    'GR_LOW_THRESHOLD': 0.75
}

pd.options.display.float_format = '{:,.4f}'.format

def calcGammaEx(S, K, vol, T, r, q, OI):
    """
    Calculate Black-Scholes gamma exposure for options.
    Note: Gamma is the same for calls and puts.
    """
    if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
        return 0
    
    try:
        dp = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        gamma = np.exp(-q * T) * norm.pdf(dp) / (S * vol * np.sqrt(T))
        # Convert to dollar gamma per 1% move
        return OI * CONFIG['CONTRACT_SIZE'] * S * S * CONFIG['PERCENT_MOVE'] * gamma
    except (ValueError, ZeroDivisionError):
        return 0

def parse_option_data(data_df):
    """Parse option data from raw dataframe"""
    data_df['CallPut'] = data_df['option'].str.slice(start=-9, stop=-8)
    data_df['ExpirationDate'] = pd.to_datetime(data_df['option'].str.slice(start=-15, stop=-9), 
                                               format='%y%m%d', errors='coerce')
    data_df = data_df.dropna(subset=['ExpirationDate'])
    
    data_df['Strike'] = pd.to_numeric(data_df['option'].str.slice(start=-8, stop=-3), errors='coerce')
    data_df = data_df.dropna(subset=['Strike'])
    
    return data_df

def validate_and_clean_data(df, for_profile=False):
    """Validate and clean option data"""
    if for_profile:
        numeric_columns = ['CallIV', 'PutIV', 'CallOpenInt', 'PutOpenInt', 'StrikePrice']
    else:
        numeric_columns = ['CallIV', 'PutIV', 'CallGamma', 'PutGamma', 'CallOpenInt', 'PutOpenInt', 'StrikePrice']
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=numeric_columns)
    df = df[df['StrikePrice'] > 0]
    df = df[(df['CallIV'] > 0) & (df['PutIV'] > 0)]
    df = df[(df['CallOpenInt'] >= 0) & (df['PutOpenInt'] >= 0)]
    df = df[(df['CallIV'] < 5) & (df['PutIV'] < 5)]  # Remove IVs > 500%
    
    return df

def fetch_options_data(index):
    """Fetch options data with error handling"""
    try:
        url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{index}.json"
        logging.info(f"Fetching data from: {url}")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        options = response.json()
        
        if 'data' not in options or 'options' not in options['data']:
            raise ValueError("Unexpected data format received from API")
            
        return options
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data: {e}")
        sys.exit(1)
    except ValueError as e:
        logging.error(f"Data format error: {e}")
        sys.exit(1)

def calculate_gamma_profile(df, from_strike, to_strike):
    """Calculate gamma exposure profile across different spot levels"""
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
        
        # Puts have negative gamma exposure from dealers' perspective (short puts)
        total = (call_gamma.sum() - put_gamma.sum()) / CONFIG['BILLION']
        total_gamma.append(total)
    
    return levels, total_gamma

def find_gamma_flip_points(levels, total_gamma):
    """Find points where gamma crosses zero using linear interpolation"""
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

def process_gamma(df, spot_price):
    """Process gamma for bar chart plotting"""
    df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * CONFIG['CONTRACT_SIZE'] * spot_price * spot_price * CONFIG['PERCENT_MOVE'] * 0.5
    df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * CONFIG['CONTRACT_SIZE'] * spot_price * spot_price * CONFIG['PERCENT_MOVE'] * -1 * 0.5
    df['TotalGamma'] = (df['CallGEX'] + df['PutGEX']) / CONFIG['BILLION']
    return df

def plot_gamma_bars(df, spot_price, from_strike, to_strike, index, title_prefix, ax1, ax2, bin_width=None, add_guidance_box=False, show_50_ticks=False):
    """Plot gamma exposure as bar charts"""
    # Binning for 6M plot
    if bin_width is not None:
        df = df.copy()
        df['StrikeBin'] = (df['StrikePrice'] // bin_width) * bin_width
        group_col = 'StrikeBin'
    else:
        group_col = 'StrikePrice'

    df_agg = df.groupby([group_col]).sum(numeric_only=True)
    strikes = df_agg.index.values

    if len(strikes) > 1:
        if bin_width is not None:
            bar_width = bin_width
        else:
            strike_diffs = np.diff(sorted(strikes))
            bar_width = np.median(strike_diffs) * 0.8
            max_width = np.min(strike_diffs) * 0.9 if len(strike_diffs) > 0 else 10
            bar_width = min(bar_width, max_width, 10)
    else:
        bar_width = bin_width if bin_width is not None else 5

    total_call_gamma = df_agg['CallGEX'].sum()
    total_put_gamma = abs(df_agg['PutGEX'].sum())
    gamma_ratio = total_call_gamma / total_put_gamma if total_put_gamma != 0 else np.nan
    total_call_oi = df_agg['CallOpenInt'].sum()
    total_put_oi = df_agg['PutOpenInt'].sum()
    put_call_ratio = total_put_oi / total_call_oi if total_call_oi != 0 else np.nan

    # Net Gamma
    ax1.grid(True, alpha=0.3)
    ax1.bar(strikes, df_agg['TotalGamma'].values, width=bar_width, linewidth=1.5 if bin_width else 0.5,
            edgecolor='k', color='steelblue', alpha=0.7, label="Net Gamma")
    ax1.set_xlim([from_strike, to_strike])
    
    # Set x-axis ticks every 50 points for 0-1DTE only
    if show_50_ticks:
        tick_start = int((from_strike // 50) * 50)
        tick_end = int(((to_strike // 50) + 1) * 50)
        ax1.set_xticks(np.arange(tick_start, tick_end, 50))
    
    ax1.set_title(f"{title_prefix} Total Gamma: ${df['TotalGamma'].sum():.2f} Bn per 1% {index} Move",
                  fontweight="bold", fontsize=14)
    ax1.set_xlabel('Strike Price', fontweight="bold")
    ax1.set_ylabel('Spot Gamma Exposure ($B/1% move)', fontweight="bold")
    ax1.axvline(x=spot_price, color='red', lw=2, linestyle='--',
                label=f"{index} Spot: ${spot_price:,.0f}")
    ax1.legend()

    # Add guidance text box based on Gamma Ratio
    if add_guidance_box and not np.isnan(gamma_ratio):
        high_gr_text = (
            f"GR > {CONFIG['GR_HIGH_THRESHOLD']} (Currently: {gamma_ratio:.2f}) - High Positive Gamma\n"
            "-----------------------------------------------------------\n"
            "Expect: Tight ranges, dip-buying, upward drift\n"
            "(especially inside 2 weeks to OPEX).\n\n"
            "Trade Ideas:\n"
            " • If mkt opens down & momentum turns up:\n"
            "   Buy ATM/SOTM calls/call spreads or sell ATM put spreads.\n"
            " • If mkt opens higher: Tougher trade, low R/R.\n"
            "   Could sell OTM puts/spreads.\n"
        )
        low_gr_text = (
            f"GR < {CONFIG['GR_LOW_THRESHOLD']} (Currently: {gamma_ratio:.2f}) - Low/Negative Gamma\n"
            "-----------------------------------------------------------\n"
            "Expect: Widening ranges, rally-selling, downside bias.\n\n"
            "Trade Ideas:\n"
            " • If mkt opens high & momentum turns down:\n"
            "   Sell calls/spreads or buy puts/spreads.\n"
            " • If mkt opens down & momentum turns up:\n"
            "   Consider opposite trades.\n"
        )
        neutral_gr_text = (
            f"{CONFIG['GR_LOW_THRESHOLD']} ≤ GR ≤ {CONFIG['GR_HIGH_THRESHOLD']} (Currently: {gamma_ratio:.2f}) - Balanced Gamma\n"
            "-----------------------------------------------------------\n"
            "Expect: Choppy or range-bound market. No strong directional\n"
            "pressure from dealer hedging.\n"
        )
        always_on_text = (
            "\nGeneral Principles:\n"
            "-----------------------------------------------------------\n"
            "4. Tight Risk: If a trade thesis is falsified, exit promptly.\n"
            "5. Position Mgt: On winning trades, consider selling a portion\n"
            "   to create a 'free trade' and let the rest run."
        )

        if gamma_ratio > CONFIG['GR_HIGH_THRESHOLD']:
            guidance_text = high_gr_text + always_on_text
        elif gamma_ratio < CONFIG['GR_LOW_THRESHOLD']:
            guidance_text = low_gr_text + always_on_text
        else:
            guidance_text = neutral_gr_text + always_on_text

        ax1.text(0.02, 0.98, guidance_text, transform=ax1.transAxes,
                 fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.4),
                 verticalalignment='top')

    # By Type
    ax2.grid(True, alpha=0.3)
    ax2.bar(strikes, df_agg['CallGEX'].values / CONFIG['BILLION'],
            width=bar_width, linewidth=1.5 if bin_width else 0.5, edgecolor='k',
            color='green', alpha=0.7, label="Call Gamma")
    ax2.bar(strikes, df_agg['PutGEX'].values / CONFIG['BILLION'],
            width=bar_width, linewidth=1.5 if bin_width else 0.5, edgecolor='k',
            color='red', alpha=0.7, label="Put Gamma")
    ax2.set_xlim([from_strike, to_strike])
    ax2.set_title(f"{title_prefix} Gamma Exposure by Option Type", fontweight="bold", fontsize=14)
    ax2.set_xlabel('Strike Price', fontweight="bold")
    ax2.set_ylabel('Spot Gamma Exposure ($B/1% move)', fontweight="bold")
    ax2.axvline(x=spot_price, color='red', lw=2, linestyle='--',
                label=f"{index} Spot: ${spot_price:,.0f}")

    metrics_text = (f'Gamma Ratio (Call/Put): {gamma_ratio:.2f}\n'
                   f'Put/Call OI Ratio: {put_call_ratio:.2f}\n'
                   f'Total Call Gamma: ${total_call_gamma/CONFIG["BILLION"]:.2f}B\n'
                   f'Total Put Gamma: ${total_put_gamma/CONFIG["BILLION"]:.2f}B')
    ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes,
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
             verticalalignment='top')
    ax2.legend()

def plot_gamma_profile(levels, total_gamma, spot_price, index, flip_points):
    """Create gamma profile plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.grid(True, alpha=0.3)
    
    # Plot gamma profile
    ax.plot(levels, total_gamma, 'b-', linewidth=2, label="Total Gamma (0 & 1 DTE)")
    
    today_str = date.today().strftime('%d %b %Y')
    ax.set_title(f"Gamma Profile (0-DTE & 1-DTE) - {index} - {today_str}", fontweight="bold", fontsize=16)
    ax.set_xlabel('Index Price', fontweight="bold")
    ax.set_ylabel('Gamma Exposure ($ billions / 1% move)', fontweight="bold")
    
    # Add zero and spot price lines
    ax.axhline(y=0, color='black', lw=1)
    ax.axvline(x=spot_price, color='red', lw=2, linestyle='--', label=f"{index} Spot: ${spot_price:,.0f}")
    
    # Mark gamma flip points
    for i, flip in enumerate(flip_points):
        if levels[0] <= flip <= levels[-1]:
            ax.axvline(x=flip, color='green', lw=1.5, linestyle=':', 
                       label=f"Gamma Flip: ${flip:,.0f}" if i == 0 else "")
    
    # Add shading for negative/positive gamma regions
    ax.fill_between(levels, total_gamma, 0, where=np.array(total_gamma) > 0, facecolor='green', alpha=0.1)
    ax.fill_between(levels, total_gamma, 0, where=np.array(total_gamma) < 0, facecolor='red', alpha=0.1)
    
    ax.set_xlim([levels[0], levels[-1]])
    ax.legend(loc='best')
    
    plt.tight_layout()
    return fig

def send_email_with_charts(chart_paths, index, spot_price):
    """Send email with multiple chart attachments"""
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

    body = f"""Attached are the gamma exposure charts for {index}.

Current Spot Price: ${spot_price:,.2f}

Charts included:
1. Gamma Exposure Bar Charts (0-1DTE and 6M)
2. Gamma Profile (0-DTE & 1-DTE)
"""
    msg.attach(MIMEText(body, 'plain'))

    # Attach all charts
    for image_path in chart_paths:
        if os.path.exists(image_path):
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

def main():
    """Main execution function"""
    if len(sys.argv) != 2:
        print("Usage: python gamma_reporter.py <INDEX>")
        print("Example: python gamma_reporter.py SPX")
        sys.exit(1)
    
    index = sys.argv[1].upper()
    logging.info(f"Analyzing gamma exposure for {index}")
    
    # Fetch data once
    options = fetch_options_data(index)
    spot_price = float(options["data"]["close"])
    logging.info(f"Current Spot Price: {spot_price:,.2f}")
    
    # Parse the raw data
    data_df = pd.DataFrame(options["data"]["options"])
    data_df = parse_option_data(data_df)
    
    today_date = date.today()
    tomorrow_date = today_date + timedelta(days=1)
    
    # ===== FIRST PLOT: Gamma Exposure Bar Charts (0-1DTE and 6M) =====
    
    # --- 0-1DTE Processing ---
    half_range = CONFIG['STRIKE_RANGE_WIDTH'] / 2
    from_strike_0dte = spot_price - half_range
    to_strike_0dte = spot_price + half_range
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
    
    calls_0dte = data_0dte[data_0dte['CallPut'] == "C"].reset_index(drop=True)
    puts_0dte = data_0dte[data_0dte['CallPut'] == "P"].reset_index(drop=True)
    
    calls_0dte = calls_0dte[['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
    puts_0dte = puts_0dte[['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
    
    df_0dte = calls_0dte.merge(puts_0dte, on=['ExpirationDate', 'Strike'], suffixes=('_call', '_put'))
    df_0dte.columns = ['ExpirationDate', 'StrikePrice', 'CallIV', 'CallGamma', 'CallOpenInt',
                       'PutIV', 'PutGamma', 'PutOpenInt']
    df_0dte['ExpirationDate'] = df_0dte['ExpirationDate'] + timedelta(hours=16)
    df_0dte = validate_and_clean_data(df_0dte)
    
    if len(df_0dte) > 0:
        df_0dte = process_gamma(df_0dte, spot_price)
        logging.info(f"Valid 0-1DTE options after cleaning: {len(df_0dte)}")
    else:
        logging.warning("No valid 0-1DTE options data after cleaning")
    
    # --- 6 Months Processing (excluding today) ---
    six_months_later = today_date + relativedelta(months=CONFIG['MONTHS_TO_INCLUDE'])
    from_strike_6m = CONFIG['STRIKE_RANGE_LOWER'] * spot_price
    to_strike_6m = CONFIG['STRIKE_RANGE_UPPER'] * spot_price
    
    # Exclude today's expiring options, start from tomorrow
    data_6m = data_df[(data_df['ExpirationDate'].dt.date > today_date) & 
                      (data_df['ExpirationDate'].dt.date <= six_months_later)]
    
    # Log the date range for 6M options
    if len(data_6m) > 0:
        earliest_expiry = data_6m['ExpirationDate'].dt.date.min()
        latest_expiry = data_6m['ExpirationDate'].dt.date.max()
        logging.info(f"6M options date range: {earliest_expiry} to {latest_expiry} (excluding today)")
    
    calls_6m = data_6m[data_6m['CallPut'] == "C"].reset_index(drop=True)
    puts_6m = data_6m[data_6m['CallPut'] == "P"].reset_index(drop=True)
    
    calls_6m = calls_6m[['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
    puts_6m = puts_6m[['ExpirationDate', 'Strike', 'iv', 'gamma', 'open_interest']]
    
    df_6m = calls_6m.merge(puts_6m, on=['ExpirationDate', 'Strike'], suffixes=('_call', '_put'))
    df_6m.columns = ['ExpirationDate', 'StrikePrice', 'CallIV', 'CallGamma', 'CallOpenInt',
                     'PutIV', 'PutGamma', 'PutOpenInt']
    df_6m['ExpirationDate'] = df_6m['ExpirationDate'] + timedelta(hours=16)
    df_6m = validate_and_clean_data(df_6m)
    
    if len(df_6m) > 0:
        df_6m = process_gamma(df_6m, spot_price)
        logging.info(f"Valid 6M options after cleaning: {len(df_6m)}")
    else:
        logging.warning("No valid 6M options data after cleaning")
    
    # Create first plot for bar charts
    fig1, axs = plt.subplots(2, 2, figsize=(20, 14))
    fig1.suptitle(f'Gamma Exposure Analysis - {index}', fontsize=16, fontweight='bold')
    
    # Plot 0-1DTE (left column) - WITH 50-tick intervals
    if len(df_0dte) > 0:
        plot_gamma_bars(df_0dte, spot_price, from_strike_0dte, to_strike_0dte, index, "0-1DTE", 
                       axs[0,0], axs[1,0], bin_width=None, show_50_ticks=True)
    else:
        axs[0,0].set_title("No valid 0-1DTE data")
        axs[1,0].set_title("No valid 0-1DTE data")
    
    # Plot 6M (right column) - WITHOUT 50-tick intervals (default behavior)
    if len(df_6m) > 0:
        plot_gamma_bars(df_6m, spot_price, from_strike_6m, to_strike_6m, index, "6M (excl. today)", 
                       axs[0,1], axs[1,1], bin_width=20, add_guidance_box=True, show_50_ticks=False)
    else:
        axs[0,1].set_title("No valid 6M data")
        axs[1,1].set_title("No valid 6M data")
    
    plt.tight_layout()
    
    # Save bar chart
    chart1_filename = f"gamma_bars_{index}_{date.today()}.png"
    fig1.savefig(chart1_filename)
    logging.info(f"Bar chart saved as {chart1_filename}")
    
    # ===== SECOND PLOT: Gamma Profile =====
    
    # Get 0-DTE and 1-DTE for profile
    unique_expiries = sorted(data_df['ExpirationDate'].unique())
    chart2_filename = None
    
    if len(unique_expiries) >= 2:
        target_expiries = unique_expiries[:2]  # Get first two expiration dates
        dte_0_str = pd.to_datetime(target_expiries[0]).strftime('%Y-%m-%d')
        dte_1_str = pd.to_datetime(target_expiries[1]).strftime('%Y-%m-%d')
        logging.info(f"Creating profile for 0-DTE ({dte_0_str}) and 1-DTE ({dte_1_str}) expirations.")
        
        data_profile = data_df[data_df['ExpirationDate'].isin(target_expiries)]
        
        # Separate calls and puts for profile
        calls_profile = data_profile[data_profile['CallPut'] == "C"][['ExpirationDate', 'Strike', 'iv', 'open_interest']]
        puts_profile = data_profile[data_profile['CallPut'] == "P"][['ExpirationDate', 'Strike', 'iv', 'open_interest']]
        
        df_profile = calls_profile.merge(puts_profile, on=['ExpirationDate', 'Strike'], suffixes=('_call', '_put'))
        df_profile.columns = ['ExpirationDate', 'StrikePrice', 'CallIV', 'CallOpenInt', 'PutIV', 'PutOpenInt']
        
        # Add time component for accurate day count
        df_profile['ExpirationDate'] = df_profile['ExpirationDate'] + timedelta(hours=16)
        
        df_profile = validate_and_clean_data(df_profile, for_profile=True)
        logging.info(f"Valid options for profile after cleaning: {len(df_profile)}")
        
        if not df_profile.empty:
            # Calculate profile
            from_strike_profile = CONFIG['STRIKE_RANGE_LOWER'] * spot_price
            to_strike_profile = CONFIG['STRIKE_RANGE_UPPER'] * spot_price
            
            levels, total_gamma = calculate_gamma_profile(df_profile, from_strike_profile, to_strike_profile)
            flip_points = find_gamma_flip_points(levels, total_gamma)
            
            # Create profile plot
            fig2 = plot_gamma_profile(levels, total_gamma, spot_price, index, flip_points)
            
            # Save profile chart
            chart2_filename = f"gamma_profile_{index}_{date.today()}.png"
            fig2.savefig(chart2_filename)
            logging.info(f"Profile chart saved as {chart2_filename}")
            
            # Print summary statistics
            spot_gamma_val = np.interp(spot_price, levels, total_gamma)
            logging.info("\n=== GAMMA PROFILE SUMMARY (0 & 1 DTE) ===")
            logging.info(f"Gamma at Spot ({spot_price:,.0f}): ${spot_gamma_val:.2f}B per 1% move")
            if flip_points:
                logging.info(f"Primary Gamma Flip Point: ${flip_points[0]:,.0f}")
        else:
            logging.error("No valid options data found for profile.")
    else:
        logging.error("Not enough expiration dates available for profile analysis.")
    
    # Close all plots to free memory
    plt.close('all')
    
    # Send email with both charts
    charts_to_send = [chart1_filename]
    if chart2_filename:
        charts_to_send.append(chart2_filename)
    
    send_email_with_charts(charts_to_send, index, spot_price)

if __name__ == "__main__":
    main()
