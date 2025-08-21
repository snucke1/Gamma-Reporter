# Modified gamma_reporter.py - Add this function after the existing plot_gamma function

def create_gamma_profile_plot(df, spot_price, index):
    """Create gamma profile plot for 0-1 DTE options"""
    from scipy.stats import norm
    
    # Define strike range for profile
    from_strike = 0.9 * spot_price
    to_strike = 1.1 * spot_price
    levels = np.linspace(from_strike, to_strike, 100)
    
    # Calculate days to expiration
    today_date = date.today()
    df['daysTillExp'] = df['ExpirationDate'].apply(
        lambda x: max(1/CONFIG['TRADING_DAYS_PER_YEAR'], 
                     np.busday_count(today_date, x.date())/CONFIG['TRADING_DAYS_PER_YEAR'])
    )
    
    def calcGammaEx(S, K, vol, T, r, q, OI):
        """Calculate Black-Scholes gamma exposure"""
        if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            dp = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
            gamma = np.exp(-q*T) * norm.pdf(dp) / (S * vol * np.sqrt(T))
            return OI * CONFIG['CONTRACT_SIZE'] * S * S * CONFIG['PERCENT_MOVE'] * gamma
        except:
            return 0
    
    total_gamma = []
    r = CONFIG['RISK_FREE_RATE']
    q = CONFIG['DIVIDEND_YIELD']
    
    for level in levels:
        call_gamma = df.apply(
            lambda row: calcGammaEx(level, row['StrikePrice'], row['CallIV'], 
                                  row['daysTillExp'], r, q, row['CallOpenInt']), 
            axis=1
        )
        put_gamma = df.apply(
            lambda row: calcGammaEx(level, row['StrikePrice'], row['PutIV'], 
                                  row['daysTillExp'], r, q, row['PutOpenInt']), 
            axis=1
        )
        total = (call_gamma.sum() - put_gamma.sum()) / CONFIG['BILLION']
        total_gamma.append(total)
    
    # Find gamma flip points
    flip_points = []
    zero_crossings = np.where(np.diff(np.sign(total_gamma)))[0]
    for idx in zero_crossings:
        if idx < len(levels) - 1:
            x1, x2 = levels[idx], levels[idx + 1]
            y1, y2 = total_gamma[idx], total_gamma[idx + 1]
            if y2 != y1:
                zero_point = x1 - y1 * (x2 - x1) / (y2 - y1)
                flip_points.append(zero_point)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.grid(True, alpha=0.3)
    ax.plot(levels, total_gamma, 'b-', linewidth=2.5, label="0-1 DTE Gamma Profile")
    
    ax.set_title(f"0-1 DTE Gamma Exposure Profile - {index} - {today_date.strftime('%d %b %Y')}", 
                fontweight="bold", fontsize=16)
    ax.set_xlabel('Index Price', fontweight="bold", fontsize=12)
    ax.set_ylabel('Gamma Exposure ($ billions/1% move)', fontweight="bold", fontsize=12)
    ax.axhline(y=0, color='black', lw=1)
    ax.axvline(x=spot_price, color='red', lw=2, linestyle='--', 
              label=f"{index} Spot: ${spot_price:,.0f}")
    
    # Mark gamma flip points
    for i, flip in enumerate(flip_points):
        if from_strike <= flip <= to_strike:
            ax.axvline(x=flip, color='green', lw=1.5, linestyle=':', 
                      label=f"Gamma Flip: ${flip:,.0f}" if i == 0 else "")
    
    # Add shading for negative/positive gamma regions
    if flip_points and len(flip_points) > 0:
        main_flip = flip_points[0]
        if spot_price < main_flip:
            ax.fill_between([from_strike, main_flip], 
                          [min(total_gamma)-1, min(total_gamma)-1], 
                          [max(total_gamma)+1, max(total_gamma)+1], 
                          facecolor='red', alpha=0.1, label='Negative Gamma Zone')
            ax.fill_between([main_flip, to_strike], 
                          [min(total_gamma)-1, min(total_gamma)-1], 
                          [max(total_gamma)+1, max(total_gamma)+1], 
                          facecolor='green', alpha=0.1, label='Positive Gamma Zone')
        else:
            ax.fill_between([from_strike, main_flip], 
                          [min(total_gamma)-1, min(total_gamma)-1], 
                          [max(total_gamma)+1, max(total_gamma)+1], 
                          facecolor='green', alpha=0.1)
            ax.fill_between([main_flip, to_strike], 
                          [min(total_gamma)-1, min(total_gamma)-1], 
                          [max(total_gamma)+1, max(total_gamma)+1], 
                          facecolor='red', alpha=0.1)
    
    ax.set_xlim([from_strike, to_strike])
    ax.set_ylim([min(total_gamma)-1, max(total_gamma)+1])
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    return fig

def send_email_with_charts(image_paths, index, spot_price):
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

    body = f"Attached are the gamma exposure charts for {index}.\n\nCurrent Spot Price: ${spot_price:,.2f}\n\n"
    body += "Chart 1: Standard gamma exposure bars (0-1DTE and 6M)\n"
    body += "Chart 2: 0-1DTE gamma profile showing how gamma changes across different spot levels"
    msg.attach(MIMEText(body, 'plain'))

    # Attach multiple images
    for image_path in image_paths:
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

# Modified main() function
def main():
    if len(sys.argv) != 2:
        print("Usage: python gamma_reporter.py <INDEX>")
        sys.exit(1)
    
    index = sys.argv[1].upper()
    options = fetch_options_data(index)
    spot_price = float(options["data"]["close"])
    data_df = pd.DataFrame(options["data"]["options"])
    data_df = parse_option_data(data_df)

    # ... (existing code for processing 0-1DTE and 6M data) ...
    
    # Create the standard gamma bar charts (existing code)
    fig1, axs = plt.subplots(2, 2, figsize=(20, 14))
    
    # Plot existing charts
    if len(df_0dte) > 0: 
        plot_gamma(df_0dte, spot_price, from_strike_0dte, to_strike_0dte, index, "0-1DTE", axs[0,0], axs[1,0])
    else: 
        axs[0,0].set_title("No valid 0-1DTE data"); axs[1,0].set_title("No valid 0-1DTE data")
    
    if len(df_6m) > 0: 
        plot_gamma(df_6m, spot_price, from_strike_6m, to_strike_6m, index, "6M (excl. today)", axs[0,1], axs[1,1], bin_width=20, add_guidance_box=True)
    else: 
        axs[0,1].set_title("No valid 6M data"); axs[1,1].set_title("No valid 6M data")

    plt.tight_layout()
    chart1_filename = f"gamma_bars_{index}_{date.today()}.png"
    plt.savefig(chart1_filename)
    plt.close()
    logging.info(f"Bar chart saved as {chart1_filename}")
    
    # Create the gamma profile chart
    chart2_filename = f"gamma_profile_{index}_{date.today()}.png"
    if len(df_0dte) > 0:
        fig2 = create_gamma_profile_plot(df_0dte, spot_price, index)
        plt.savefig(chart2_filename)
        plt.close()
        logging.info(f"Profile chart saved as {chart2_filename}")
        
        # Send both charts
        send_email_with_charts([chart1_filename, chart2_filename], index, spot_price)
    else:
        # If no 0-1DTE data, just send the bar chart
        send_email_with_chart(chart1_filename, index, spot_price)

if __name__ == "__main__":
    # Add scipy import at the top of the file
    from scipy.stats import norm
    main()
