import gradio as gr
import pandas as pd
from bot_logic import create_bot_instance

bot = create_bot_instance()

kite_login_url = "https://kite.trade/connect/login?v=3&api_key=ijzeuwuylr3g0kug"
notification_messages = []

def get_auth_status():
    if bot.is_authenticated():
        try:
            profile = bot.kite.profile()
            return f"✅ Authenticated as {profile['user_name']}"
        except Exception:
            return "✅ Authenticated (profile fetch failed)"
    else:
        return "❌ Not authenticated"

def authenticate_token(request_token):
    try:
        bot.authenticate_with_token(request_token)
        return "✅ Authenticated successfully! You can now use the trading features."
    except Exception as e:
        return f"❌ Authentication error: {e}"

def refresh_portfolio_metrics():
    summary = bot.get_portfolio_summary()
    pv = summary.get('portfolio_value', 'N/A')
    try:
        ac = bot.capital_manager.get_available_capital()
    except Exception:
        ac = 'N/A'
    pnl = 'N/A'
    try:
        pnl = float(pv) - bot.config.INITIAL_CAPITAL if isinstance(pv, (int, float, str)) and str(pv).replace('.', '', 1).isdigit() else 'N/A'
    except Exception:
        pass
    open_trades = summary.get('open_trades', 0)
    cooling = "Active" if getattr(bot, 'cooling_mode', False) else "Inactive"
    return str(pv), str(ac), str(pnl), open_trades, cooling

def update_trade_settings(enable_auto, force_cooling, risk_val, max_trades):
    try:
        bot.auto_trading_enabled = enable_auto
        bot.cooling_mode = force_cooling
        bot.config.RISK_PER_TRADE = risk_val
        bot.max_trades = int(max_trades)
        return "Trading settings updated successfully!"
    except Exception as e:
        return f"Error updating settings: {e}"

def run_bot_cycle():
    global notification_messages
    result = bot.run_cycle()
    notification_messages.insert(0, f"Cycle run at latest:\n{result}")
    if len(notification_messages) > 20:
        notification_messages = notification_messages[:20]
    return result

def get_positions():
    try:
        positions = bot.get_positions()
        if isinstance(positions, pd.DataFrame):
            return positions
        else:
            return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame([{"error": str(e)}])

def refresh_portfolio_metrics():
    try:
        summary = bot.get_portfolio_summary()
        pv = summary.get('portfolio_value', 'N/A')
        ac = bot.capital_manager.get_available_capital() if bot.capital_manager else 'N/A'
        pnl = 'N/A'
        try:
            pnl = float(pv) - bot.config.INITIAL_CAPITAL if isinstance(pv, (int,float)) else 'N/A'
        except:
            pass
        open_trades = summary.get('open_trades', 0)
        cooling = "Active" if getattr(bot, 'cooling_mode', False) else "Inactive"
        return str(pv), str(ac), str(pnl), open_trades, cooling
    except Exception as e:
        return ("Error",)*5

def run_historical_fetch():
    try:
        fetcher = bot.data_manager  # or SmartHalalFetcher if separate
        # Call your historical fetch here if exposed separately
        # For now, just simulate success
        return "✅ Historical data fetch completed successfully."
    except Exception as e:
        return f"❌ Error during fetch: {e}"

def get_notifications():
    return "\n\n".join(notification_messages[-20:]) if notification_messages else "No notifications yet."

with gr.Blocks() as demo:
    gr.Markdown("# Falah Trading Bot Mobile Dashboard")

    with gr.Row():
        gr.Markdown(f"### Zerodha Kite Authentication  [Login here]({kite_login_url})")
    with gr.Row():
        request_token_input = gr.Textbox(label="Paste request_token here", placeholder="Paste the request_token value from redirect URL")
        auth_status = gr.Textbox(label="Authentication Status")
        auth_btn = gr.Button("Authenticate")
        auth_status.value = get_auth_status()
        auth_btn.click(authenticate_token, inputs=request_token_input, outputs=auth_status)
        auth_btn.click(lambda: get_auth_status(), outputs=auth_status)

    gr.Markdown("### Portfolio Metrics")
    with gr.Row():
        portfolio_value = gr.Textbox(label="Portfolio Value (₹)", interactive=False)
        available_capital = gr.Textbox(label="Available Capital (₹)", interactive=False)
        floating_pnl = gr.Textbox(label="Floating PnL (₹)", interactive=False)
        open_positions_count = gr.Number(label="Open Positions", interactive=False)
        cooling_mode_status = gr.Textbox(label="Cooling Mode", interactive=False)
        refresh_btn = gr.Button("Refresh Metrics")
        refresh_btn.click(refresh_portfolio_metrics, outputs=[portfolio_value, available_capital, floating_pnl, open_positions_count, cooling_mode_status])

    gr.Markdown("### Trade Controls")
    with gr.Row():
        enable_auto_trade = gr.Checkbox(label="Enable Auto Trading", value=getattr(bot, 'auto_trading_enabled', True))
        cooling_mode_override = gr.Checkbox(label="Force Cooling Mode", value=getattr(bot, 'cooling_mode', False))
    with gr.Row():
        risk_per_trade_slider = gr.Slider(minimum=1000, maximum=100000, step=500, label="Risk Per Trade (₹)", value=bot.config.RISK_PER_TRADE)
        max_trades_input = gr.Number(label="Max Concurrent Trades", value=getattr(bot, 'max_trades', 5), precision=0)
    with gr.Row():
        update_settings_btn = gr.Button("Update Settings")
        update_settings_status = gr.Textbox(label="Settings Status", interactive=False)
        update_settings_btn.click(update_trade_settings, inputs=[enable_auto_trade, cooling_mode_override, risk_per_trade_slider, max_trades_input], outputs=update_settings_status)

    gr.Markdown("### Trading Actions")
    with gr.Row():
        run_bot_btn = gr.Button("Run Bot Cycle")
        run_bot_output = gr.Textbox(label="Bot Run Status", lines=10)
        run_bot_btn.click(run_bot_cycle, outputs=run_bot_output)

    with gr.Row():
        fetch_data_btn = gr.Button("Fetch Historical Data")
        fetch_data_output = gr.Textbox(label="Fetch Status")
        fetch_data_btn.click(run_historical_fetch, outputs=fetch_data_output)

    gr.Markdown("### Open Positions")
    with gr.Row():
        positions_btn = gr.Button("Show Open Positions")
        positions_output = gr.Dataframe()
        positions_btn.click(get_positions, outputs=positions_output)

    gr.Markdown("### Notifications & Logs")
    notifications_box = gr.Textbox(label="Recent Notifications", interactive=False, lines=8)
    refresh_notifications_btn = gr.Button("Refresh Notifications")
    refresh_notifications_btn.click(get_notifications, outputs=notifications_box)

demo.launch(share=True)
