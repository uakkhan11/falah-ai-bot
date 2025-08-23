# dashboard.py - Mobile & Trade-Friendly Falāh Bot Dashboard
import gradio as gr
from bot_logic import create_bot_instance

bot = create_bot_instance()

kite_login_url = "https://kite.zerodha.com/connect/login?api_key=YOUR_API_KEY&v=3"

def authenticate_token(request_token):
    try:
        bot.authenticate_with_token(request_token)
        return "✅ Authenticated successfully! You can now use the trading features."
    except Exception as e:
        return f"❌ Authentication error: {e}"

def generate_access_code(request_code):
    if not request_code:
        return "Please enter a request code."
    access_code = hash(request_code) % 1000000
    return f"Access Code: {access_code:06d}"

def update_config(capital, max_trades):
    try:
        if capital is not None and capital > 0:
            bot.config.INITIAL_CAPITAL = capital
            bot.capital_manager.update_funds()  # refresh available funds after update
        if max_trades is not None and max_trades > 0:
            bot.max_trades = int(max_trades)
        return f"Updated capital: {capital}, max trades: {max_trades}"
    except Exception as e:
        return f"Error updating config: {e}"

def run_bot_cycle():
    return bot.run_cycle()

def get_portfolio_summary():
    return bot.get_portfolio_summary()

def get_positions():
    import pandas as pd
    positions = bot.get_positions()
    if positions and len(positions) > 0 and isinstance(positions, list):
        return pd.DataFrame(positions)
    else:
        return "No open positions"


with gr.Blocks() as demo:
    gr.Markdown("# Falah Trading Bot Dashboard")

    # Kite login and authentication section
    gr.Markdown(f"## Zerodha Kite Authentication\n[Login here]({kite_login_url})")
    request_token_input = gr.Textbox(label="Paste request_token here", placeholder="Paste the request_token value from redirect URL")
    auth_status = gr.Textbox(label="Authentication Status")
    auth_btn = gr.Button("Authenticate")
    auth_btn.click(authenticate_token, inputs=request_token_input, outputs=auth_status)

    with gr.Row():
        request_code_input = gr.Textbox(label="Enter Request Code", placeholder="Enter your request code here")
        access_code_output = gr.Textbox(label="Generated Access Code")
        gen_code_btn = gr.Button("Generate Access Code")
        gen_code_btn.click(generate_access_code, inputs=request_code_input, outputs=access_code_output)

    with gr.Row():
        capital_input = gr.Number(label="Capital (₹)", value=100000)
        max_trades_input = gr.Number(label="Max Trades", value=5, precision=0)
        update_config_output = gr.Textbox(label="Config Update Status")
        update_config_btn = gr.Button("Update Trading Configuration")
        update_config_btn.click(update_config, inputs=[capital_input, max_trades_input], outputs=update_config_output)

    with gr.Row():
        run_bot_btn = gr.Button("Run Bot Cycle")
        run_bot_output = gr.Textbox(label="Bot Run Status")
        run_bot_btn.click(run_bot_cycle, outputs=run_bot_output)

    with gr.Row():
        portfolio_btn = gr.Button("Show Portfolio Summary")
        portfolio_output = gr.JSON()
        portfolio_btn.click(get_portfolio_summary, outputs=portfolio_output)

    with gr.Row():
        positions_btn = gr.Button("Show Open Positions")
        positions_output = gr.Dataframe()
        positions_btn.click(get_positions, outputs=positions_output)

demo.launch(share=True)
