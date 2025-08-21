# dashboard.py - Mobile & Trade-Friendly Falāh Bot Dashboard
import gradio as gr
from bot_logic import create_bot_instance

# Create a single global bot instance on start
bot = create_bot_instance()

def generate_access_code(request_code):
    # Example simple hash-based access code; customize as needed
    if not request_code:
        return "Please enter a request code."
    access_code = hash(request_code) % 1000000
    return f"Access Code: {access_code:06d}"

def update_config(capital, max_trades):
    try:
        if capital is None or capital <= 0:
            return "Invalid capital value."
        if max_trades is None or max_trades <= 0:
            return "Invalid max trades value."
        bot.capital_manager.set_user_capital(capital)
        bot.max_trades = int(max_trades)
        return f"Configuration updated: Capital = ₹{capital}, Max Trades = {max_trades}"
    except Exception as e:
        return f"Error updating config: {e}"

def run_bot_cycle():
    try:
        bot.run_cycle()
        return "Bot cycle executed successfully."
    except Exception as e:
        return f"Error running bot cycle: {e}"

def get_portfolio_summary():
    try:
        summary = bot.get_portfolio_summary()
        return summary
    except Exception as e:
        return {"error": str(e)}

def get_positions():
    try:
        positions = bot.get_positions()
        # Convert list of dicts to DataFrame for nice display in Gradio
        import pandas as pd
        if positions and len(positions) > 0:
            return pd.DataFrame(positions)
        else:
            return "No open positions"
    except Exception as e:
        return f"Error fetching positions: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# Falah Trading Bot Dashboard")

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
