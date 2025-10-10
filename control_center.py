#!/usr/bin/env python3
import os, subprocess, csv, json
from datetime import datetime
import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
STATE_DIR = os.path.join(BASE_DIR, "state")
RUN_LOG = os.path.join(BASE_DIR, "run.log")
LIVE_TRADES = os.path.join(REPORTS_DIR, "live_trades.csv")

# TODO: set a strong PIN for dashboard unlock
DASHBOARD_PIN = os.environ.get("FALAH_DASH_PIN", "123456")

def stream_cmd(cmd):
    """Yield lines live from a subprocess until it exits."""
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=BASE_DIR)
        for line in p.stdout:
            yield line.rstrip("\n")
        p.wait()
        yield f"[exit] code={p.returncode}"
    except Exception as e:
        yield f"[error] {e}"

def unlock_dashboard(pin):
    if pin.strip() == DASHBOARD_PIN:
        return True, "Unlocked. Actions are enabled."
    return False, "Wrong PIN. Try again."

def zerodha_show_login_url():
    try:
        # Minimal way: run a small Python one-liner that prints Config login URL if available
        code = "from config import Config; c=Config(); print(getattr(c,'login_url','Login URL not available here.'))"
        out = subprocess.check_output(["python3","-c",code], cwd=BASE_DIR, text=True)
        return out.strip()
    except Exception as e:
        return f"Error fetching login URL: {e}"

def zerodha_save_access_token(request_token):
    try:
        code = f"from config import Config; c=Config(); print(c.authenticate_with_token('{request_token.strip()}') or 'OK')"
        out = subprocess.check_output(["python3","-c",code], cwd=BASE_DIR, text=True)
        return f"Auth result: {out.strip()}"
    except Exception as e:
        return f"Auth error: {e}"

def refresh_index():
    # TODO: replace with your real index refresh script/command if present
    # For now, just tail info if not implemented
    yield "Starting index refresh..."
    # Example: yield from stream_cmd(["python3","-u","tools/update_index.py"])
    yield "Index refresh stub: implement real updater when ready."
    yield "[exit] code=0"

def refresh_daily():
    # Try to call improved_fetcher.run_daily_refresh via a one-liner
    yield "Starting daily data refresh..."
    one_liner = "from improved_fetcher import run_daily_refresh as R; R()"
    yield from stream_cmd(["python3","-u","-c", one_liner])

def run_orchestrator(dry, refresh, notify, sheet):
    cmd = [
        "python3","-u","live_orchestrator.py",
        "--dry_run", "true" if dry else "false",
        "--refresh_data", "true" if refresh else "false",
        "--notify", "true" if notify else "false",
        "--push_sheet", "true" if sheet else "false"
    ]
    yield f"Running: {' '.join(cmd)}"
    # Stream into run.log as well using tee
    # If live_orchestrator already tees internally, plain stream is fine:
    yield from stream_cmd(cmd)

def tail_run_log(n=15):
    try:
        if not os.path.exists(RUN_LOG):
            return "run.log not found."
        with open(RUN_LOG, "r") as f:
            lines = f.readlines()[-n:]
        return "".join(lines)
    except Exception as e:
        return f"Error reading run.log: {e}"

def preview_trades(k=10):
    try:
        if not os.path.exists(LIVE_TRADES):
            return "No trades CSV yet."
        rows = []
        with open(LIVE_TRADES, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        header = rows[0] if rows else []
        data = rows[-k:] if len(rows)>1 else rows
        out = []
        if header:
            out.append(", ".join(header))
        for r in data[1:] if header else data:
            out.append(", ".join(r))
        return "\n".join(out) if out else "No rows."
    except Exception as e:
        return f"Error reading trades: {e}"

with gr.Blocks() as demo:
    unlocked = gr.State(False)
    gr.Markdown("## Falah Control Center")

    with gr.Row():
        pin = gr.Textbox(label="Dashboard PIN", type="password", placeholder="Enter PIN")
        unlock_btn = gr.Button("Unlock")
    unlock_status = gr.Markdown()

    with gr.Accordion("Zerodha Authentication", open=False):
        login_btn = gr.Button("Show Login URL")
        login_url = gr.Textbox(label="Login URL")
        req_token = gr.Textbox(label="Paste request_token from Zerodha redirect")
        save_btn = gr.Button("Save Access Token")
        auth_status = gr.Markdown()

    with gr.Row():
        idx_btn = gr.Button("Refresh Index CSV")
        idx_log = gr.Textbox(label="Index logs", lines=12)
    with gr.Row():
        daily_btn = gr.Button("Refresh Daily Data")
        daily_log = gr.Textbox(label="Daily logs", lines=12)

    gr.Markdown("### Run Orchestrator")
    with gr.Row():
        dry = gr.Checkbox(value=False, label="Dry Run")
        refresh = gr.Checkbox(value=True, label="Refresh Data")
        notify = gr.Checkbox(value=True, label="Notify")
        sheet = gr.Checkbox(value=True, label="Push Sheet")
    run_btn = gr.Button("Run Now")
    run_log = gr.Textbox(label="Run logs", lines=18)

    gr.Markdown("### Reports")
    tail_btn = gr.Button("Show Last run.log")
    tail_out = gr.Textbox(label="run.log tail", lines=12)
    trades_btn = gr.Button("Show Recent Trades")
    trades_out = gr.Textbox(label="Trades preview", lines=12)

    # Wiring
    unlock_btn.click(fn=unlock_dashboard, inputs=pin, outputs=[unlocked, unlock_status])

    login_btn.click(fn=zerodha_show_login_url, outputs=login_url)
    save_btn.click(fn=zerodha_save_access_token, inputs=req_token, outputs=auth_status)

    idx_btn.click(fn=refresh_index, outputs=idx_log)
    daily_btn.click(fn=refresh_daily, outputs=daily_log)
    run_btn.click(fn=run_orchestrator, inputs=[dry, refresh, notify, sheet], outputs=run_log)

    tail_btn.click(fn=tail_run_log, outputs=tail_out)
    trades_btn.click(fn=preview_trades, outputs=trades_out)

if __name__ == "__main__":
    # For production, consider server_name="0.0.0.0", custom port, and proxy behind Nginx
    demo.launch(server_name="0.0.0.0", server_port=7860, auth=("user","change-me-strong"))
