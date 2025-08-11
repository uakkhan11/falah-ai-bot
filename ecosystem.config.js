module.exports = {
  apps: [{
    name: 'falah-trading-bot',
    script: 'main.py',
    interpreter: 'python3',
    interpreter_args: 'trading-env/bin/python',
    cwd: '/root/falah-trading-bot',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      NODE_ENV: 'development'
    },
    env_production: {
      NODE_ENV: 'production',
      ZERODHA_API_KEY: 'your_api_key',
      ZERODHA_API_SECRET: 'your_api_secret'
    },
    log_file: './logs/combined.log',
    out_file: './logs/out.log',
    error_file: './logs/error.log',
    time: true
  }]
};
