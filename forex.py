import os
import logging
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Environment Variables
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_ENV = os.getenv("OANDA_ENV", "practice")
TELEGRAM_TOKEN = os.getenv("AAE29u264oOFf9qH0oBSmjfTKQLSlUu_TUo")
ADMIN_TELEGRAM_ID = os.getenv("7786434709")

OANDA_URL = f"https://api-fxpractice.oanda.com/v3/accounts/{OANDA_ACCOUNT_ID}"

headers = {
    "Authorization": f"Bearer {OANDA_API_KEY}",
    "Content-Type": "application/json"
}

# Check balance
async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_user.id) != ADMIN_TELEGRAM_ID:
        await update.message.reply_text("⛔ Unauthorized")
        return

    r = requests.get(f"{OANDA_URL}/summary", headers=headers)
    data = r.json()
    bal = data['account']['balance']
    pl = data['account']['pl']
    await update.message.reply_text(f"💰 Balance: {bal}\n📊 P/L: {pl}")

# Ping bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 AI Forex Bot running!")

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("balance", balance))

    app.run_polling()
