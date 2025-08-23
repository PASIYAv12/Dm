# auto_forwarder.py
# Requires: telethon, python-dotenv
# pip install telethon python-dotenv

import re
import os
import asyncio
from telethon import TelegramClient, events
from telethon.errors import RPCError
from dotenv import load_dotenv

load_dotenv()

# Required from https://my.telegram.org
API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "")

# Specify the source chat (channel/group) and destination chat.
# These can be usernames like "examplechannel" OR numeric IDs (use negative for channels sometimes)
SOURCE_CHAT = os.getenv("SOURCE_CHAT", "")     # e.g. "@signals_channel" OR " -1001234567890 "
DEST_CHAT = os.getenv("DEST_CHAT", "")         # e.g. "@my_private_group" OR chat id string

# Filtering: regular expression to match messages you want forwarded.
# Empty string -> forward everything
FILTER_REGEX = os.getenv("FILTER_REGEX", "")   # e.g. r"(BUY|SELL|STOP LOSS|SL|ENTRY)"
# Forward as copy? "true" => use client.send_message with message.text (copy), otherwise use forward
FORWARD_AS_COPY = os.getenv("FORWARD_AS_COPY", "false").lower() in ("1","true","yes")

# Optional: if you want the script to send a summary to your admin user on errors
ADMIN_USER = os.getenv("ADMIN_USER", "")  # e.g. "@yourusername"

# Create client (session file will be created locally)
SESSION_NAME = os.getenv("SESSION_NAME", "auto_forwarder_session")

if API_ID == 0 or API_HASH == "":
    raise SystemExit("ERROR: Please set API_ID and API_HASH in .env file (from my.telegram.org)")

client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

def matches_filter(text: str) -> bool:
    if not FILTER_REGEX:
        return True
    try:
        return re.search(FILTER_REGEX, text, re.IGNORECASE) is not None
    except re.error:
        # invalid regex -> fallback to false so nothing forwarded
        return False

async def safe_send_summary(msg: str):
    if ADMIN_USER:
        try:
            await client.send_message(ADMIN_USER, msg)
        except Exception:
            pass

@client.on(events.NewMessage(chats=SOURCE_CHAT))
async def handler(event):
    try:
        msg = event.message
        text = (msg.message or "")  # message text
        # Optional: also check msg.media, or msg.entities for structured signals
        if not matches_filter(text):
            return  # skip non-matching

        # Build a user-readable caption (optional) or forward as copy/forward
        if FORWARD_AS_COPY:
            # If media exists, forward as copy preserving media + caption:
            if msg.media:
                # send_file will copy media to destination
                await client.send_file(DEST_CHAT, msg.media, caption=msg.text or "")
            else:
                # plain text
                await client.send_message(DEST_CHAT, msg.text or "")
        else:
            # forward original message (keeps original sender info)
            await client.forward_messages(DEST_CHAT, msg, from_peer=SOURCE_CHAT)

        print(f"Forwarded msg id={msg.id} from {SOURCE_CHAT} -> {DEST_CHAT}")
    except RPCError as e:
        print("RPCError:", e)
        await safe_send_summary(f"RPCError while forwarding: {e}")
    except Exception as ex:
        print("Error forwarding:", ex)
        await safe_send_summary(f"Error forwarding: {ex}")

async def main():
    print("Starting auto-forwarder...")
    await client.start()
    # show info about source/dest
    try:
        src = await client.get_entity(SOURCE_CHAT)
        dst = await client.get_entity(DEST_CHAT)
        print("Source:", getattr(src, "title", getattr(src, "username", str(src))))
        print("Destination:", getattr(dst, "title", getattr(dst, "username", str(dst))))
    except Exception as e:
        print("Warning: couldn't resolve one of the chats. Make sure SESSION account is a member of the source and has access to destination. Error:", e)

    print("Listening for new messages... (CTRL+C to stop)")
    await client.run_until_disconnected()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped by user.")
