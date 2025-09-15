import os
import discord
from discord.ext import commands
from dotenv import load_dotenv

# Load environment variables from .env file
env_file = ".env"
try:
    load_dotenv(dotenv_path=env_file)
    #print(f"\u2705 Loaded environment variables from {env_file}")
except FileNotFoundError:
    #print(f"\u26a0\ufe0f  Environment file {env_file} not found")
    pass
except Exception as e:
    #print(f"\u274c Error loading environment file: {e}")
    pass


#print("\ud83e\udd16 Starting ARIA ELITE Discord Bot...")

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not DISCORD_BOT_TOKEN:
    #print("\u274c DISCORD_BOT_TOKEN not set!")
    #print("\ud83d\udcdd Please edit local.env file and add your Discord bot token")
    #print("\ud83d\udd17 Get your token from: https://discord.com/developers/applications")
    exit(1)


@bot.event
async def on_ready():
    #print("\u2705 Discord bot initialized successfully!")
    #print("\ud83d\ude80 Starting bot... (Press Ctrl+C to stop)")
    print(f'{bot.user} has connected to Discord!')


try:
    bot.run(DISCORD_BOT_TOKEN)
except discord.errors.LoginFailure as e:
    #print("\n\ud83d\uded1 Discord bot stopped by user")
    #print(f"\u274c Error starting Discord bot: {e}")
    #print("\ud83d\udcdd Make sure you have:")
    #print("   1. Created a Discord bot at https://discord.com/developers/applications")
    #print("   2. Set DISCORD_BOT_TOKEN in local.env file")
    #print("   3. Set DISCORD_ALLOWED_USERS with your Discord user ID")
    #print("   4. Invited the bot to your server with proper permissions")
    exit(1)
