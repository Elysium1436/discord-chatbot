# bot.py
import os
import discord
from dotenv import load_dotenv
import random
from discord.ext import commands
import ml_models
import re
import asyncio
bot = commands.Bot(command_prefix='**')

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')


discord_ai = ml_models.DiscordModelWrapper()


@bot.event
async def on_ready():
    print(f'{bot.user.name} is up and going!')


@bot.command('sunny-chan')
async def discord_emu(ctx, *args):
    init_text = " ".join(args)
    init_text = init_text.replace('\n', '<ENTER>')
    init_text = "<START>" + init_text + "<END>"
    await ctx.send("Writing, please wait a moment...")
    async with ctx.typing():
        text = discord_ai.generate_response(init_text, 300)
        await ctx.send("Finished writing")
        text = re.sub(r'(?:<START>|<END>)', '', text)

        for message in text.split('<ENTER>'):
            await ctx.send(message)
            await asyncio.sleep(1)


bot.run(TOKEN)
