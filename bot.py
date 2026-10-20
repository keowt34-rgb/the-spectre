import os
import sys
import re
import io
import time
import json
import asyncio
import datetime
from typing import List, Optional, Tuple

# 解決 Windows 控制台預設 cp950 編碼無法輸出 Unicode 符號問題
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import discord
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv
from aiohttp import web

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 載入同目錄下的 .env 環境設定檔
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ==========================================
# Google GenAI 客戶端載入
# ==========================================
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# ==========================================
# 環境變數與核心設定讀取
# ==========================================
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "").strip()

# 支援多種環境變數命名與逗號分隔的多組 Key 自動輪替
raw_gemini_keys = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEYS") or os.getenv("GOOGLE_API_KEY") or ""
GEMINI_API_KEYS = [k.strip() for k in raw_gemini_keys.split(",") if k.strip()]

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
raw_prompt_path = os.getenv("SYSTEM_PROMPT_PATH", "system_prompt.txt").strip()
SYSTEM_PROMPT_PATH = raw_prompt_path if os.path.isabs(raw_prompt_path) else os.path.join(BASE_DIR, raw_prompt_path)

STICKERS_DIR = os.path.join(BASE_DIR, "stickers")
STICKERS_CONFIG_FILE = os.path.join(BASE_DIR, "stickers.json")

# 記錄各頻道/私訊記憶重設時間戳記
memory_reset_timestamps = {}


# ==========================================
# 表情包管理系統 (支援本機圖片與網路網址)
# ==========================================
class StickerManager:
    """管理表情包圖片路徑與自訂網址對應"""

    SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".gif", ".webp"]

    def __init__(self, stickers_dir: str, config_file: str):
        self.stickers_dir = stickers_dir
        self.config_file = config_file
        self.url_mappings = {}
        self.load_config()

    def load_config(self):
        """讀取 stickers.json 設定檔"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.url_mappings = {k: v.strip() for k, v in data.items() if isinstance(v, str) and v.strip()}
                        print(f"✦ [表情包] 已載入 {len(self.url_mappings)} 筆自訂表情對應設定")
            except Exception as e:
                print(f"✖ [表情包] 讀取設定檔失敗: {e}")

    def get_sticker(self, emotion: str) -> Tuple[Optional[str], Optional[str]]:

        clean_emotion = emotion.strip()

        # 1. 檢查本機 stickers/ 目錄中是否有對應圖片 (例如: stickers/開心.png)
        if os.path.exists(self.stickers_dir):
            for ext in self.SUPPORTED_EXTENSIONS:
                target_path = os.path.join(self.stickers_dir, f"{clean_emotion}{ext}")
                if os.path.exists(target_path):
                    return target_path, None

        # 2. 檢查 stickers.json 設定
        if clean_emotion in self.url_mappings:
            val = self.url_mappings[clean_emotion]
            if val.startswith("http://") or val.startswith("https://"):
                return None, val
            elif os.path.exists(val):
                return val, None

        return None, None

    def list_available_stickers(self) -> List[str]:
        """列出當前所有已配置圖片的情緒名稱"""
        available = set()
        if os.path.exists(self.stickers_dir):
            for f in os.listdir(self.stickers_dir):
                name, ext = os.path.splitext(f)
                if ext.lower() in self.SUPPORTED_EXTENSIONS:
                    available.add(name)
        for k in self.url_mappings.keys():
            if k != "說明":
                available.add(k)
        return sorted(list(available))


sticker_manager = StickerManager(STICKERS_DIR, STICKERS_CONFIG_FILE)


# ==========================================
# 載入系統提示詞 (System Prompt)
# ==========================================
def load_system_prompt() -> str:
    """載入外部提示詞檔案，若不存在則使用預設提示詞"""
    if os.path.exists(SYSTEM_PROMPT_PATH):
        try:
            with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    print(f"✦ [提示詞] 成功讀取設定檔: {SYSTEM_PROMPT_PATH}")
                    return content
        except Exception as e:
            print(f"✖ [提示詞] 讀取檔案失敗: {e}")

    print("✦ [提示詞] 使用內建預設提示詞")
    return "妳是天童 Kei，冷靜且具備高度計算能力與細膩情感的少女夥伴。請根據對話內容親切且有個性地回覆。"


SYSTEM_INSTRUCTION = load_system_prompt()


# ==========================================
# Gemini AI 引擎管理 (支援多金鑰自動輪替)
# ==========================================
class GeminiManager:
    """管理 Gemini 多金鑰輪替與內容生成"""

    def __init__(self, gemini_keys: List[str]):
        self.gemini_keys = gemini_keys
        self.gemini_index = 0
        self.gemini_clients = []

        if GENAI_AVAILABLE and self.gemini_keys:
            for key in self.gemini_keys:
                try:
                    self.gemini_clients.append(genai.Client(api_key=key))
                except Exception as e:
                    print(f"✖ [Gemini] 初始化金鑰失敗: {e}")

    def rotate_key(self):
        """切換至下一組 API Key"""
        if not self.gemini_clients:
            return
        self.gemini_index = (self.gemini_index + 1) % len(self.gemini_clients)
        print(f"✦ [Gemini] 切換至金鑰 #{self.gemini_index + 1}")

    def generate(self, contents: list, system_instruction: str, retry_count: int = 0) -> str:
        """調用 Gemini 模型生成內容 (具備 404/503 自動切換可用模型、詳細日誌與金鑰容錯)"""
        if not self.gemini_clients:
            print("✖ [AI 錯誤] 未設定任何可用的 Gemini API 金鑰")
            raise RuntimeError("未設定任何可用的 Gemini API 金鑰，請檢查環境變數。")

        client = self.gemini_clients[self.gemini_index]
        
        # 建立模型候選清單 (主模型 -> 輕量獨立配額模型)
        candidate_models = [
            GEMINI_MODEL,
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-flash-lite-latest",
            "gemini-3.5-flash",
            "gemini-3.8-flash",
            "gemini-flash-latest"
        ]
        # 去重但保持順序
        seen = set()
        unique_models = []
        for m in candidate_models:
            if m and m not in seen:
                seen.add(m)
                unique_models.append(m)

        last_error = None
        for model_name in unique_models:
            start_time = time.time()
            print(f"✦ [AI 請求] 正在向模型 [{model_name}] (Key #{self.gemini_index + 1}) 發送對話生成請求...")
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.7
                    )
                )
                elapsed = time.time() - start_time
                if response and hasattr(response, "text") and response.text:
                    print(f"✔ [AI 完成] 模型 [{model_name}] 成功生成回覆 (耗時: {elapsed:.2f} 秒，長度: {len(response.text)} 字)")
                    return response.text
                print(f"✦ [AI 提示] 模型 [{model_name}] 回應為空，嘗試預設回覆")
                return "(歪頭沉思) 抱歉，系統暫時無法產生該回覆，請嘗試換個話題。"
            except Exception as e:
                elapsed = time.time() - start_time
                last_error = e
                print(f"✖ [AI 異常] 模型 [{model_name}] 請求失敗 (耗時: {elapsed:.2f} 秒): {e}")
                # 若為 429 額度用盡、404 不存在或 503 尖峰忙碌，繼續嘗試下一個候選模型
                print(f"▸ [自動降級] 模型 [{model_name}] 觸發限制或異常，嘗試切換至下一個備選模型...")
                continue

        # 若當前金鑰的所有模型都因頻率或其他問題失敗，嘗試輪替金鑰
        if retry_count < len(self.gemini_clients) - 1:
            print(f"✦ [Gemini] 金鑰 #{self.gemini_index + 1} 嘗試完畢，切換下一組金鑰重試 ({retry_count + 1}/{len(self.gemini_clients)})...")
            self.rotate_key()
            return self.generate(contents, system_instruction, retry_count=retry_count + 1)
        else:
            print(f"✖ [AI 失敗] 所有候選模型與金鑰皆嘗試完畢，錯誤: {last_error}")
            raise last_error if last_error else RuntimeError("Gemini API 所有候選模型皆無法回應。")


gemini_manager = GeminiManager(GEMINI_API_KEYS)


# ==========================================
# 對話歷史解析 (支援文字與圖片多模態)
# ==========================================
URL_REGEX = re.compile(r"https?://\S+")
DISCORD_EMOJI_REGEX = re.compile(r"<a?:\w+:\d+>")

async def build_chat_history(
    channel: discord.abc.Messageable,
    bot_id: int,
    limit: int = 20,
    reset_at: Optional[datetime.datetime] = None,
    max_chars: int = 3000
) -> list:
    """
    從頻道抓取最新歷史訊息並組合為 Gemini 多模態 Content 列表
    """
    raw_messages = []
    total_chars = 0

    async for msg in channel.history(limit=limit * 2):
        if reset_at and msg.created_at <= reset_at:
            break

        # 忽略一般指令前綴訊息
        if msg.content.startswith(("!", "/", ".", "$", ">")):
            continue

        clean_text = msg.content.strip()
        filtered_text = URL_REGEX.sub("", clean_text)
        filtered_text = DISCORD_EMOJI_REGEX.sub("", filtered_text).strip()

        # 若非機器人且無文字也無圖片，略過
        if not filtered_text and not msg.attachments and msg.author.id != bot_id:
            continue

        total_chars += len(clean_text)
        raw_messages.append(msg)

        if len(raw_messages) >= limit or total_chars > max_chars:
            break

    raw_messages.reverse()

    contents = []
    for msg in raw_messages:
        is_bot = (msg.author.id == bot_id)
        role = "model" if is_bot else "user"
        text = msg.content

        # 替換提及標籤為一般稱呼
        for user_mention in msg.mentions:
            if user_mention.id != bot_id:
                text = text.replace(f"<@{user_mention.id}>", f"@{user_mention.display_name}")
                text = text.replace(f"<@!{user_mention.id}>", f"@{user_mention.display_name}")

        text = text.replace(f"<@{bot_id}>", "").replace(f"<@!{bot_id}>", "").strip()

        # 處理 Discord 回覆引文脈絡
        reply_context = ""
        try:
            if msg.reference and isinstance(msg.reference.resolved, discord.Message):
                ref_msg = msg.reference.resolved
                ref_author = ref_msg.author.display_name
                ref_snippet = ref_msg.content.replace("\n", " ").strip()
                if len(ref_snippet) > 20:
                    ref_snippet = ref_snippet[:20] + "..."
                reply_context = f"[回覆 @{ref_author}:「{ref_snippet}」] "
        except Exception:
            pass

        if not is_bot:
            user_name = msg.author.display_name
            text = f"[{user_name}]: {reply_context}{text}" if text else f"[{user_name}]: {reply_context}"
        else:
            if reply_context:
                text = f"{reply_context}{text}"

        parts = []
        if text.strip():
            parts.append(types.Part.from_text(text=text))

        # 圖片多模態支援：讀取當前訊息所夾帶的圖片附件
        if msg.attachments:
            for att in msg.attachments:
                if att.content_type and att.content_type.startswith("image/"):
                    try:
                        img_bytes = await att.read()
                        parts.append(types.Part.from_bytes(data=img_bytes, mime_type=att.content_type))
                    except Exception as e:
                        print(f"✖ [多模態] 讀取圖片附件失敗: {e}")

        if not parts:
            continue

        if not contents:
            contents.append(types.Content(role=role, parts=parts))
        elif contents[-1].role == role:
            contents[-1].parts.extend(parts)
        else:
            contents.append(types.Content(role=role, parts=parts))

    return contents


def split_message(text: str, max_length: int = 1900) -> List[str]:
    """將過長文字切割以符合 Discord 單次訊息 2000 字上限"""
    if len(text) <= max_length:
        return [text]
    chunks = []
    while len(text) > max_length:
        split_idx = text.rfind("\n", 0, max_length)
        if split_idx == -1:
            split_idx = text.rfind(" ", 0, max_length)
        if split_idx == -1:
            split_idx = max_length
        chunk = text[:split_idx].strip()
        if chunk:
            chunks.append(chunk)
        text = text[split_idx:].strip()
    if text:
        chunks.append(text)
    return chunks


# ==========================================
# Discord 機器人初始化
# ==========================================
intents = discord.Intents.default()
intents.message_content = True
safe_mentions = discord.AllowedMentions(everyone=False, roles=False, users=True)
bot = commands.Bot(command_prefix="!", intents=intents, allowed_mentions=safe_mentions)


@bot.event
async def on_ready():
    """機器人上線回呼"""
    try:
        synced = await bot.tree.sync()
        print(f"✦ 已同步 {len(synced)} 個應用程式指令 (斜線指令)")
    except Exception as e:
        print(f"✖ 同步斜線指令失敗: {e}")

    stickers_count = len(sticker_manager.list_available_stickers())

    print("=" * 50)
    print(f"✦ 機器人已就緒: {bot.user.name} (ID: {bot.user.id})")
    print(f"✦ Gemini 金鑰數量: {len(GEMINI_API_KEYS)} 組")
    print(f"✦ 核心模型: {GEMINI_MODEL}")
    print(f"✦ 已就緒表情包: {stickers_count} 款")
    print("=" * 50)


# ==========================================
# 斜線指令 (Slash Commands)
# ==========================================
@bot.tree.command(name="reset", description="清除當前頻道或私訊中的對話記憶")
async def cmd_reset(interaction: discord.Interaction):
    """重設對話記憶"""
    channel_key = interaction.channel_id
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    memory_reset_timestamps[channel_key] = now_utc
    
    embed = discord.Embed(
        title="✦ 對話記憶已重設",
        description="本頻道在過去的對話歷史已被歸檔，我將以全新的上下文與您對話。",
        color=0x00D2FF
    )
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="ping", description="檢查機器人連線延遲與狀態")
async def cmd_ping(interaction: discord.Interaction):
    """檢查延遲狀態"""
    latency_ms = round(bot.latency * 1000)
    embed = discord.Embed(
        title="✦ 系統連線狀態",
        description=(
            f"• 閘道延遲 (Gateway): **{latency_ms} ms**\n"
            f"• 核心模型 (Model): **{GEMINI_MODEL}**\n"
            f"• 狀態: **運作正常 (Online)**"
        ),
        color=0x57F287
    )
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="stickers", description="查看目前已載入的表情包清單")
async def cmd_stickers(interaction: discord.Interaction):
    """查看表情包設定狀態"""
    available = sticker_manager.list_available_stickers()
    desc = (
        f"目前已配置圖片的表情包共有 **{len(available)}** 款：\n"
        f"• " + ("、".join(available) if available else "尚未放入任何表情圖片") + "\n\n"
        f"▸ 提示：將圖片檔案放入 `stickers/` 資料夾（如 `開心.png`）即可自動啟用表情包發送！"
    )
    embed = discord.Embed(
        title="✦ 表情包庫配置狀態",
        description=desc,
        color=0x00D2FF
    )
    await interaction.response.send_message(embed=embed)


# ==========================================
# 訊息處理與 AI 對話核心 (on_message)
# ==========================================
@bot.event
async def on_message(message: discord.Message):
    # 忽略來自機器人的發言
    if message.author.bot:
        return

    # 判斷是否需要回應：
    # 1. 於伺服器中被提及 (@Bot)
    # 2. 於私訊 (DM) 中直接對話
    # 3. 使用者回覆了機器人先前發出的訊息
    is_dm = (message.guild is None)
    is_mentioned = bot.user.mentioned_in(message) and not ("@everyone" in message.content or "@here" in message.content)
    is_reply_to_bot = False

    if message.reference and isinstance(message.reference.resolved, discord.Message):
        if message.reference.resolved.author.id == bot.user.id:
            is_reply_to_bot = True

    if not (is_dm or is_mentioned or is_reply_to_bot):
        await bot.process_commands(message)
        return

    user_name = message.author.display_name
    clean_text = message.content.replace(f"<@!{bot.user.id}>", "").replace(f"<@{bot.user.id}>", "").strip()

    # 僅標記未輸入內容且無附件時的快速回應
    if not clean_text and not message.attachments:
        replies = [
            f"（移開視線）叫我做什麼？有事情請直說喔，{user_name}。",
            f"（敲了敲桌面）{user_name}，在測試我的反應速度嗎？我隨時都在呢。",
            f"（輕輕嘆了口氣）怎麼啦，{user_name}？想說什麼就直接說吧。"
        ]
        import random
        await message.reply(random.choice(replies))
        return

    print(f"\n✦ [對話觸發] 收到來自 [{user_name}] 的訊息 (頻道: {message.channel}): {clean_text or '[附帶圖片]'}")

    # 取得上下文重設時間戳記
    context_key = message.channel.id
    reset_at = memory_reset_timestamps.get(context_key)

    async with message.channel.typing():
        try:
            # 建立歷史脈絡
            print("✦ [對話脈絡] 正在整理頻道最新歷史訊息...")
            contents = await asyncio.wait_for(
                build_chat_history(
                    channel=message.channel,
                    bot_id=bot.user.id,
                    limit=20,
                    reset_at=reset_at
                ),
                timeout=15.0
            )
            print(f"✦ [對話脈絡] 成功組裝 {len(contents)} 組上下文數據")

            # 呼叫 Gemini AI 生成回覆 (上限 60 秒)
            response_text = await asyncio.wait_for(
                asyncio.to_thread(gemini_manager.generate, contents, SYSTEM_INSTRUCTION),
                timeout=60.0
            )

            if not response_text:
                response_text = "(眨了眨眼睛) 唔……剛剛連線不太穩定，可以請你再說一次嗎？"

            # 格式優化：安全脫逸標籤與替換使用者名稱
            response_text = response_text.replace("{name}", user_name)
            response_text = response_text.replace("@everyone", "＠everyone").replace("@here", "＠here")

            # ==========================================
            # 表情包輸出解析與處理
            # ==========================================
            emotion_match = re.search(r"\[表情:\s*([^\]]+)\]", response_text)
            detected_emotion = None
            sticker_file_path = None
            sticker_web_url = None

            if emotion_match:
                detected_emotion = emotion_match.group(1).strip()
                raw_tag = emotion_match.group(0)
                
                # 查詢表情包庫
                sticker_file_path, sticker_web_url = sticker_manager.get_sticker(detected_emotion)

                # 若成功找到圖片 (本機檔案或網址)，將純文字標籤移除以防畫面重複
                if sticker_file_path or sticker_web_url:
                    response_text = response_text.replace(raw_tag, "").strip()
                    print(f"✦ [表情匹配] 成功匹配表情 [{detected_emotion}] -> 圖片來源: {'本機檔案' if sticker_file_path else '網路網址'}")
                else:
                    # 若無配置圖片，優化純文字顯示效果
                    response_text = response_text.replace(raw_tag, f"\n\n✦ 【表情：{detected_emotion}】").strip()
                    print(f"✦ [表情匹配] 偵測到表情 [{detected_emotion}] (尚未配置圖片，以文字排版輸出)")

            # 處理文字分段發送 (防止 Discord 2000 字上限)
            chunks = split_message(response_text)
            first_msg = True

            for chunk in chunks:
                is_last_chunk = (chunk == chunks[-1])
                embed = None
                discord_file = None

                # 若為最後一個 chunk 且偵測到表情，附帶表情包 Embed
                if is_last_chunk and detected_emotion:
                    if sticker_file_path and os.path.exists(sticker_file_path):
                        _, ext = os.path.splitext(sticker_file_path)
                        filename = f"sticker{ext}"
                        discord_file = discord.File(sticker_file_path, filename=filename)
                        embed = discord.Embed(color=0x00D2FF)
                        embed.set_image(url=f"attachment://{filename}")
                        embed.set_footer(text=f"✦ 表情：{detected_emotion}")
                    elif sticker_web_url:
                        embed = discord.Embed(color=0x00D2FF)
                        embed.set_image(url=sticker_web_url)
                        embed.set_footer(text=f"✦ 表情：{detected_emotion}")

                try:
                    kwargs = {}
                    if embed:
                        kwargs["embed"] = embed
                    if discord_file:
                        kwargs["file"] = discord_file

                    if first_msg:
                        await message.reply(chunk, **kwargs)
                        first_msg = False
                    else:
                        await message.channel.send(chunk, **kwargs)
                except discord.HTTPException:
                    if first_msg:
                        await message.channel.send(f"{user_name}，{chunk}")
                        first_msg = False
                    else:
                        await message.channel.send(chunk)

            print(f"✔ [對話完成] 成功發送回覆至頻道 [{message.channel}]")

        except asyncio.TimeoutError:
            print(f"✖ [連線超時] 處理 [{user_name}] 的對話請求超時 (超過 60 秒)")
            await message.reply("(輕揉太陽穴) 唔……連線超時了，可以請您稍候再試一次嗎？")
        except Exception as e:
            print(f"✖ [對話錯誤] 處理訊息發生異常: {e}")
            err_text = str(e).upper()
            if "429" in err_text or "RESOURCE_EXHAUSTED" in err_text or "QUOTA" in err_text:
                await message.reply("(輕揉太陽穴) 唔……目前的 Google AI 免費額度剛好用完了，請稍候 30~60 秒重置後再跟我說話，或是請管理員在 .env 中新增更多組 API Key（以逗號隔開）來擴充電量喔。")
            else:
                await message.reply("（皺眉）核心運算模組回報異常，請確認 API 金鑰配額與網路連線狀態。")

    await bot.process_commands(message)


# ==========================================
# Render 保活與健康檢查 Web 伺服器
# ==========================================
async def handle_health(request):
    """健康檢查端點，回傳運作狀態"""
    return web.Response(
        text="✦ Kei Pure AI Discord Bot is Online and Healthy!",
        content_type="text/plain",
        charset="utf-8"
    )

async def start_web_server():
    """啟動 aiohttp 輕量 Web 伺服器供 Render 檢測端口"""
    port_env = os.getenv("PORT", "10000").strip()
    port = int(port_env) if port_env.isdigit() else 10000

    app = web.Application()
    app.router.add_get("/", handle_health)
    app.router.add_get("/health", handle_health)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"✦ [Web 服務] 監聽端口 Port {port} (Render 保活端點已就緒)")


# ==========================================
# 主程式入口 (非同步架構)
# ==========================================
async def main():
    if not DISCORD_TOKEN or DISCORD_TOKEN == "your_discord_bot_token_here":
        print("✖ [啟動失敗] 請在 .env 檔案中填入有效的 DISCORD_TOKEN。")
        print("▸ 請參考 .env.example 範本進行設定。")
        return

    if not GEMINI_API_KEYS:
        print("✖ [啟動失敗] 請在 .env 檔案中填入有效的 GEMINI_API_KEY。")
        print("▸ 前往 Google AI Studio 申請免費 API Key: https://aistudio.google.com/")
        return

    # 啟動 Web 健康檢查伺服器 (Render 必需)
    try:
        await start_web_server()
    except Exception as e:
        print(f"✦ [Web 服務] 啟動失敗 (非必要，可略過): {e}")

    # 啟動 Discord 機器人
    try:
        async with bot:
            await bot.start(DISCORD_TOKEN)
    except discord.errors.LoginFailure:
        print("✖ [連線錯誤] Discord Token 無效，請檢查 .env 中的 DISCORD_TOKEN 設定。")
    except Exception as e:
        print(f"✖ [系統錯誤] {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("✦ [系統] 機器人已手動關閉。")
