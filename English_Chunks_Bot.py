import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from groq import Groq
from streamlit_mic_recorder import mic_recorder
from io import BytesIO
import json
import random
import asyncio
import edge_tts
import base64
from streamlit_gsheets import GSheetsConnection

# --- 1. 初始化與設定 ---
st.set_page_config(page_title="English Chunk Master (Online)", layout="centered", page_icon="🦁")

if 'current_mode' not in st.session_state: st.session_state.current_mode = None
if 'current_chunks' not in st.session_state: st.session_state.current_chunks = []
if 'current_topic' not in st.session_state: st.session_state.current_topic = ""
if 'current_indices' not in st.session_state: st.session_state.current_indices = []
if 'current_level' not in st.session_state: st.session_state.current_level = "B1"
if 'generated_prompt' not in st.session_state: st.session_state.generated_prompt = ""
if 'feedback' not in st.session_state: st.session_state.feedback = None
if 'processed' not in st.session_state: st.session_state.processed = False
if 'api_key_input' not in st.session_state: st.session_state.api_key_input = ""
if 'df' not in st.session_state: st.session_state.df = None
# [修改點 1] 新增 recorder_key 來強制重置錄音元件
if 'recorder_key' not in st.session_state: st.session_state.recorder_key = str(random.randint(1000, 9999))

conn = st.connection("gsheets", type=GSheetsConnection)

# --- 2. 資料處理 ---

def load_data():
    try:
        df = conn.read(worksheet="Sheet1", ttl=0)
        df.columns = df.columns.str.strip()
        required = ['Chunks', 'Topic', 'Date', 'Times', 'Next']
        now = datetime.now()
        today_str = f"{now.year}/{now.month}/{now.day}"
        
        for col in required:
            if col not in df.columns:
                if col == 'Times': df[col] = 0
                elif col in ['Date', 'Next']: df[col] = today_str
                else: df[col] = ""

        df['Times'] = pd.to_numeric(df['Times'], errors='coerce').fillna(0).astype(int)

        for col in ['Next', 'Date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                fill_val = (pd.Timestamp.now() - pd.Timedelta(days=1)) if col == 'Next' else pd.Timestamp.now()
                df[col] = df[col].fillna(fill_val).dt.normalize()
        return df
    except Exception as e:
        st.error(f"讀取 Google Sheet 失敗: {e}")
        return pd.DataFrame(columns=['Chunks', 'Topic', 'Date', 'Times', 'Next'])

def save_data(df):
    try:
        save_df = df.copy()
        def custom_date_fmt(dt):
            if pd.isnull(dt): return ""
            return f"{dt.year}/{dt.month}/{dt.day}"

        if 'Next' in save_df.columns: save_df['Next'] = save_df['Next'].apply(custom_date_fmt)
        if 'Date' in save_df.columns: save_df['Date'] = save_df['Date'].apply(custom_date_fmt)
        
        conn.update(worksheet="Sheet1", data=save_df)
        st.cache_data.clear()
        st.toast("☁️ 進度已同步", icon="✅")
    except Exception as e:
        st.error(f"寫入 Google Sheet 失敗: {e}")

# --- 3. AI 與 語音邏輯 (Prompt 優化版) ---

def get_cefr_level(times):
    if times < 3: return "A2"
    if times < 6: return "B1"
    elif times < 9: return "B2"
    else: return "C1"

def get_groq_client():
    if not st.session_state.api_key_input: return None
    return Groq(api_key=st.session_state.api_key_input)

def transcribe_audio(audio_bytes):
    client = get_groq_client()
    if not client: return ""
    try:
        audio_file = BytesIO(audio_bytes)
        audio_file.name = "audio.webm"
        return client.audio.transcriptions.create(
            file=audio_file, model="whisper-large-v3", language="en", response_format="text"
        )
    except Exception as e: return f"Error: {str(e)}"

async def generate_tts(text):
    communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural", rate="-10%")
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

def play_audio_bytes(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""<audio controls autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>"""
    st.markdown(md, unsafe_allow_html=True)

def generate_challenge(phrase, level):
    client = get_groq_client()
    if not client: return "No API Key"
    
    # 優化：要求產生更自然、針對性更強的情境
    prompt = (
        f"You are an English teacher. Target Phrase: '{phrase}'. Level: {level}. "
        f"Create a SHORT Chinese sentence (Traditional TW) that forces the student to use this exact phrase to answer. "
        f"Rules: \n"
        f"1. Length: Max 1 sentence (concise).\n"
        f"2. Do NOT mention the English phrase in the output.\n"
        f"3. The scenario should imply the need for '{phrase}'.\n"
        f"Output ONLY the Chinese sentence."
    )
    
    completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}])
    return completion.choices[0].message.content

def evaluate_submission(user_text, target_phrases, mode, context_prompt=""):
    client = get_groq_client()
    if not client: return {}
    
    targets_str = ", ".join(target_phrases) if isinstance(target_phrases, list) else target_phrases
    
    system_instruction = (
        "You are a strict but helpful English pronunciation and grammar coach. "
        "Your task is to evaluate if the user correctly used the Target Phrase in a sentence based on the Context. "
        "The sentence must be natural and grammatically correct. "
        "You must output valid JSON only."
    )
    
    user_prompt = f"""
    Context (Chinese): "{context_prompt}"
    Target Phrase(s): "{targets_str}"
    User Audio Transcript: "{user_text}"
    
    Please evaluate based on these STRICT rules:
    
    1. **Usage Check (CRITICAL)**: 
       - Did the user use the Target Phrase "{targets_str}"? 
       - If the target phrase is MISSING or significantly CHANGED -> Score MUST be under 60.
       
    2. **Grammar & Flow**:
       - If Target Phrase is present but grammar is bad -> Score 60-75.
       - If Target Phrase is present and grammar is okay -> Score 80-90.
       - If Perfect -> Score 91-100.
       
    3. **Feedback (Traditional Chinese)**:
       - Briefly explain why they got this score.
       - Point out grammar mistakes or unnatural phrasing.
       
    4. **Better Sentence (English)**:
       - Provide a natural, native-level sentence using the Target Phrase that fits the Context.
       
    Output Format (JSON):
    {{
        "score": (int),
        "feedback": "(string in Traditional Chinese)",
        "better_sentence": "(string in English)"
    }}
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instruction}, 
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"JSON Error: {e}")
        return {"score": 0, "feedback": "系統評分錯誤，請重試。", "better_sentence": ""}

# --- 4. 主介面 ---

with st.sidebar:
    st.title("⚙️ 設定")
    groq_api_key = st.text_input("Groq API Key", type="password", value=st.session_state.api_key_input)
    st.session_state.api_key_input = groq_api_key
    
    st.markdown("---")
    if st.button("🔄 重新載入雲端資料"):
        st.session_state.df = load_data()
        st.session_state.processed = False
        st.session_state.feedback = None
        st.session_state.generated_prompt = ""
        st.session_state.current_chunks = []
        st.session_state.recorder_key = str(random.randint(1000, 9999)) # 重置錄音鍵值
        st.rerun()

if st.session_state.df is None:
    with st.spinner("正在連線至 Google Drive..."):
        st.session_state.df = load_data()

st.title("🦁 English Chunk Master (Online)")

if not groq_api_key:
    st.info("💡 請先在左側輸入 Groq API Key")
elif st.session_state.df is None or st.session_state.df.empty:
    st.warning("⚠️ 無法讀取資料，請檢查 Google Sheets 連結或權限設定。")
else:
    today = pd.Timestamp.now().normalize()
    due_items = st.session_state.df[st.session_state.df['Next'] <= today]

    if len(due_items) == 0:
        st.success("🎉 今日進度已完成！")
        if st.button("🔥 強制複習全部 (Demo)"):
            st.session_state.df['Next'] = today
            st.rerun()
    else:
        st.markdown(f"##### 📅 今日待複習: **{len(due_items)}** 筆")
        st.progress(min(1.0, len(due_items)/max(1, len(st.session_state.df))))

        # 1. 抽取題目
        if not st.session_state.processed and not st.session_state.current_chunks:
            random_idx = random.choice(due_items.index)
            row = st.session_state.df.loc[random_idx]
            
            topic = row['Topic']
            phrase = row['Chunks']
            times = row['Times']
            
            topic_siblings = due_items[due_items['Topic'] == topic]
            
            if len(topic_siblings) >= 2 and random.random() > 0.5:
                st.session_state.current_mode = "Story"
                sample_n = min(3, len(topic_siblings))
                selected = topic_siblings.sample(sample_n)
                st.session_state.current_chunks = selected['Chunks'].tolist()
                st.session_state.current_indices = selected.index.tolist()
                st.session_state.generated_prompt = "Story Mode"
            else:
                st.session_state.current_mode = "Single"
                st.session_state.current_chunks = [phrase]
                st.session_state.current_indices = [random_idx]
                st.session_state.current_level = get_cefr_level(times)
                with st.spinner("AI 出題中..."):
                    st.session_state.generated_prompt = generate_challenge(phrase, st.session_state.current_level)

        # 2. 顯示題目
        mode = st.session_state.current_mode
        if not st.session_state.current_indices: st.rerun()

        st.markdown(f"### Topic: {st.session_state.df.loc[st.session_state.current_indices[0], 'Topic']}")
        
        if mode == "Single":
            st.caption(f"Level: {st.session_state.current_level}")
            st.markdown(f"""
            <div style="background-color:#403F6F; padding:20px; border-radius:10px; margin-bottom:15px;">
                <div style="font-size:1.5em; font-weight:bold; color: white;">{st.session_state.generated_prompt}</div>
                <div style="color:#A5B4FC; margin-top:10px;">Target: <b>{st.session_state.current_chunks[0]}</b></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("請說一段話，包含以下片語：")
            cols = st.columns(len(st.session_state.current_chunks))
            for i, c in enumerate(st.session_state.current_chunks):
                cols[i].markdown(f"**{i+1}. {c}**")

        # 3. 錄音
        # [修改點 1] 使用 session_state.recorder_key 作為 key
        audio_data = mic_recorder(
            start_prompt="🎙️ 開始回答", 
            stop_prompt="⏹️ 完成", 
            key=st.session_state.recorder_key
        )

        if audio_data and not st.session_state.processed:
            user_text = transcribe_audio(audio_data['bytes'])
            st.write(f"👂 You said: {user_text}")
            
            with st.spinner("AI 評分中..."):
                feedback = evaluate_submission(user_text, st.session_state.current_chunks, mode, st.session_state.generated_prompt)
                st.session_state.feedback = feedback
                st.session_state.processed = True
                st.rerun()

        # 4. 結果與更新
        if st.session_state.processed and st.session_state.feedback:
            res = st.session_state.feedback
            score = res.get('score', 0)
            color = "green" if score >= 80 else "red"
            st.markdown(f"## Score: :{color}[{score}]")
            st.markdown(f"**💡 AI 建議:** {res.get('feedback')}")
            st.markdown(f"**🌟 最佳範例:** {res.get('better_sentence')}")
            
            if res.get('better_sentence'):
                audio_bytes = asyncio.run(generate_tts(res['better_sentence']))
                play_audio_bytes(audio_bytes)

            if st.button("➡️ 下一題 (Next)"):
                is_correct = score >= 80
                today_obj = pd.Timestamp.now().normalize()
                
                # [修改點 2] 只有答對才更新資料
                if is_correct:
                    for idx in st.session_state.current_indices:
                        current_times = int(st.session_state.df.loc[idx, 'Times'])
                        new_times = current_times + 1
                        next_date = today_obj + timedelta(days=new_times)
                        
                        st.session_state.df.loc[idx, 'Times'] = new_times
                        st.session_state.df.loc[idx, 'Next'] = next_date

                    with st.spinner("☁️ 正在同步至 Google Sheets..."):
                        save_data(st.session_state.df)
                else:
                    st.toast("💪 加油！下次再挑戰，進度保持不變。", icon="🔁")
                
                # [修改點 3] 重置所有狀態，並更新 recorder_key
                st.session_state.current_chunks = []
                st.session_state.current_indices = []
                st.session_state.current_mode = None
                st.session_state.generated_prompt = "" 
                st.session_state.processed = False
                st.session_state.feedback = None
                st.session_state.recorder_key = str(random.randint(1000, 9999)) # 關鍵：換掉錄音元件的ID
                
                st.rerun()