import streamlit as st
import asyncio
import sounddevice as sd
from invoke.config import Config
import yaml
from io import StringIO
import threading # スレッドを扱うため

# main.py から必要な関数とクラスをインポート
# Streamlit環境で実行するため、相対インポートを修正
from src.tts.base import TextToSpeech
from src.asr.base import SpeechToText
from src.tts.voicevox import VoiceVox
from src.tts.coeiroink import CoeiroInk
from src.tts.aivisspeech import AivisSpeech
from src.lmm.common import LLMConfig, history_to_text
from src.lmm.llm import LLMs
from src.lmm.img import ImageGenerator

# Streamlitのセッション状態を初期化
if "history" not in st.session_state:
    st.session_state.history = []
if "turn" not in st.session_state:
    st.session_state.turn = ""
if "cfg" not in st.session_state:
    st.session_state.cfg = None
if "llmcfg" not in st.session_state:
    st.session_state.llmcfg = None
if "llms" not in st.session_state:
    st.session_state.llms = None
if "image_generator" not in st.session_state:
    st.session_state.image_generator = None
if "asr" not in st.session_state:
    st.session_state.asr = None
if "engines" not in st.session_state:
    st.session_state.engines = {}
if "ai_config" not in st.session_state:
    st.session_state.ai_config = {}
if "playback_queue" not in st.session_state:
    st.session_state.playback_queue = asyncio.Queue()
if "synthesis_queue" not in st.session_state:
    st.session_state.synthesis_queue = asyncio.Queue()
if "current_ai_response" not in st.session_state:
    st.session_state.current_ai_response = ""
if "async_loop" not in st.session_state: # 専用のasyncioループ
    st.session_state.async_loop = None
if "workers_started" not in st.session_state: # ワーカー起動フラグ
    st.session_state.workers_started = False
if "recording_in_progress" not in st.session_state: # 録音中フラグ
    st.session_state.recording_in_progress = False

# 非同期イベントループを別スレッドで実行するための関数
def run_async_loop_in_thread(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

# StreamlitのUI
st.title("AIチャットアプリケーション")

# 設定ファイルのアップロード
uploaded_file = st.sidebar.file_uploader("設定ファイルをアップロード (invoke-utf8.yaml)", type=["yaml", "yml"])

if uploaded_file is not None:
    # YAMLファイルを読み込む
    string_data = StringIO(uploaded_file.getvalue().decode("utf-8"))
    config_data = yaml.safe_load(string_data)
    st.session_state.cfg = Config(config_data)

    # 初期化処理
    if st.session_state.llmcfg is None:
        st.session_state.llmcfg = LLMConfig(st.session_state.cfg)
        st.session_state.llms = LLMs(st.session_state.llmcfg)
        st.session_state.image_generator = ImageGenerator(st.session_state.llmcfg)

        # 音声認識の設定
        user_input_mode = st.session_state.cfg.chat.user.input
        if user_input_mode == "vosk":
            from src.asr.vosk_asr import VoskASR
            st.session_state.asr = VoskASR(**st.session_state.cfg.vosk)
        elif user_input_mode == "whisper":
            from src.asr.whisper_asr import WhisperASR
            st.session_state.asr = WhisperASR(**st.session_state.cfg.whisper, **st.session_state.cfg.webrtcvad)
        elif user_input_mode == "gemini":
            from src.asr.gemini_asr import GeminiASR
            st.session_state.asr = GeminiASR(st.session_state.cfg.gemini.model, **st.session_state.cfg.webrtcvad)
        else: # textまたはaiモードの場合、asrはNoneのまま
            st.session_state.asr = None


        # 音声合成の設定
        st.session_state.engines = {
            "voicevox": VoiceVox(),
            "coeiroink": CoeiroInk(),
            "aivisspeech": AivisSpeech(),
        }
        st.session_state.ai_config = {ai["name"]: ai["voice"] for ai in st.session_state.cfg.chat.ai}
        if user_input_mode == "ai":
            st.session_state.ai_config[st.session_state.llmcfg.user_name] = st.session_state.cfg.chat.user.voice

        # 初期メッセージとターン設定
        st.session_state.history = [
            {"name": st.session_state.llmcfg.format(item["name"]), "content": st.session_state.llmcfg.format(item["content"])}
            for item in st.session_state.cfg.chat.initial_message
        ]
        st.session_state.turn = st.session_state.llmcfg.format(st.session_state.cfg.chat.initial_turn)

        # 再生・合成用のワーカーを起動
        async def playback_worker_streamlit():
            while True:
                data, sr = await st.session_state.playback_queue.get()
                if st.session_state.asr is not None:
                    # sd.pause() はブロッキング操作なので asyncio.to_thread で実行
                    await asyncio.to_thread(st.session_state.asr.pause)
                # sd.play() と sd.wait() もブロッキング操作なので asyncio.to_thread で実行
                await asyncio.to_thread(sd.play, data, sr)
                await asyncio.to_thread(sd.wait)
                if st.session_state.asr is not None:
                    # sd.resume() もブロッキング操作なので asyncio.to_thread で実行
                    await asyncio.to_thread(st.session_state.asr.resume)
                st.session_state.playback_queue.task_done()

        async def synthesis_worker_streamlit():
            while True:
                name, text_segment = await st.session_state.synthesis_queue.get()
                cfg_voice = st.session_state.ai_config[name]
                if cfg_voice["engine"] is not None:
                    tts = st.session_state.engines[cfg_voice["engine"]]
                    # synthesize_async は既に非同期なので asyncio.to_thread は不要
                    data, sr = await tts.synthesize_async(text_segment, **cfg_voice["config"])
                    await st.session_state.playback_queue.put((data, sr))
                st.session_state.synthesis_queue.task_done()

        # ワーカーをバックグラウンドで実行
        if not st.session_state.workers_started:
            # 新しいイベントループを作成し、別スレッドで実行
            st.session_state.async_loop = asyncio.new_event_loop()
            t = threading.Thread(target=run_async_loop_in_thread, args=(st.session_state.async_loop,))
            t.daemon = True # スレッドがプログラムの終了を妨げないようにする
            t.start()

            # 別スレッドのイベントループにワーカータスクをスケジュール
            st.session_state.async_loop.call_soon_threadsafe(
                st.session_state.async_loop.create_task, playback_worker_streamlit()
            )
            st.session_state.async_loop.call_soon_threadsafe(
                st.session_state.async_loop.create_task, synthesis_worker_streamlit()
            )
            st.session_state.workers_started = True

    st.sidebar.write("設定ファイルがロードされました。")

# チャット履歴の表示
for message in st.session_state.history:
    with st.chat_message(message["name"]):
        st.write(message["content"])

# ユーザー入力
if st.session_state.cfg:
    user_input_mode = st.session_state.cfg.chat.user.input
    user_name = st.session_state.llmcfg.user_name

    if st.session_state.turn == user_name and user_input_mode != "ai":
        if user_input_mode == "text":
            user_message = st.chat_input("メッセージを入力してください")
            if user_message:
                # ユーザーメッセージを履歴に追加し、即座に表示
                st.session_state.history.append({"name": user_name, "content": user_message})
                with st.chat_message(user_name):
                    st.write(user_message)
                try:
                    st.session_state.turn = st.session_state.llms.get_next_speaker(st.session_state.history, except_names=[user_name])
                except Exception as e:
                    st.error(f"LLMエラー: 次の話し手を決定できませんでした。エラー: {e}")
                    # LLMが失敗した場合、ユーザーのターンを維持
                    st.session_state.turn = user_name
                st.rerun()
        elif user_input_mode in ["vosk", "whisper", "gemini"]:
            # ASRオブジェクトが初期化されていることを確認してから使用
            if st.session_state.asr is not None:
                st.write("マイク入力モードです。話してください...")
                # 録音中ではない場合のみ録音を開始
                if not st.session_state.recording_in_progress:
                    st.session_state.recording_in_progress = True # 録音中フラグを立てる
                    with st.spinner("録音中...話してください"):
                        user_audio_text = None
                        try:
                            # audio_inputはブロッキング操作なので asyncio.to_thread で実行
                            user_audio_text = asyncio.run(asyncio.to_thread(st.session_state.asr.audio_input))
                            if user_audio_text:
                                # ユーザーメッセージを履歴に追加し、即座に表示
                                st.session_state.history.append({"name": user_name, "content": user_audio_text})
                                with st.chat_message(user_name):
                                    st.write(user_audio_text)
                            else:
                                st.warning("音声が認識されませんでした。")
                        except Exception as e:
                            st.error(f"音声認識エラー: {e}") # ASRのエラーはここで捕捉
                        finally:
                            st.session_state.recording_in_progress = False # 録音中フラグを解除
                            # ASRが成功した場合のみ次の話者を決定
                            if user_audio_text:
                                try:
                                    st.session_state.turn = st.session_state.llms.get_next_speaker(st.session_state.history, except_names=[user_name])
                                except Exception as e:
                                    st.error(f"LLMエラー: 次の話し手を決定できませんでした。エラー: {e}")
                                    # LLMが失敗した場合、ユーザーのターンを維持
                                    st.session_state.turn = user_name
                            st.rerun() # UIを更新して次のターンに進める
            else:
                st.warning(f"設定ファイルで指定された音声入力モード '{user_input_mode}' に対応する音声認識エンジンが初期化されていません。設定を確認してください。")
    else: # AIのターン
        if st.session_state.cfg:
            with st.chat_message(st.session_state.turn):
                message_placeholder = st.empty()
                st.session_state.current_ai_response = "" # AI応答を初期化
                text_queue = asyncio.Queue() # このキューはメインスレッドのイベントループで処理される

                async def process_text_queue_streamlit():
                    answer_segment = ""
                    while True:
                        chunk = await text_queue.get()
                        if chunk is None:
                            break
                        st.session_state.current_ai_response += chunk.content # session_stateを直接更新
                        answer_segment += chunk.content
                        message_placeholder.write(st.session_state.current_ai_response + "▌") # カーソル表示
                        if answer_segment and answer_segment[-1] in st.session_state.cfg.chat.streaming_voice_output:
                            # 合成キューへの投入は別スレッドのイベントループにスケジュール
                            st.session_state.async_loop.call_soon_threadsafe(
                                st.session_state.synthesis_queue.put_nowait, (st.session_state.turn, answer_segment)
                            )
                            answer_segment = ""
                        text_queue.task_done()
                    if answer_segment: # 残りのテキストを処理
                        st.session_state.async_loop.call_soon_threadsafe(
                            st.session_state.synthesis_queue.put_nowait, (st.session_state.turn, answer_segment)
                        )
                    message_placeholder.write(st.session_state.current_ai_response) # カーソルを削除

                async def generate_text_streamlit():
                    utter_chain = st.session_state.llms.get_utter_chain()
                    utter_prompt_vars = {"speaker": st.session_state.turn, "messages": history_to_text(st.session_state.history)}
                    for chunk in utter_chain.stream(utter_prompt_vars):
                        # メインスレッドのtext_queueにチャンクを投入
                        await text_queue.put(chunk)
                    await text_queue.put(None) # ストリーム終了の合図

                # AIのターン全体のロジックを非同期関数として定義し、asyncio.runで実行
                async def run_ai_turn_logic():
                    # テキスト処理タスクとテキスト生成タスクを開始
                    processing_task = asyncio.create_task(process_text_queue_streamlit())
                    generate_text_task = asyncio.create_task(generate_text_streamlit())

                    # 両方のタスクが完了するのを待つ
                    await generate_text_task
                    await processing_task

                # AIのターンロジックをメインスレッドのイベントループで実行
                asyncio.run(run_ai_turn_logic())

                st.session_state.history.append({"name": st.session_state.turn, "content": st.session_state.current_ai_response})

                # 画像生成
                if len(st.session_state.history) % st.session_state.cfg.chat.image.interval == 0:
                    st.write("画像を生成中...")
                    # generate_imageはブロッキング操作なので asyncio.to_thread で実行
                    asyncio.run(asyncio.to_thread(
                        st.session_state.image_generator.generate_image,
                        st.session_state.history,
                        st.session_state.cfg.chat.image.edit
                    ))
                    if st.session_state.image_generator.image_data:
                        st.image(st.session_state.image_generator.image_data, caption="生成された画像")
                    else:
                        st.warning("画像の生成に失敗しました。")

                try:
                    st.session_state.turn = st.session_state.llms.get_next_speaker(st.session_state.history, except_names=[st.session_state.turn])
                except Exception as e:
                    st.error(f"LLMエラー: 次の話し手を決定できませんでした。エラー: {e}")
                    # LLMが失敗した場合、ユーザーのターンに戻す
                    st.session_state.turn = user_name
                st.rerun()
else:
    st.info("左側のサイドバーから設定ファイル (invoke-utf8.yaml) をアップロードしてください。")
