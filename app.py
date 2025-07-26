import streamlit as st
import sounddevice as sd
from invoke.config import Config
import yaml
import numpy as np
import os
import traceback

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

# --- 非同期関連のキューやワーカースレッドは全て削除 ---

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
if "recording_in_progress" not in st.session_state:
    st.session_state.recording_in_progress = False


# StreamlitのUI
st.title("AIチャットアプリケーション")

# 設定ファイルのパス (app.pyと同じディレクトリにあると仮定)
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "invoke-utf8.yaml")

# 設定ファイルを自動で読み込む
if st.session_state.cfg is None:
    try:
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        st.session_state.cfg = Config(config_data)
        st.sidebar.success(f"設定ファイル '{os.path.basename(CONFIG_FILE_PATH)}' を読み込みました。")
    except FileNotFoundError:
        st.error(f"エラー: 設定ファイル '{os.path.basename(CONFIG_FILE_PATH)}' が見つかりません。")
        st.stop()
    except yaml.YAMLError as e:
        st.error(f"エラー: 設定ファイルの解析に失敗しました。YAML形式を確認してください: {e}")
        st.stop()
    except Exception as e:
        st.error(f"エラー: 設定ファイルの読み込み中に予期せぬエラーが発生しました: {e}")
        st.stop()

# 初期化処理 (設定ファイルが読み込まれた後のみ実行)
if st.session_state.cfg is not None and st.session_state.llmcfg is None:
    print("--- アプリケーションの初期化を開始 ---")
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
    else:
        st.session_state.asr = None

    # 音声合成エンジンの同期的な初期化
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
    print("--- アプリケーションの初期化が完了 ---")


# チャット履歴の表示
for message in st.session_state.history:
    with st.chat_message(message["name"]):
        st.write(message["content"])

# ユーザー入力
if st.session_state.cfg:
    user_input_mode = st.session_state.cfg.chat.user.input
    user_name = st.session_state.llmcfg.user_name

    if st.session_state.turn == user_name and user_input_mode != "ai":
        with st.chat_message(user_name):
            user_message_placeholder = st.empty()

        if user_input_mode == "text":
            user_message = st.chat_input("メッセージを入力してください")
            if user_message:
                st.session_state.history.append({"name": user_name, "content": user_message})
                user_message_placeholder.write(user_message)
                with st.spinner("次の話し手を決定中..."):
                    try:
                        st.session_state.turn = st.session_state.llms.get_next_speaker(st.session_state.history, except_names=[user_name])
                    except Exception as e:
                        st.error(f"LLMエラー: 次の話し手を決定できませんでした。エラー: {e}")
                        st.session_state.turn = user_name
                st.rerun()
        elif user_input_mode in ["vosk", "whisper", "gemini"]:
            if st.session_state.asr is None:
                st.warning(f"音声認識エンジン '{user_input_mode}' が初期化されていません。")
                st.stop()

            user_message_placeholder.write("マイク入力モードです。話してください...")
            if not st.session_state.recording_in_progress:
                st.session_state.recording_in_progress = True
                user_audio_text = None
                try:
                    with st.spinner("録音中..."):
                        user_audio_text = st.session_state.asr.audio_input()
                    if user_audio_text:
                        st.session_state.history.append({"name": user_name, "content": user_audio_text})
                        user_message_placeholder.write(user_audio_text)
                    else:
                        user_message_placeholder.warning("音声が認識されませんでした。")
                except Exception as e:
                    user_message_placeholder.error(f"音声認識エラー: {e}")
                finally:
                    st.session_state.recording_in_progress = False
                    if user_audio_text:
                        with st.spinner("次の話し手を決定中..."):
                            try:
                                st.session_state.turn = st.session_state.llms.get_next_speaker(st.session_state.history, except_names=[user_name])
                            except Exception as e:
                                st.error(f"LLMエラー: 次の話し手を決定できませんでした。エラー: {e}")
                                st.session_state.turn = user_name
                    else:
                        st.session_state.turn = user_name
                    st.rerun()
    # --- ここからが修正箇所 ---
    else:  # AIのターン
        if st.session_state.cfg:
            with st.spinner("AIが考え中..."):
                with st.chat_message(st.session_state.turn):
                    message_placeholder = st.empty()
                    
                    full_response = ""
                    answer_segment = ""
                    
                    # 音声合成と再生を行う同期関数
                    def synthesize_and_play(text_to_speak, speaker_name):
                        print(f"[{speaker_name}]が'{text_to_speak}'を合成・再生します。")
                        ai_voice_config = st.session_state.ai_config.get(speaker_name)
                        if not ai_voice_config or not ai_voice_config.get("engine"):
                            print("音声設定が見つからないため、スキップします。")
                            return

                        engine_name = ai_voice_config["engine"]
                        tts_engine = st.session_state.engines[engine_name]
                        
                        try:
                            # synthesize_asyncではなく、同期的なsynthesizeを使用
                            data, sr = tts_engine.synthesize(text_to_speak, **ai_voice_config["config"])
                            
                            if data is not None and sr is not None and data.size > 0:
                                print("合成成功。再生を開始します...")
                                # ASR（音声認識）を一時停止
                                if st.session_state.asr is not None:
                                    st.session_state.asr.pause()
                                
                                sd.play(data, sr, blocking=True) # blocking=Trueで再生完了を待つ
                                
                                # ASRを再開
                                if st.session_state.asr is not None:
                                    st.session_state.asr.resume()
                                print("再生完了。")
                            else:
                                print("合成に失敗したか、無音データが生成されました。")
                        except Exception as e:
                            print(f"音声合成または再生中にエラーが発生しました: {e}")
                            traceback.print_exc()

                    # LLMチェーンを準備
                    utter_chain = st.session_state.llms.get_utter_chain()
                    utter_prompt_vars = {"speaker": st.session_state.turn, "messages": history_to_text(st.session_state.history)}

                    # 同期的なループでLLMからのストリームを処理
                    for chunk in utter_chain.stream(utter_prompt_vars):
                        chunk_content = chunk.content
                        full_response += chunk_content
                        answer_segment += chunk_content
                        message_placeholder.write(full_response + "▌") # UIを更新

                        # 区切り文字が来たら、そのセグメントを音声合成・再生
                        if answer_segment and answer_segment[-1] in st.session_state.cfg.chat.streaming_voice_output:
                            synthesize_and_play(answer_segment, st.session_state.turn)
                            answer_segment = "" # セグメントをリセット
                    
                    # ループ後に残ったテキストを処理
                    if answer_segment:
                        synthesize_and_play(answer_segment, st.session_state.turn)

                    # 最終的なテキストをUIに表示
                    message_placeholder.write(full_response)
                    st.session_state.history.append({"name": st.session_state.turn, "content": full_response})

                    # 画像生成
                    if len(st.session_state.history) % st.session_state.cfg.chat.image.interval == 0:
                        st.write("画像を生成中...")
                        st.session_state.image_generator.generate_image(
                            st.session_state.history,
                            st.session_state.cfg.chat.image.edit
                        )
                        if st.session_state.image_generator.image_data:
                            st.image(st.session_state.image_generator.image_data, caption="生成された画像")
                        else:
                            st.warning("画像の生成に失敗しました。")

            # 次のターンへ
            try:
                st.session_state.turn = st.session_state.llms.get_next_speaker(st.session_state.history, except_names=[st.session_state.turn])
            except Exception as e:
                st.error(f"LLMエラー: 次の話し手を決定できませんでした。エラー: {e}")
                st.session_state.turn = user_name
            st.rerun()
    # --- ここまでが修正箇所 ---