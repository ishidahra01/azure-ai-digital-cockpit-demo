import streamlit as st
import json
import os
from dotenv import load_dotenv
import re

# ---- Utils ----
def clean_json_text(text):
    # ```json や ``` を取り除く
    # ```json\n～``` の形式をマッチさせる
    code_block_pattern = r"```(?:json)?\n(.*?)```"
    
    # 正規表現で中身だけ抽出
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()  # 中身だけ返す

    # コードブロックじゃなければそのまま返す
    return text.strip()



# ---- ここで環境変数をロードする例（必要に応じてカスタマイズ）----
load_dotenv(override=True)

# ---- ここから Azure のクライアントなどを呼び出す場合のサンプル ----
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage

# 環境認識エージェントのクライアント (例)
def get_env_agent_client():
    return ChatCompletionsClient(
        endpoint=os.environ["AZURE_INFERENCE_ENDPOINT_PHI35_VISION"],
        credential=AzureKeyCredential(os.environ["AZURE_INFERENCE_CREDENTIAL_PHI35_VISION"]),
    )

# 提案エージェントのクライアント (例)
def get_proposal_agent_client():
    return ChatCompletionsClient(
        endpoint=os.environ["AZURE_INFERENCE_ENDPOINT_PHI35"],
        credential=AzureKeyCredential(os.environ["AZURE_INFERENCE_CREDENTIAL_PHI35"]),
    )

############################################
# 3) Temperature Control Agent（処理エージェント）
############################################
# ここでは例として「function calling」用のエンドポイントを呼ぶ関数を用意しています。
# もちろん ChatCompletionsClient でも構いませんが、
# ノートブック例では get_azureml_inference() というHTTP経由の呼び出しをしていました。
# 必要に応じて、環境に合わせて書き換えてください。

import urllib.request
import ssl

def allowSelfSignedHttps(allowed):
    # self-signed certificate対応（環境に応じて不要なら削除）
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

def call_temperature_control_agent(messages):
    """
    例: ノートブック中の get_azureml_inference(messages) 相当。
    温度制御用エージェントに問い合わせして、最終的に
    <tool_call>{"name": ..., "arguments": {...}}</tool_call>
    形式で返ってくる想定。
    """
    allowSelfSignedHttps(True)

    import json
    body = {
        "input_data": messages,
        "params": {
            "temperature": 0.1,
            "max_new_tokens": 2048,
            "do_sample": True,
            "return_full_text": False
        }
    }
    body_str = json.dumps(body).encode("utf-8")

    # 環境変数から読み込む or 直書き（サンプル）
    url = os.environ.get("AZURE_INFERENCE_ENDPOINT_PHI35_TEMPERATURE", "<YOUR_TEMPERATURE_CTRL_ENDPOINT>")
    api_key = os.environ.get("AZURE_INFERENCE_CREDENTIAL_PHI35_TEMPERATURE", "<YOUR_KEY>")

    if not api_key or not url:
        raise ValueError("Temperature Control Agent のendpointまたはキーが設定されていません。")

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key
    }

    req = urllib.request.Request(url, data=body_str, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            result = resp.read().decode("utf-8")
            parsed = json.loads(result)
            return parsed.get("result", "")  # {"result": "..."} の想定
    except urllib.error.HTTPError as e:
        st.error(f"Temperature Control Agent呼び出しでHTTPエラー: {e} {e.read().decode('utf-8')}")
        return ""

# タイトル
st.title("車載AIエージェント デモ")

# 説明文
st.write("""
このデモは、車両の状況やユーザーの好み、利用可能なAPI情報をもとに  
エージェントが連携し、最適なアクションを決定・実行する流れを確認するものです。
""")

# 処理の流れ
st.header("処理フロー")

st.markdown("""
### 1. 環境認識エージェント  
- 車両情報（位置、速度、気象など）や  
  ユーザー情報（好み、状態）を入力として受け取り、  
  **現在のコンテキスト**を把握・整理します。

### 2. 提案エージェント  
- 認識されたコンテキストをもとに、  
  ユーザーに対してどのようなアクションを提案すべきかを検討します。  
- 利用可能なAPIの一覧と組み合わせて、  
  **実行候補となるAPIアクション**を生成します。

### 3. 処理エージェント  
- 提案されたアクションの中から実行可能なものを選択し、  
  実際のAPIコールとして構築・実行します。  
- 最終的に、**どのAPIをどのパラメータで呼び出すか**を決定し、  
  実行結果を表示します。
""")

# 補足説明
st.write("""
左のメニューから入力データを変更し、異なる状況に応じたエージェントの判断を試すことができます。
""")

# --- デフォルト例 JSON ---
default_vehicle_data = """{
  "InteriorTemperature": 25.5,
  "ExteriorTemperature": 30.2,
  "Humidity": 60,
  "Weather": "Sunny",
  "Speed": "80km/h",
  "Acceleration": "1.2m/s²",
  "SteeringAngle": "15°",
  "FuelLevel": 45,
  "EngineRPM": 2500,
  "BatteryVoltage": 12.8
}"""

default_driver_state = """{
  "HeartRate": 72,
  "EyeStatus": "Open",
  "Yawning": false,
  "AttentionLevel": "High"
}"""

default_window_status = """{
  "FrontLeft": "Closed",
  "FrontRight": "Closed",
  "RearLeft": "Open",
  "RearRight": "Closed"
}"""

default_door_status = """{
  "FrontLeft": "Closed",
  "FrontRight": "Closed",
  "RearLeft": "Closed",
  "RearRight": "Closed",
  "Trunk": "Closed"
}"""

default_can_data = """{
  "BrakePressure": 2.3,
  "ThrottlePosition": 25.4,
  "GearPosition": "Drive",
  "TurnSignal": "Off"
}"""

default_user_preferences = """{
  "preferred_temperature": 20,
  "preferred_humidity": 40,
  "preferred_noise_level": "low",
  "preferred_air_quality": "good",
  "preferred_lighting": "dim",
  "preferred_music": "cheerful",
  "favorite_places": ["beach"]
}"""

default_agent_list = """[
  "temperature_control_agent",
  "window_control_agent",
  "entertainment_control_agent"
]"""

default_api_list = """[
    {
        "type": "function",
        "function": {
            "name": "set_cabin_temperature",
            "description": "Set cabin temperature to specified degree.",
            "strict": true,
            "parameters": {
                "type": "object",
                "properties": {
                    "temperature": {
                        "type": "number",
                        "description": "Target temperature to be set in the cabin."
                    }
                },
                "required": ["temperature"],
                "additionalProperties": false
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "auto_adjust_cabin_environment",
            "description": "Automatically adjust the cabin environment.",
            "strict": true
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_fan_speed",
            "description": "Set fan speed to specified level.",
            "strict": true,
            "parameters": {
                "type": "object",
                "properties": {
                    "fan_speed": {
                        "type": "integer",
                        "enum": [1, 2, 3, 4, 5],
                        "description": "Fan speed level (1: Low, 5: High)."
                    }
                },
                "required": ["fan_speed"],
                "additionalProperties": false
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_seat_heater_level",
            "description": "Set seat heater to specified level.",
            "strict": true,
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3],
                        "description": "Seat heater level (0: Off, 3: Max)."
                    }
                },
                "required": ["level"],
                "additionalProperties": false
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_seat_ventilation_level",
            "description": "Set seat ventilation to specified level.",
            "strict": true,
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3],
                        "description": "Seat ventilation level (0: Off, 3: Max)."
                    }
                },
                "required": ["level"],
                "additionalProperties": false
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_steering_heater_level",
            "description": "Set steering heater to specified level.",
            "strict": true,
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "enum": [0, 1, 2],
                        "description": "Steering heater level (0: Off, 2: Max)."
                    }
                },
                "required": ["level"],
                "additionalProperties": false
            }
        }
    }
]"""

# --- サイドバーやエリアで JSON 入力を受け付ける ---
st.sidebar.write("## 車両データ入力")
vehicle_data_str = st.sidebar.text_area("VehicleData (JSON)", default_vehicle_data, height=150)
driver_state_str = st.sidebar.text_area("DriverState (JSON)", default_driver_state, height=150)
window_status_str = st.sidebar.text_area("WindowStatus (JSON)", default_window_status, height=150)
door_status_str = st.sidebar.text_area("DoorStatus (JSON)", default_door_status, height=150)
can_data_str = st.sidebar.text_area("CANData (JSON)", default_can_data, height=150)

st.sidebar.write("## ユーザ好み・APIリスト")
user_prefs_str = st.sidebar.text_area("User Preferences (JSON)", default_user_preferences, height=150)
agent_list_str = st.sidebar.text_area("利用可能なエージェント一覧 (JSON)", default_agent_list, height=100)
api_list_str = st.sidebar.text_area("利用可能なAPI一覧 (JSON)", default_api_list, height=100)

# 送信ボタン
if st.sidebar.button("送信"):
    # JSONパース
    try:
        VehicleData = json.loads(vehicle_data_str)
        DriverState = json.loads(driver_state_str)
        WindowStatus = json.loads(window_status_str)
        DoorStatus = json.loads(door_status_str)
        CANData = json.loads(can_data_str)
        user_preferences = json.loads(user_prefs_str)
        available_action_agents = json.loads(agent_list_str)
    except Exception as e:
        st.error(f"JSONの読み込みに失敗しました: {e}")
        st.stop()

    # --------------------
    # 1) 環境認識エージェント呼び出し
    # --------------------
    env_agent_system_message = """\
You are an Environmental Recognition Agent.
Based on the data collected from the car's sensors, generate JSON-format data that represents the current state of the vehicle and driver.
You MUST output ONLY the JSON OUTPUT. You MUST NOT include any other additional comment, text, code block.
"""

    env_agent_user_message = f"""\
# Input
"VehicleData": {VehicleData},
"DriverState": {DriverState},
"WindowStatus": {WindowStatus},
"DoorStatus": {DoorStatus},
"CANData": {CANData}
"""

    with st.spinner("環境認識エージェント呼び出し中..."):
        try:
            env_client = get_env_agent_client()

            env_agent_response = env_client.complete(
                messages=[
                    SystemMessage(content=env_agent_system_message),
                    UserMessage(content=env_agent_user_message),
                ],
            )
            env_result_text = env_agent_response.choices[0].message.content
            env_result_text = clean_json_text(env_result_text)
            st.subheader("① 環境認識エージェントの出力")
            st.code(env_result_text, language="json")
        except Exception as e:
            st.error(f"環境認識エージェント呼び出しに失敗: {e}")
            st.stop()

    # --------------------
    # 2) 提案エージェント呼び出し
    # --------------------
    proposal_agent_system_message = """\
You are an agent tasked with leveraging in-car systems to provide an optimal and comfortable environment for the user.
Based on the current state, user preferences, and available action agents, suggest an ideal comfortable state.
Then, determine the appropriate agent and the instructions needed to achieve the ideal state.
You MUST output ONLY the JSON OUTPUT as shown in the output example. You MUST NOT include any other additional comment, text, code block.
"""

    output_example = {
        "target_agent": "Temperature Control Agent",
        "ideal_state": {
            "temperature": 26,
            "humidity": 50
        },
        "instructions": {
            "action": "set_temperature",
            "value": 26
        }
    }

    proposal_agent_user_message = f"""\
# Input

## Current Environment State
{env_result_text}

## Available Action Agents
{available_action_agents}

## User Preferences
{user_preferences}

# Output example:
{json.dumps(output_example, ensure_ascii=False, indent=2)}
"""

    with st.spinner("提案エージェント呼び出し中..."):
        try:
            proposal_client = get_proposal_agent_client()

            proposal_agent_response = proposal_client.complete(
                messages=[
                    SystemMessage(content=proposal_agent_system_message),
                    UserMessage(content=proposal_agent_user_message),
                ],
            )
            proposal_result_text = proposal_agent_response.choices[0].message.content
            proposal_result_text = clean_json_text(proposal_result_text)
            st.subheader("② 提案エージェントの出力")
            st.code(proposal_result_text, language="json")
        except Exception as e:
            st.error(f"提案エージェント呼び出しに失敗: {e}")
            st.stop()

    # --------------------
    # 3) Processing Agent: Temperature Control Agentの呼び出し
    # --------------------
    st.subheader("③ Processing Agent (Temperature Control Agent) 呼び出し")
    st.write("提案エージェントの結果JSONから `target_agent` が `temperature_control_agent` なら、さらに詳細なAPIコール方法を取得。")

    try:
        proposal_json = json.loads(proposal_result_text)
    except json.JSONDecodeError as e:
        st.error(f"提案結果のjson解析に失敗: {e}")
        st.stop()

    target_agent = proposal_json.get("target_agent", "")
    if target_agent == "temperature_control_agent":
        # 次にTemperature Control Agentへ問い合わせて、具体的な function call を出してもらう

        temperature_control_agent_system_message = f"""\
You are a function calling AI model. 
You are provided with function signatures within <tools> </tools> XML tags.
You may call one or more functions to assist with the user query.
Don't make assumptions about what values to plug into functions.
Ask for clarification if a user request is ambiguous.

<tools>
{api_list_str}
</tools>

For each function call return a json object with function name and arguments within <tool_call> </tool_call> tags, 
with the schema:
<tool_call>
{{'arguments': <args-dict>, 'name': <function-name>}}
</tool_call>
"""

        temperature_control_agent_user_message = f"""\
# Input

## Proposal Contents from the Proposal Agent
{json.dumps(proposal_json, ensure_ascii=False)}

## User Preferences
{json.dumps(user_preferences, ensure_ascii=False)}
"""

        messages_for_temp_ctrl = [
            {"role": "system", "content": temperature_control_agent_system_message},
            {"role": "user", "content": temperature_control_agent_user_message},
        ]

        with st.spinner("Temperature Control Agent呼び出し中..."):
            temp_ctrl_response = call_temperature_control_agent(messages_for_temp_ctrl)
            st.write("#### Temperature Control Agentの最終出力")
            st.code(temp_ctrl_response, language="json")

        st.write("""
        - 上記の出力には、`<tool_call> {...} </tool_call>` 形式で
          実際に呼ぶべき車両APIとパラメータが入っている想定です。
        - あとはこのJSONをパースし、例えば `"name": "set_cabin_temperature", "arguments": {"temperature": 20}` といった情報を取り出して実際の車両APIを呼び出します。
        """)

    else:
        st.info(f"今回の提案では temperature_control_agent ではなく `{target_agent}` が選ばれました。該当エージェントの処理を実装してください。")