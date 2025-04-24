import streamlit as st
import json
import os
from dotenv import load_dotenv
import re
from opentelemetry import trace
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, CompletionsFinishReason
from azure.core.credentials import AzureKeyCredential
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry.trace import get_tracer
import time

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

def calculate_json_size(obj):
    """Calculate the size of a JSON object in bytes"""
    if isinstance(obj, str):
        return len(obj.encode('utf-8'))
    return len(json.dumps(obj).encode('utf-8'))

def get_truncated_string(text, max_length=500):
    """Truncate text for logging purposes"""
    if len(text) > max_length:
        return text[:max_length] + "... [truncated]"
    return text

# ---- ここで環境変数をロードする例（必要に応じてカスタマイズ）----
load_dotenv(override=True)

# Azure Monitorのトレース設定（必要に応じてカスタマイズ）
os.environ['AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED'] = 'true'

tracer = get_tracer(__name__)

configure_azure_monitor(connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"))

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

# 新しく追加する関数: 環境認識エージェント呼び出し
def call_environment_recognition_agent(vehicle_data, driver_state, window_status, door_status, can_data):
    with tracer.start_as_current_span("call_environment_recognition_agent") as span:
        span.set_attribute("agent", "environment_recognition")
        
        # Input metrics
        span.set_attribute("input.vehicle_data.size_bytes", calculate_json_size(vehicle_data))
        span.set_attribute("input.driver_state.size_bytes", calculate_json_size(driver_state))
        span.set_attribute("input.window_status.size_bytes", calculate_json_size(window_status))
        span.set_attribute("input.door_status.size_bytes", calculate_json_size(door_status))
        span.set_attribute("input.can_data.size_bytes", calculate_json_size(can_data))
        
        # Log key input values for debugging
        span.set_attribute("input.vehicle.interior_temp", vehicle_data.get("InteriorTemperature", "N/A"))
        span.set_attribute("input.vehicle.exterior_temp", vehicle_data.get("ExteriorTemperature", "N/A"))
        span.set_attribute("input.vehicle.weather", vehicle_data.get("Weather", "N/A"))
        span.set_attribute("input.driver.attention_level", driver_state.get("AttentionLevel", "N/A"))
        
        env_agent_system_message = """\
        You are an Environmental Recognition Agent.
        Based on the data collected from the car's sensors, generate JSON-format data that represents the current state of the vehicle and driver.
        You MUST output ONLY the JSON OUTPUT. You MUST NOT include any other additional comment, text, code block.
        """

        env_agent_user_message = f"""\
        # Input
        "VehicleData": {vehicle_data},
        "DriverState": {driver_state},
        "WindowStatus": {window_status},
        "DoorStatus": {door_status},
        "CANData": {can_data}
        """
        
        # Set message sizes
        span.set_attribute("prompt.system_message.size_bytes", len(env_agent_system_message.encode('utf-8')))
        span.set_attribute("prompt.user_message.size_bytes", len(env_agent_user_message.encode('utf-8')))
        
        span.add_event("sending_request_to_env_agent", {
            "message": get_truncated_string(env_agent_user_message),
            "timestamp_ms": int(time.time() * 1000)
        })
        
        start_time = time.time()
        try:
            env_client = get_env_agent_client()
            
            # Record start of API call
            api_call_start = time.time()
            span.add_event("api_call_start", {"timestamp_ms": int(api_call_start * 1000)})
            
            env_agent_response = env_client.complete(
                messages=[
                    SystemMessage(content=env_agent_system_message),
                    UserMessage(content=env_agent_user_message),
                ],
            )
            
            # Record end of API call and calculate duration
            api_call_end = time.time()
            api_call_duration = api_call_end - api_call_start
            span.set_attribute("api_call.duration_ms", int(api_call_duration * 1000))
            
            env_result_text = env_agent_response.choices[0].message.content
            cleaned_result = clean_json_text(env_result_text)
            
            # Record response metadata
            span.set_attribute("response.raw_size_bytes", len(env_result_text.encode('utf-8')))
            span.set_attribute("response.cleaned_size_bytes", len(cleaned_result.encode('utf-8')))
            span.set_attribute("response.finish_reason", env_agent_response.choices[0].finish_reason.value if hasattr(env_agent_response.choices[0], 'finish_reason') else "unknown")
            
            try:
                # Try to parse as JSON to extract some key metrics
                result_json = json.loads(cleaned_result)
                span.set_attribute("response.json_keys", str(list(result_json.keys())))
                span.set_attribute("response.json_key_count", len(result_json.keys()))
            except:
                span.set_attribute("response.is_valid_json", False)
            
            end_time = time.time()
            span.set_attribute("total.duration_ms", int((end_time - start_time) * 1000))
            
            span.add_event("received_response_from_env_agent", {
                "result": get_truncated_string(cleaned_result),
                "timestamp_ms": int(time.time() * 1000)
            })
            
            return cleaned_result
        except Exception as e:
            end_time = time.time()
            span.set_attribute("total.duration_ms", int((end_time - start_time) * 1000))
            span.set_attribute("error.type", e.__class__.__name__)
            span.set_attribute("error.message", str(e))
            span.record_exception(e)
            span.set_status(trace.StatusCode.ERROR, str(e))
            raise e

# 新しく追加する関数: 提案エージェント呼び出し
def call_proposal_agent(env_result, available_agents, user_preferences):
    with tracer.start_as_current_span("call_proposal_agent") as span:
        span.set_attribute("agent", "proposal")
        
        # Input metrics
        span.set_attribute("input.env_result.size_bytes", calculate_json_size(env_result))
        span.set_attribute("input.available_agents.size_bytes", calculate_json_size(available_agents))
        span.set_attribute("input.user_preferences.size_bytes", calculate_json_size(user_preferences))
        
        # Log key user preferences for debugging
        span.set_attribute("input.user_prefs.preferred_temp", 
                         user_preferences.get("preferred_temperature", "N/A"))
        span.set_attribute("input.user_prefs.preferred_humidity", 
                         user_preferences.get("preferred_humidity", "N/A"))
        
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
        {env_result}

        ## Available Action Agents
        {available_agents}

        ## User Preferences
        {user_preferences}

        # Output example:
        {json.dumps(output_example, ensure_ascii=False, indent=2)}
        """
        
        # Message size metrics
        span.set_attribute("prompt.system_message.size_bytes", len(proposal_agent_system_message.encode('utf-8')))
        span.set_attribute("prompt.user_message.size_bytes", len(proposal_agent_user_message.encode('utf-8')))
        
        span.add_event("sending_request_to_proposal_agent", {
            "message": get_truncated_string(proposal_agent_user_message),
            "timestamp_ms": int(time.time() * 1000)
        })
        
        start_time = time.time()
        try:
            proposal_client = get_proposal_agent_client()
            
            # Record start of API call
            api_call_start = time.time()
            span.add_event("api_call_start", {"timestamp_ms": int(api_call_start * 1000)})
            
            proposal_agent_response = proposal_client.complete(
                messages=[
                    SystemMessage(content=proposal_agent_system_message),
                    UserMessage(content=proposal_agent_user_message),
                ],
            )
            
            # Record end of API call and calculate duration
            api_call_end = time.time()
            api_call_duration = api_call_end - api_call_start
            span.set_attribute("api_call.duration_ms", int(api_call_duration * 1000))
            
            proposal_result_text = proposal_agent_response.choices[0].message.content
            cleaned_result = clean_json_text(proposal_result_text)
            
            # Response metrics
            span.set_attribute("response.raw_size_bytes", len(proposal_result_text.encode('utf-8')))
            span.set_attribute("response.cleaned_size_bytes", len(cleaned_result.encode('utf-8')))
            span.set_attribute("response.finish_reason", proposal_agent_response.choices[0].finish_reason.value if hasattr(proposal_agent_response.choices[0], 'finish_reason') else "unknown")
            
            try:
                # Try to parse as JSON and extract key metrics
                result_json = json.loads(cleaned_result)
                span.set_attribute("response.target_agent", result_json.get("target_agent", "unknown"))
                if "ideal_state" in result_json:
                    span.set_attribute("response.ideal_state", str(result_json["ideal_state"]))
                if "instructions" in result_json:
                    span.set_attribute("response.instructions", str(result_json["instructions"]))
            except:
                span.set_attribute("response.is_valid_json", False)
            
            end_time = time.time()
            span.set_attribute("total.duration_ms", int((end_time - start_time) * 1000))
            
            span.add_event("received_response_from_proposal_agent", {
                "result": get_truncated_string(cleaned_result),
                "timestamp_ms": int(time.time() * 1000)
            })
            
            return cleaned_result
        except Exception as e:
            end_time = time.time()
            span.set_attribute("total.duration_ms", int((end_time - start_time) * 1000))
            span.set_attribute("error.type", e.__class__.__name__)
            span.set_attribute("error.message", str(e))
            span.record_exception(e)
            span.set_status(trace.StatusCode.ERROR, str(e))
            raise e

# 新しく追加する関数: 処理エージェント（Temperature Control Agent）呼び出し
def call_processing_agent(proposal_json, user_preferences, api_list):
    with tracer.start_as_current_span("call_processing_agent") as span:
        span.set_attribute("agent", "processing")
        
        # Input metrics
        span.set_attribute("input.proposal_json.size_bytes", calculate_json_size(proposal_json))
        span.set_attribute("input.user_preferences.size_bytes", calculate_json_size(user_preferences))
        span.set_attribute("input.api_list.size_bytes", calculate_json_size(api_list))
        
        target_agent = proposal_json.get("target_agent", "")
        span.set_attribute("target_agent", target_agent)
        
        # Log proposal details
        if "ideal_state" in proposal_json:
            span.set_attribute("proposal.ideal_state", str(proposal_json["ideal_state"]))
        if "instructions" in proposal_json:
            span.set_attribute("proposal.instructions", str(proposal_json["instructions"]))
        
        if target_agent != "temperature_control_agent":
            span.add_event("skipping_temperature_control_agent", {
                "reason": f"Target agent is {target_agent}",
                "timestamp_ms": int(time.time() * 1000)
            })
            return None
            
        temperature_control_agent_system_message = f"""\
        You are a function calling AI model. 
        You are provided with function signatures within <tools> </tools> XML tags.
        You may call one or more functions to assist with the user query.
        Don't make assumptions about what values to plug into functions.
        Ask for clarification if a user request is ambiguous.

        <tools>
        {api_list}
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
        
        # Message size metrics
        span.set_attribute("prompt.system_message.size_bytes", len(temperature_control_agent_system_message.encode('utf-8')))
        span.set_attribute("prompt.user_message.size_bytes", len(temperature_control_agent_user_message.encode('utf-8')))
        
        span.add_event("sending_request_to_processing_agent", {
            "messages": get_truncated_string(str(messages_for_temp_ctrl)),
            "timestamp_ms": int(time.time() * 1000)
        })
        
        start_time = time.time()
        try:
            # Record start of API call
            api_call_start = time.time()
            span.add_event("api_call_start", {"timestamp_ms": int(api_call_start * 1000)})
            
            result = call_temperature_control_agent(messages_for_temp_ctrl)
            
            # Record end of API call and calculate duration
            api_call_end = time.time()
            api_call_duration = api_call_end - api_call_start
            span.set_attribute("api_call.duration_ms", int(api_call_duration * 1000))
            
            # Response metrics
            span.set_attribute("response.size_bytes", len(result.encode('utf-8')))
            
            # Try to extract function call information
            tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
            match = re.search(tool_call_pattern, result, re.DOTALL)
            if match:
                try:
                    function_call = json.loads(match.group(1).strip())
                    span.set_attribute("function_call.name", function_call.get("name", "unknown"))
                    span.set_attribute("function_call.arguments", str(function_call.get("arguments", {})))
                except:
                    span.set_attribute("function_call.parse_error", True)
            else:
                span.set_attribute("function_call.found", False)
            
            end_time = time.time()
            span.set_attribute("total.duration_ms", int((end_time - start_time) * 1000))
            
            span.add_event("received_response_from_processing_agent", {
                "result": get_truncated_string(result),
                "timestamp_ms": int(time.time() * 1000)
            })
            
            return result
        except Exception as e:
            end_time = time.time()
            span.set_attribute("total.duration_ms", int((end_time - start_time) * 1000))
            span.set_attribute("error.type", e.__class__.__name__)
            span.set_attribute("error.message", str(e))
            span.record_exception(e)
            span.set_status(trace.StatusCode.ERROR, str(e))
            raise e

# 全体をトレースするラッパー関数
def process_agent_workflow(vehicle_data, driver_state, window_status, door_status, can_data, user_preferences, available_agents, api_list):
    with tracer.start_as_current_span("process_agent_workflow") as span:
        span.set_attribute("workflow", "agent_chain")
        
        # Add workflow input metrics
        span.set_attribute("workflow.input.vehicle_data_size", calculate_json_size(vehicle_data))
        span.set_attribute("workflow.input.user_preferences_size", calculate_json_size(user_preferences))
        span.set_attribute("workflow.input.api_list_size", calculate_json_size(api_list))
        
        workflow_start_time = time.time()
        
        # 1. 環境認識エージェント呼び出し
        env_start_time = time.time()
        with st.spinner("環境認識エージェント呼び出し中..."):
            try:
                env_result = call_environment_recognition_agent(
                    vehicle_data, driver_state, window_status, door_status, can_data
                )
                env_end_time = time.time()
                span.set_attribute("step1.env_recognition.duration_ms", int((env_end_time - env_start_time) * 1000))
                span.set_attribute("step1.env_recognition.result_size", len(env_result.encode('utf-8')))
                
                st.subheader("① 環境認識エージェントの出力")
                st.code(env_result, language="json")
            except Exception as e:
                env_end_time = time.time()
                span.set_attribute("step1.env_recognition.duration_ms", int((env_end_time - env_start_time) * 1000))
                span.set_attribute("step1.env_recognition.error", str(e))
                
                st.error(f"環境認識エージェント呼び出しに失敗: {e}")
                span.record_exception(e)
                span.set_status(trace.StatusCode.ERROR, str(e))
                return None, None, None
        
        # 2. 提案エージェント呼び出し
        proposal_start_time = time.time()
        with st.spinner("提案エージェント呼び出し中..."):
            try:
                proposal_result = call_proposal_agent(
                    env_result, available_agents, user_preferences
                )
                proposal_end_time = time.time()
                span.set_attribute("step2.proposal.duration_ms", int((proposal_end_time - proposal_start_time) * 1000))
                span.set_attribute("step2.proposal.result_size", len(proposal_result.encode('utf-8')))
                
                st.subheader("② 提案エージェントの出力")
                st.code(proposal_result, language="json")
                
                try:
                    proposal_json = json.loads(proposal_result)
                    span.set_attribute("step2.proposal.target_agent", proposal_json.get("target_agent", "unknown"))
                except json.JSONDecodeError as e:
                    st.error(f"提案結果のjson解析に失敗: {e}")
                    span.set_attribute("step2.proposal.json_parse_error", str(e))
                    span.record_exception(e)
                    span.set_status(trace.StatusCode.ERROR, str(e))
                    return env_result, proposal_result, None
            except Exception as e:
                proposal_end_time = time.time()
                span.set_attribute("step2.proposal.duration_ms", int((proposal_end_time - proposal_start_time) * 1000))
                span.set_attribute("step2.proposal.error", str(e))
                
                st.error(f"提案エージェント呼び出しに失敗: {e}")
                span.record_exception(e)
                span.set_status(trace.StatusCode.ERROR, str(e))
                return env_result, None, None
        
        # 3. 処理エージェント呼び出し
        st.subheader("③ Processing Agent (Temperature Control Agent) 呼び出し")
        target_agent = proposal_json.get("target_agent", "")
        span.set_attribute("step3.target_agent", target_agent)
        
        processing_start_time = time.time()
        if target_agent == "temperature_control_agent":
            with st.spinner("Temperature Control Agent呼び出し中..."):
                try:
                    temp_ctrl_result = call_processing_agent(
                        proposal_json, user_preferences, api_list
                    )
                    processing_end_time = time.time()
                    span.set_attribute("step3.processing.duration_ms", int((processing_end_time - processing_start_time) * 1000))
                    span.set_attribute("step3.processing.result_size", len(temp_ctrl_result.encode('utf-8')) if temp_ctrl_result else 0)
                    
                    st.write("#### Temperature Control Agentの最終出力")
                    st.code(temp_ctrl_result, language="json")
                    
                    # Extract function call information for tracing
                    if temp_ctrl_result:
                        tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
                        match = re.search(tool_call_pattern, temp_ctrl_result, re.DOTALL)
                        if match:
                            try:
                                function_call = json.loads(match.group(1).strip())
                                span.set_attribute("step3.function_call.name", function_call.get("name", "unknown"))
                                span.set_attribute("step3.function_call.arguments", str(function_call.get("arguments", {})))
                            except:
                                span.set_attribute("step3.function_call.parse_error", True)
                    
                    st.write("""
                    - 上記の出力には、`<tool_call> {...} </tool_call>` 形式で
                      実際に呼ぶべき車両APIとパラメータが入っている想定です。
                    - あとはこのJSONをパースし、例えば `"name": "set_cabin_temperature", "arguments": {"temperature": 20}` といった情報を取り出して実際の車両APIを呼び出します。
                    """)
                except Exception as e:
                    processing_end_time = time.time()
                    span.set_attribute("step3.processing.duration_ms", int((processing_end_time - processing_start_time) * 1000))
                    span.set_attribute("step3.processing.error", str(e))
                    
                    st.error(f"Temperature Control Agent呼び出しに失敗: {e}")
                    span.record_exception(e)
                    span.set_status(trace.StatusCode.ERROR, str(e))
                    return env_result, proposal_result, None
        else:
            processing_end_time = time.time()
            span.set_attribute("step3.processing.duration_ms", int((processing_end_time - processing_start_time) * 1000))
            span.set_attribute("step3.processing.skipped", True)
            span.set_attribute("step3.processing.reason", f"Target agent is {target_agent}")
            
            st.info(f"今回の提案では temperature_control_agent ではなく `{target_agent}` が選ばれました。該当エージェントの処理を実装してください。")
            temp_ctrl_result = None
        
        workflow_end_time = time.time()
        span.set_attribute("workflow.total_duration_ms", int((workflow_end_time - workflow_start_time) * 1000))
        
        return env_result, proposal_result, temp_ctrl_result

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
        available_apis = json.loads(api_list_str)
    except Exception as e:
        st.error(f"JSONの読み込みに失敗しました: {e}")
        st.stop()

    # ワークフロー全体を実行
    process_agent_workflow(
        VehicleData, 
        DriverState,
        WindowStatus,
        DoorStatus,
        CANData,
        user_preferences,
        available_action_agents,
        available_apis
    )