# SLM Fine-tuning workshop with Azure AI Services

## LLM/SLM Training and Fine-tuning

Fine-tuning はモデルのカスタマイズ方法の1つの手法である。
以下はチューニング方法のサマリ。

| Fine-Tuning Method                          | Description                                                                 | Main use Cases                                            | Data needs                                                                     | Resource and time needs                                                | Updated parameters       |
|--------------------------------------------|----------------------------------------------------------------------------|----------------------------------------------------------|--------------------------------------------------------------------------------|-------------------------------------------------------------------------|--------------------------|
| **Continual Pretraining**                  | Continue **unsupervised** training of an LLM/SLM.                          | • Add domain-specific data <br> • Add up-to-date data <br> • Multilingual   | **Large** amount of **unlabeled** data. Smaller amount if it is an SLM.        | **Large** amount of resource and time. Less amount if it is an SLM.    | All parameters           |
| **Supervised Fine-tuning (SFT)**           | **Supervised** training on large amount of labeled data.                   | • Learn new behavior and style <br> • Learn domain-specific task            | **Large** amount of labeled data. Smaller amount if it is an SLM.              | **Large** amount of resource and time. Less amount if it is an SLM.    | All or the majority of parameters |
| **Parameter-Efficient Fine-Tuning (PEFT)** | **Efficient supervised** training while minimizing the amount of parameters updated and resources needed | • Learn new behavior and style <br> • Learn domain-specific task            | High quality labeled data, even **small** amount of data can start to show improvement | **Small** amount of resource and time compared to continual training and SFT. | Small amount of parameters |

## **Fine-tuning のフロー**

### **1. データを収集・準備する**
---
#### **十分なデータがある場合** 
**既存データを使用**: この場合は手持ちのデータセットを使えばいいため、データクリーニングやフォーマット整形（モデルが期待しているフォーマットへデータセットを変換する）が主要な作業となる。

#### **データが不足している場合**  
**合成データを生成**：
ドキュメントや画像など何らかの参照データが存在することを前提にして、合成データを生成する際の設計・チューニング観点について説明する。ジェネラルな情報は[こちらのドキュメント](https://azure.github.io/slm-innovator-lab/1_synthetic_data/)を参照。
- ドキュメントの前処理
	- PDFドキュメントをテキストと画像を抽出。
		- 抽出は Azure AI Document Intelligence を利用。テキストは階層構造抽出機能（マークダウン形式での抽出）、画像はバウンディングボックス検知機能を利用して、画像抽出処理を自前で実施。
			- [参照リンク](https://techcommunity.microsoft.com/blog/azure-ai-services-blog/unlocking-advanced-document-insights-with-azure-ai-document-intelligence/4109675)
  		- 画像をテキスト情報にするためにマルチモーダルモデル（gpt-4o）を利用する。
			- ドキュメント内に画像が多く含まれる場合（スライドなどのビジネスドキュメント）は、テキスト情報の大半を占めることになるため、影響度が大きい。プロンプトもドキュメントに特化した指示、前提情報を与えるべきである。

- Coverage Dataset (Seed)の作成
	- ユーザの指示とアシスタント（AI）の応答のペアを作成する。LLMで作成する場合は、LLMがデータを正確に扱えるようにテキストをチャンキングする。
		- チャンクはコンテキストのかたまりごとに分割するのがベスト。
		- ただし、チャンクをまたいだ情報をもとにして回答を作成する必要がある場合は、別途処理が必要。
			- 手動で元のテキスト情報から作成する。
			- チャンク化したデータ同士の関連性を表現する（グラフ構造、ベクトルＤＢなど）
	- 理想的なデータペアを作成する。
		- 可能であれば、ドメイン知識をもった専門家がデータセットを手動で作成する。
		- LLMを使うにしても、想定されるユーザの質問やアシスタントの理想的な回答内容、スタイル、ふるまいを定義しておくべきである。
			- これをプロンプトのFew-shotとしてあたえる。
	- データ品質をチェックする。
		- データが想定のフォーマットに従っているかや、内容が充足しているかを確認する。
		- 後続でデータ拡張を実施する場合は、このデータセットの内容によって、拡張されたデータ品質も左右されるため非常に重要な要素である。
		- データ品質も人間によるレビューが可能であれば理想的だが、難しい場合はLLMでの評価も選択肢の1つである。
			- Azure AI おいて、[データ品質の評価を支援する機能](https://learn.microsoft.com/en-us/azure/ai-studio/concepts/evaluation-approach-gen-ai)も提供している。

- データ拡張
	- タスクやデータセットの内容には依存するが、Fine-tuningには数1000件以上のデータセットが必要になるケースが多い。
	- すべて手動で作成するのは現実的ではないため、LLMを使ってデータセットを拡張させる。
	- 具体的な手法として、以下のような手法がよく利用される。
		- [Evolve Instruct](https://azure.github.io/slm-innovator-lab/1_2_evolve-instruct/)
		- [GLAN](https://azure.github.io/slm-innovator-lab/1_3_glan/)


##### **どのようなデータセットを用意すべきか？**
データセットを準備するうえで気になる・留意する観点を列挙する。
- チューニング対象のモデル：パラメータやモデルが得意な領域によって必要なデータが異なる
- 目的のタスク：
	- 比較的データの準備が簡単：タスクが単純、アウトプットのフォーマットや振る舞いのみを学習させたいパターン
	- モデルに知識を与える：そもそもPre-trainingで大部分の知識を獲得すると言われているため、Fine-tuningで知識を与える場合はデータ量や学習方法のケアが必要である。（参考論文：[https://arxiv.org/abs/2305.11206](https://arxiv.org/abs/2305.11206)）
- データセットの量：データ件数、1件あたりの情報量をどうするか？
- データの言語：単一の言語、複数言語
- データのフォーマット：会話形式（シングルターン、マルチターン）、一貫性（文体、用語をそろえるか、バリエーションをもたせるべきか）
- データ分布：タスクに特化したデータを非常に狭く用意してモデルを複数にわけるか、幅広いデータを集めてある程度のタスクに対応できるようにするか
- バイアス・プライバシー：害を与えるような内容（暴力、虚偽）やバイアス（差別や誤解）がある偏ったデータになっていないか、混入したらNGな機密情報が含まれいないか

データセットの作成手法について解説している記事や情報は多いが、具体的にどのようなデータセットを作ったら結果的に良かったかの情報は少ないため、ここにまとめる。
###### AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp
[Borea-Phi-3.5-mini-Instruct-Jp](https://huggingface.co/AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp)は、AXCXEPT社が開発したモデル。phi-3.5-mini-Instructをベースとして、複数のチューニング手法を採用のうえ、汎用的にベースモデルから性能を向上させたモデル。特に日本語性能が向上している。
- データセットの量：日本語のWikiデータおよび、FineWebから良質なデータのみを抽出し、Instructionデータを作成。件数は明記なし。
- データの言語：FineWebを利用しているため複数言語を含んでいると推察。
- データのフォーマット：マルチターンかつ複数のタスク（[instruction synthesizer](https://huggingface.co/instruction-pretrain/instruction-synthesizer)を利用しているため）
- データ分布：マルチターンかつ複数のタスク（[instruction synthesizer](https://huggingface.co/instruction-pretrain/instruction-synthesizer)を利用しているため）
- バイアス・プライバシー：良質なデータを抽出していることからケアしていると推察。


### **2. ベースモデルを選択する**
---
一般的なユースケースでは、ベンチマークを重視してLLMを選択するケースが多い。
例えば、以下で日本語LLMをまとめてくれているが、ほとんどは数10B以上のベースモデルを用いている。（事前、事後学習ともに）
https://llm-jp.github.io/awesome-japanese-llm/

SLMはクラウドリソースを使えないユースケースや、タスク特化で大規模な知識が必要ないケースで選択されるケースが多い。
	- [Cerence Introduces Pioneering Embedded Small Language Model, Purpose-Built for Automotive | Cerence AI](https://investors.cerence.com/news-releases/news-release-details/cerence-introduces-pioneering-embedded-small-language-model)


### **3. Fine-tuning 手法・技術を選択する**

大規模言語モデル（LLM）のFine-tuningは、タスクやリソースに応じてさまざまな手法が選択されます。以下は代表的なFine-tuning技術とその特徴についての説明です。

---

#### **1. Instruction Tuning（Supervised）**
**概要:**
- 指示（Instruction）に基づいてモデルをFine-tuningする手法です。
- 主に**教師あり学習**を用いて、入力テキストと対応する出力をペアで学習します。
- ユーザーからの自然な指示や命令に対するモデルの応答性能を高める目的で利用されます。

**特徴:**
- データセットに **入力指示 → 期待される出力** の形式が必要。
- OpenAIのGPT-3.5/4やGoogleのPaLMなどの大規模モデルもInstruction Tuningを採用。


#### **2. Alignment Tuning（RLHF, DPO）**
**概要:**
- モデルの出力結果を人間の意図に合わせるためのFine-tuning手法です。
- **強化学習（Reinforcement Learning）** を活用し、ユーザーの好む出力や安全な応答を学習します。  
- **代表的な手法**:
  - **RLHF**: Reinforcement Learning from Human Feedback（人間のフィードバックに基づく強化学習）  
  - **DPO**: Direct Preference Optimization（直接的な好み最適化）

**特徴:**
- 報酬モデル（Reward Model）を利用して、好ましい出力を強化。
- 人間の評価データやフィードバックが必要。
- **RLHF**は多段階のプロセスが必要（教師ありFine-tuning → 報酬モデル → 強化学習）。


#### **3. SFT（教師ありFine-tuning）**
**概要:**
- 教師ありデータを用いてモデルをFine-tuningする基本的な手法です。
- ラベル付きデータセットを用意し、入力と出力の関係を学習させます。

**特徴:**
- 高品質な教師ありデータが必須。
- タスク固有のデータを追加することで、既存のLLMを特定タスクに特化させることが可能。
- 大規模な計算リソースが必要な場合が多い。


#### **4. PEFT（パラメータ効率の良いFine-tuning, 例: LoRA）**
**概要:**
- **モデル全体のパラメータを更新せず**に、少量の追加パラメータでFine-tuningする手法です。
- 代表例として**LoRA（Low-Rank Adaptation）** があり、効率的にFine-tuningが可能です。

**特徴:**
- **計算コスト・メモリ使用量が大幅に削減**される。
- モデルの一部（例: 重み行列の一部）を低ランク近似で更新するため、リソースが限られた環境でも適用可能。
- 事前学習済みの大規模モデルを保持したまま、追加タスクの学習ができる。


#### Fine-tuning（LoRA）の設計
- 基本的に、リソースやデータセットが最初から十分にそろっているケースはまれなので、初手はPEFT（LoRA）を選ぶことが多い。
- データやリソースが十分に揃った段階で、SFT や Alignment Tuning に移行することで、さらに精度や応答品質を高める。


### **4. トレーニングパイプライン（Fine-tuning、評価、デプロイ）をセットアップする**
---
- モデルのトレーニング環境を準備。Azureでは、機械学習開発プラットフォームである、Azure Machine Learning を利用して一連のトレーニング環境を実現可能。
- Azure Machine Learning による実装方法は[こちら](https://azure.github.io/slm-innovator-lab/2_fine-tuning/)を参照。

### **5. LLMOps（継続的運用）**
---
- Fine-tuningについても、通常の機械学習モデル開発と同様に繰り返しの試行錯誤が発生する。（データセットの作成、モデル選定、パラメータチューニングなど）
- Azure AI Foundry によって、実験、モデル品質評価、デプロイ、およびFine-tuningされた LLM 後の Prompt flow やその他のツールを使用したパフォーマンス監視までエンドツーエンドのLLMOpsを実現可能。
- Azure AI Foundry による実装方法は[こちら](https://azure.github.io/slm-innovator-lab/3_llmops-aifoundry/README.html)を参照。


# Fine-tuning（LoRA）の設計
- 基本的に、リソースやデータセットが最初から十分にそろっているケースはまれなので、初手はPEFT（LoRA）を選ぶことが多い。
- データやリソースが十分に揃った段階で、SFT や Alignment Tuning に移行することで、さらに精度や応答品質を高める。
- 今回のワークショップでもLoRAを利用したチューニング方法を採用する。

## コードの設計
以下は、ファインチューニング(Fine-tuning)の設計方針やパラメータ選択、その背景について詳細にドキュメント化した内容です。コード全体を通して、主な目的は、`AutoModelForCausalLM` でロードした因果言語モデル(Causal Language Model)を、LoRA (Low-Rank Adaptation) を用いて効率的にFine-tuningし、学習ログやハイパーパラメータをMLflowやWandBを通じて追跡すること、最終的にFine-tuning済みモデルを保存することとなっています。

## 全体像と手法選択の背景

事前学習済み言語モデル（`model_name_or_path` で指定した `microsoft/Phi-3.5-mini-instruct`）に対して、 PEFT（LoRA）を実施するためのスクリプトです。以下の特徴が見られます。

- **モデルの読み込み**  
  `AutoModelForCausalLM` と `AutoTokenizer` を使用して、事前学習モデルをロードします。  
  `tokenizer.model_max_length` を設定し、Padトークンを`unk_token`に割り当てるなど、ファインチューニング向けのトークナイザ前処理が行われます。

- **LoRAによるFine-tuning**  
  モデル自体への大規模な再学習ではなく、LoRAを用いて特定の変換行列に対する低ランク近似を加えることで、メモリ使用量を抑えつつ効率的なパラメトリックチューニングを行います。  
  LoRAは特に大規模モデルで有効で、フルファインチューニングに比べて計算・メモリコストが大幅に削減されます。

- **SFTTrainer** (from `trl`) の活用  
  `SFTTrainer` クラスを用いて学習ループを簡略化しています。これにより、LoRAの適用や学習処理が標準的な`Trainer`ループよりスムーズに行えます。

- **メトリクス・ロギング環境**  
  MLflow や WandB へのログ出力を対応し、再現性や学習過程の可視化・管理を容易にしています。  
  `mlflow`を利用することでパラメータやメトリクスが自動的に追跡され、`WANDB`を使用することで学習曲線や内部状態を可視化できます。

---

## データ処理フロー

1. **データセットの読み込み**  
   `datasets`ライブラリを用いて`train.jsonl`と`eval.jsonl`からデータをロードします。

2. **Chatテンプレート適用**  
   `apply_chat_template`関数で、入力メッセージ（`messages`）をモデルに適したプロンプト形式（システム、ユーザ、アシスタントのターンを統合したフォーマット）に整え、`"text"`フィールドに格納します。

3. **トークナイザによるトークン化**  
   `max_seq_length`に従って入力シーケンスをトークナイズします。  
   トークン数が長すぎる場合は切り詰められ、メモリ・計算コストを抑えます。

---

## ハイパーパラメータ設計の背景

以下のハイパーパラメータは、主に大規模言語モデルのSFTにおいて一般的かつ有用な設定が考慮されています。

### トレーニングパラメータ（`TrainingArguments`）

- `bf16`:  
  BFloat16精度での学習を有効化。BFloat16はFP16と同等のリソース削減を行いつつ、範囲特性により安定した学習が可能です。

- `learning_rate`:  
  学習率は微調整全体の安定性と性能を最も大きく左右するハイパーパラメータの一つです。小さすぎる学習率ではモデルが局所解に陥りやすく、学習が遅くなる。一方で、大きすぎる学習率は、事前学習済みのパラメータを破壊してしまい、既存の能力を損ねる「破壊的忘却」が発生しやすくなります。2e-5といった比較的低い値は、大規模言語モデルの微調整で一般的に用いられる範囲であり、安定した収束と既存知識の保持の両立を目指しています。特にLoRAなど軽量学習手法では、元モデルパラメータは凍結され、LoRA側パラメータが主に更新されますが、学習率が高すぎるとLoRAパラメータが過度に変化し、期待する方向へ収束しづらくなります。

- `num_train_epochs`:  
  デフォルトは1に設定。InstructionやSFTの場合、過学習を防ぎ、既存の知識を壊さないようにするため、少ないエポックでFine-tuningを行うことが多いです。

- `lr_scheduler_type`:  
  `"linear"`を用いることで、学習率を徐々に減少させる一般的なスケジューリングを採用。  
  Warmupとの組み合わせで初期学習を安定化させます。

- `warmup_ratio`:  
  ウォームアップは、学習初期段階で徐々に学習率を増やしていく仕組みであり、これも性能に顕著な影響を与えます。モデルパラメータが高次元で複雑な場合、最初から大きな学習率で更新を行うと不安定な方向へパラメータが振れてしまいます。ウォームアップを設けることで、初期にはより慎重なステップでモデルを更新し、学習が安定した後に本来の学習率に到達できます。通常は5～10%程度から始めることが多いですが、本コードでは20%とやや高めに設定することで、初期段階での不安定性をより抑え、精度向上につなげる狙いがあります。

- `train_batch_size`, `eval_batch_size`, `gradient_accumulation_steps`:  
  メモリや計算資源を考慮した妥協点として設計。データが少数のため `train_batch_size=1` を設定。`gradient_accumulation_steps=4` により、実質的なバッチサイズを拡張し、勾配の安定化を図ります。

- `max_seq_length`:  
  最大シーケンス長は、モデルが1度に処理できるトークン数を決定します。より長いコンテキストを与えることで、モデルは文脈をより深く理解でき、タスクに関連する追加情報を取り込むことで出力精度向上が期待できます。

- `logging_steps`, `save_steps`:  
  頻繁なログ出力(2ステップごと)や、100ステップごとのモデル保存など、開発・デバッグ段階でのトレーニング挙動観察・復元性を担保します。  
  実運用時には適宜見直すことができます。

### LoRAパラメータ（`peft_config`）

- `r`:  
  LoRAは低ランク近似を用いて重み更新を行う手法です。rはその低ランク近似のランクを決定するパラメータであり、直接的にモデルが表現できる空間の次元数に影響します。値が大きいほど表現力が高くなる可能性はありますが、同時に学習が不安定になったり、汎化性能が損なわれたりするリスクもあります。一般的な経験則として、r=8～64程度の範囲でタスク特性に応じて調整します。r=16は多くの実験的報告でバランスのとれた設定とされ、モデル性能への正の寄与が期待できます。

- `lora_alpha`:  
  lora_alphaはLoRAによって追加される更新パラメータをスケーリングするための要素です。このスケールが小さすぎるとLoRAパラメータはモデルに十分な影響を与えられず、大きすぎると不安定な収束を引き起こします。適度なスケール(ここでは16程度)を設定することで、LoRA層が元のモデル表現に対して適度に修正を加え、精度向上につながります。

- `lora_dropout`:  
  0.05という軽微なドロップアウトを入れて過学習を防止。LoRAには少しのドロップアウトを用いることで汎化性能を向上させる効果がある場合があります。

- `target_modules`:  
  `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` といった、LoRAを適用するモジュールの選択は、精度において非常に重要です。これらはTransformerブロック内のAttention機構やFFN(Feed-Forward Network)の重要な重み行列であり、モデルの表現力とタスク適応度を直接決定します。特にq_proj, k_proj, v_projはAttention計算の中核であり、ここの重みをLoRAで微調整することでモデルが新たな指示や文脈に適応する能力が大きく変わります。この層選択は、タスク依存的な情報処理に大きな影響を与えるため、精度改善に対する効果が大きいとされています。

---

## ロギングと可視化

- **MLflowによるログ**:  
  `log_params_from_dict`関数を通じて、トレーニング・LoRAパラメータをMLflowに記録します。これにより、後からモデルのハイパーパラメータや実行時設定を容易に再現できます。

- **WandBによるトラッキング（オプション）**:  
  `WandB`を用いることで、学習曲線、損失などの可視化を行い、チームでの共有や実験比較が容易になります。  
  `WANDB_PROJECT`, `WANDB_RUN_NAME`などを環境変数または引数で渡すことで、実行時に柔軟な管理が可能です。

---

## 学習プロセス

`SFTTrainer` により `trainer.train()` を呼び出して学習が開始されます。モデルはLoRAで注入された追加パラメータのみ更新し、元のモデルパラメータは凍結しているため、リソース消費を抑えながらタスク特化のパラメータチューニングが行われます。

学習中はログが定期的に出力され、GPUメモリ使用量やトレーニング時間などが確認できます。学習終了後、`trainer.save_state()` によって最終的なステートが保存されます。

---

## モデル保存戦略

- **`save_merged_model` が True の場合**:  
  LoRAを適用した重みを元のモデルにマージし、フルモデルとして再保存します。これにより、LoRAが不要な単一のモデルでの推論が可能になります。

- **`save_merged_model` が False の場合**:  
  LoRAアダプタ付きのモデルとして保存します。この場合、推論時にはLoRAアダプタを再度読み込む必要があるが、元モデルを分離して管理できる利点があります。

どちらの保存方法を選択するかは、デプロイの容易性やストレージコスト、モデルの再利用性などによって決定されます。

---

## まとめ

このコードベースは、以下の点に注意しながら設計されています。

- 大規模言語モデルをLoRAにより効率的にFine-tuningすることで、少ない追加パラメータで所望のタスクやスタイルに合わせられる。
- 十分にチューニングされたハイパーパラメータ（学習率、warmup、LoRAパラメータ）を用いることで、安定的で効果的な学習を行う。
- MLflowやWandBを用いたメタデータログによって、再現性やモデル管理を強化。
- 長いコンテキスト（`max_seq_length=2048`）や、アテンション関連層へのLoRA適用により、チャットボット的なInstruction対応モデルを実現しやすい。