# Multiple Personality CoD (Chain of Discussion)

[![tests](https://github.com/FreeBSE-Osaka/mp_cod/actions/workflows/tests.yml/badge.svg)](https://github.com/FreeBSE-Osaka/mp_cod/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一つのローカルLLMを、学派・方法論・リスク選好が異なる複数の専門家人格として独立に呼び出し、反論と擦り合わせを経て結論を作るCoD実験です。

ここでいう「Personality」はソフトウェア上の専門家ロールであり、精神医学的な概念を指しません。また、これはニューラルネットワーク内部のMoEではなく、複数人格の推論を調停するオーケストレーターです。人格別LoRA Weightは任意で追加できます。

内部の逐語的な思考（Chain of Thought）は扱わず、外部検証できる主張・前提・反論・証拠IDだけを記録します。

## 特徴

- 各人格は、他者の文章を見ずに初期見解を独立生成
- 通常討論は、相互反証 → 司会 → 独立監査まで実行
- イベント討論は、順番制ではなく `異議あり！` → `賛同＋補足` → 新規主張の優先順で発言
- イベント討論では、固定台帳にある主張コードと `D01` 形式の証拠IDだけを許可
- 内部の証拠文 `statement` と、表示専用の会話文 `utterance` を分離し、LoRAは後者だけに限定
- 異議・賛同は相手の原文ではなく構造化主張だけを見て会話調で応答
- renderer失敗時も、異議・賛同・維持・変更ごとの複数templateをローテーションして自然文表示
- `--no-renderer`で、すり合わせを残したまま検証済みstatementから会話文を合成
- `--body-adapter`で、LoRAはclaim本文だけを生成し、発話行為はコードで安全に合成
- 証拠のない主張は自動失格し、対立は最大ラウンド内で3/4に達した時点で終了
- 人格ごとに効用と損失を分け、反論は前提否定・反例・トレードオフの型を使用
- prompt/configだけを比較するbounded RSI shadow gateを同梱
- 全呼び出しをJSON保存し、人間が承認した発言だけをLoRA用JSONLへ変換

## 必要環境

- Python 3.11
- 通常討論: 起動済みの [Ollama](https://ollama.com/) とローカルモデル
- MLX討論・Weight評価: Apple Silicon と [MLX-LM](https://github.com/ml-explore/mlx-lm)

通常討論・データ生成・テストに追加Pythonパッケージはありません。

## クイックスタート（Ollama）

```sh
git clone https://github.com/FreeBSE-Osaka/mp_cod.git
cd mp_cod

ollama pull qwen3.5:4b
python3.11 cod_model.py list --domain software

python3.11 cod_model.py debate --domain software \
  "新しいローカル検索アプリをSwiftで作るべきか、RustとWeb UIを組み合わせるべきか"
```

既定モデルは `qwen3.5:4b`、既定APIは `http://127.0.0.1:11434/api/chat` です。別モデルや別URLは `--model`、`--api-url` で指定できます。

実行中は各人格の公開発言を実況し、完全な構造化ログを `runs/` に保存します。途中失敗時は `.partial.json` が残ります。

## 証拠ID付きイベント討論（MLX / Ollama GGUF）

サンプル台帳には、データ本文、許可する主張コード、各主張を裏付ける `D` 番号、対立関係、人格ごとの関心領域が入っています。

```sh
python3.11 -m venv .venv
.venv/bin/pip install "mlx-lm[train]"

.venv/bin/python cod_model.py event-debate \
  --ledger data/typhoon18_20260825/claim_ledger.json \
  --domain weather \
  --backend mlx \
  --model-path mlx-community/Qwen3-1.7B-4bit \
  --max-turns 12 \
  --reconcile-rounds 2 \
  --max-tokens 600 \
  --prompt-profile orthogonal_fewshot
```

GGUFを変換せずOllamaで使う場合:

```sh
python3.11 cod_model.py event-debate \
  --ledger data/general_conversation_v2_holdout/claim_ledger.json \
  --domain general \
  --backend ollama \
  --ollama-model qwen3.5:4b \
  --max-turns 12 \
  --reconcile-rounds 2
```

Ollama backendも同じD番号検証、raw保存、ラウンド、hard gateを使います。MLX LoRAの`--adapter-map`はMLX backend専用です。

live shadow向けの軽量経路:

```sh
<mlx-python> cod_model.py event-debate \
  --ledger <claim-ledger.json> \
  --domain general \
  --model-path <base-model> \
  --fast
```

`--fast`は各人格2主張、最大2発言/人格、すり合わせ0回です。Adapterがない場合は会話rendererも呼ばず、検証済み`statement`またはmove別templateから表示文を作ります。独立判断とD番号validatorは維持しますが、自然文Weightと合意形成を省くためhard gateは通らず、live shadow専用です。Qwen3-1.7Bの未学習EV台帳では8 eventが4 call・29.5秒、4 eventのv4再実測が4 call・23.2秒でした。

すり合わせを残してrendererだけ省く場合:

```sh
<mlx-python> cod_model.py event-debate \
  --ledger <claim-ledger.json> \
  --domain general \
  --model-path <base-model> \
  --max-turns 12 \
  --reconcile-rounds 2 \
  --no-renderer
```

`--no-renderer`は主張数やラウンドを縮小せず、検証済みstatementへmove別導入句を合成します。Adapter指定とは併用できません。EVの2 event・1 round実測では11 call・68.8秒から8 call・39.2秒へ短縮し、4人格投票と3/4合意を維持しました。合成文は`composed_statement_fallback`として記録され、Weight成功やhard gate通過には数えません。

モデルには他人格の文章を渡さず、検証済みの主張コードと証拠IDだけを共有します。採決に使うのは検証済みコードだけです。

- `statement`: D番号付きの内部証拠文
- `utterance`: D番号や内部codeを読まない、UI・実況用の会話文

異議なら相手の見落としを指摘した後に、代案・修正版・採用条件のいずれかを必須とします。賛同は追加観点を伴い、見解変更は複数の自然な言い回しを人格・ラウンドごとに使い分けます。相手の原文は見せず、対象claimのlabelだけを渡すため、文章コピーによる擬似合意を避けます。

不正なD番号は拒否し、選択済みDだけで `statement` を一度修復します。本文は使えてD表記だけが欠けた場合は `model_sanitized` とします。すり合わせ修復文が会話本文へD根拠句を出した場合は根拠句だけを除去し、agree / maintain / revise等のmove表現が欠ける場合は検証済み定型句を補います。モデル本文と修復rawは残し、会話化できない場合だけ証拠文由来の表示へ戻します。初回raw、反応raw、修復rawはすべて保存します。

会話例:

```text
批判的設計者: そのまま進めるより、対象を絞ったpilotを代案として先に試すべきです。
仮説構築者: その案には懸念があります。代わりに、観測された改善を広く検証したいです。
実証監査者: その案には賛成です。私としては、評価期間を固定する点も大事だと思います。
実行設計者: 現場で回すなら、担当者と停止条件まで決めて小さく始めるのが現実的です。
仮説構築者: 考え直しました。今回はpilot案を支持します。
```

`--prompt-profile` は `baseline`、`orthogonal`、`orthogonal_bare`、`orthogonal_fewshot` から選べます。既定値は目的関数と1件の形式例を使う `orthogonal_fewshot` です。`orthogonal_bare`はWeight評価用で、人格別speech例とfew-shotを外します。

Generalは既存台帳では従来の3人格を保ちます。`role_preferences`に`pragmatic_operator`を含むv2台帳だけ、4人目の「実行設計者」が参加します。4人時の合意閾値は3票です。

### utterance renderer LoRA

構造判断は常にBaseで行い、claim code、証拠、statement、投票が確定した後だけLoRAをロードします。現在のAdapter学習分布に合わせ、Adapter使用時は1発言ずつ描画します。AdapterなしのBase rendererだけ最大3発言をまとめます。

```json
{
  "schema_version": 1,
  "adapters": {
    "dynamical_modeler": "/path/to/dynamical-adapter",
    "ensemble_probabilist": "/path/to/ensemble-adapter"
  }
}
```

```sh
<mlx-python> cod_model.py event-debate \
  --ledger <claim-ledger.json> \
  --domain weather \
  --model-path <base-model> \
  --adapter-map <adapter-map.json>
```

各Adapterは `adapter_config.json` と `adapters.safetensors` が必須で、実行ログに両方のSHA-256を保存します。KVキャッシュは人格間で共有しません。

共有rendererは次の1 directoryだけを指定できます。`--adapter-map`とは排他的です。

```sh
<mlx-python> cod_model.py event-debate \
  --ledger <claim-ledger.json> \
  --domain general \
  --model-path <base-model> \
  --renderer-adapter <shared-renderer-adapter>
```

全文を作るv3〜v6のrenderer Weightは引き続き研究用です。現行のClaim Body v3 step128だけは、検証済みclaim本文に限定し、validatorとfallbackを必須にした条件付きWeightとして利用できます。

```sh
<mlx-python> cod_model.py event-debate \
  --ledger <claim-ledger.json> \
  --domain general \
  --model-path <base-model> \
  --body-adapter <claim-body-v3-step128-adapter>
```

`--body-adapter`は1発言ずつ本文だけをtemperature 0で生成します。入力は固定ID`B01`、話者名、凍結claimだけで、evidenceをrendererへ渡しません。証拠と投票はAdapterを外したBaseが担当し、D番号付きログへ保持します。object / agree / maintain / reviseはコード合成します。本文はstrict schema、丁寧完全文、時制、制限、数字、競合claimを検査し、不合格ならstatementへ戻します。`--adapter-map`、`--renderer-adapter`、`--no-renderer`とは排他的です。

台風データは2026年8月25日15時50分JST時点の再現用スナップショットで、現在の予報には使えません。

## 学習データの承認と出力

成功しただけのログは学習へ入りません。内容を確認し、採否と使用する発言を明示します。

```sh
python3.11 cod_model.py mark runs/<run>.json approved --calls 2,5,8 \
  --note "技術名、根拠、反論後の修正を確認済み"

python3.11 cod_model.py mark runs/<run>.json rejected \
  --note "未確認の固有名と誤った計算が混入"

python3.11 cod_model.py export --runs runs --out data/sft
```

`approved` 以外はexportされず、承認済みrunでも `--calls` にない発言は除外されます。出力先は `data/sft/<domain>/<persona>/` です。

### Event会話データ

event-debateの会話は、完走・hard gate再計算・人間レビューを通ったrunだけ承認します。

```sh
python3.11 cod_model.py mark-event runs/<event-run>.json approved \
  --reviewer FreeBSE \
  --note "根拠、異議、賛同、見解変更を確認"

python3.11 cod_model.py export-dialogue \
  runs/<approved-weather>.json \
  runs/<approved-software>.json \
  --out data/dialogue_sft \
  --min-per-persona 30 \
  --batch-size 3 \
  --shared-renderer
```

承認時はreview欄を除いたrun全体のSHA-256を固定し、承認後に変更されたrunはexportを拒否します。export対象はモデル由来utteranceだけで、fallback、機械的定型文、近似同文、不正文を除外します。出力は人格別のMLX chat JSONLとmanifestです。`data/dialogue_sft/` はGit管理外です。

2026-08-28のDialogue v1凍結snapshotは3domain合計154件で、Generalは実証監査31件、批判的設計者34件、仮説構築者30件でした。3人格とも時系列valid/testを持ち、最低30件へ到達しています。このsnapshotで人格別LoRAを学習しましたが、未学習transfer台帳で自然会話がBaseから改善しなかったため非昇格です。

v2では`「現時点では」`、`「可能性を重く見ています」`、`「確かに、見落としていました」`等を機械文として除外し、人格別speech例、対案必須の異議、複数のagree / maintain / revise、任意4人目を導入しました。承認済み8runから実証監査33、批判的設計33、仮説構築33、実行設計32件を凍結し、4人格の個別LoRAを新規学習しました。全人格で凍結test lossは改善しましたが、未学習payloadの`utterance`生成はBaseと同値で不合格だったため、v2 Weightも非昇格です。

v3では学習JSONLとruntimeを同じ`items -> utterances`契約にし、LoRAを表示用rendererへ完全分離しました。人格別pilotと共有rendererの実Weightはschemaを学習しましたが、未学習payloadでmove混同・主張省略・反復崩壊が残ったため全て非昇格です。一方、Adapterなしの`--fast`は構造判断4 call・29.5秒まで短縮しました。

v4では6 train topicとemail/EV holdoutを分離し、scheduler上で到達可能なpersona×moveを最低6件へ均等化しました。共有step160はEVでBase 0/18から14/18へ改善しましたが、emailは5/16、実走Weight由来1/4に留まったため非昇格です。無効な会話文はmove別templateへ安全に戻し、Adapterなしfastは4 call・23.2秒で自然な異議文を表示しました。

## 軽量Weight実験

Pythonが正解を厳密生成する有限集合カリキュラムを作れます。

```sh
python3.11 cod_model.py curriculum --count 240 --out data/auditor_curriculum
```

実験済みの `Qwen3-1.7B-4bit + empirical_auditor LoRA step 8` は、決定論的ツール証拠がある場合だけ条件付き昇格しています。証拠なしの汎用推論Weightではありません。

```sh
<mlx-python> cod_model.py weight-audit \
  --model <mlx-model-or-hf-repo> \
  --adapter <adapter-directory> \
  --upper 100 --condition 2 --target 3 --hypothesis 1/3
```

Adapter WeightはこのGitリポジトリに含めていません。再現条件と評価値は [実験記録](docs/weight_experiment_20260825.md)、ハッシュと運用境界は [昇格記録](promotions/qwen3-1.7b-auditor-r1-step8.json) にあります。

General Dialogue v1の人格別LoRAは全て凍結test lossを改善しましたが、自然会話transferが同値だったため非昇格です。設定、loss、SHA、停止理由は [General Dialogue Weight v1実験記録](docs/general_dialogue_weight_v1_20260828.md) にあります。

General Dialogue v2も4人格のLoRA Weight本体は作成済みです。凍結loss、未学習transfer 0/4、実行設計者step16追加学習の停止理由、全SHAは [General Dialogue Weight v2実験記録](docs/general_dialogue_weight_v2_20260828.md) にあります。現行運用はBase + v2 prompt / sanitizer / hard gateです。

General Dialogue v3はLoRAをutterance rendererだけへ分離し、人格別・共有・早期停止を比較しました。Weight本体、29.5秒のfast実測、全SHA、非昇格理由は [General Dialogue utterance renderer v3実験記録](docs/general_dialogue_weight_v3_20260829.md) にあります。

General Dialogue v4はvalidatorの誤失格修正、到達可能moveの均等化、共有・move均衡・人格別Weightを比較しました。全評価、23.2秒のfast実測、SHA、停止理由は [General Dialogue utterance renderer v4実験記録](docs/general_dialogue_weight_v4_20260903.md) にあります。

Horse renderer v1は競馬固有の限定表現を対象にv4 step160から48 iteration継続学習しました。horse holdoutとemailは改善しましたが、horse実走8 eventで直接採用0件のため非昇格です。意味反転を止めたvalidator修復、全評価、SHAは [Horse renderer Weight v1実験記録](docs/horse_renderer_weight_v1_20260903.md) にあります。

既存のMLX Qwen3.5-4B-4bitもbatch 1 / last 4 layers / rank 4で確認しました。学習はOOMしませんでしたが約0.081 iteration/秒で、難所3 moveはBaseとstep20が同じ1/3だったため停止しました。設定、memory、Weight、全SHAは [Qwen3.5-4B renderer smoke](docs/qwen35_4b_renderer_smoke_20260904.md) にあります。

Natural specialist v5ではQwen3-14B teacherから実行役event-agreeの直接合格自然文を3件得ましたが、親step160からの専用継続はholdout 1/3のまま、Base specialistは0/3でした。move別few-shotもobjectへ賛同例が混入したためruntimeへ採用せず、全結果を [General Dialogue natural specialist v5実験記録](docs/general_dialogue_weight_v5_20260904.md) に残しています。

Natural specialists v6では12の異なるtopicから仮説object 12件・実行event-agree 13件を集め、人格・phase・move別LoRAとrepair LoRAを評価しました。数値創作を拒否するgrounding guardは採用しましたが、全Weightが未学習holdoutで親同等以下だったため非昇格です。全target、評価、SHA、停止理由は [General Dialogue natural specialists v6実験記録](docs/general_dialogue_weight_v6_20260904.md) にあります。

発話行為をコードで確定し、検証済みstatement本文だけを接続する`--no-renderer`経路は、4人格のすり合わせを残したまま39.2秒へ短縮しました。設計、実測、会話全文は [Composed statement renderer v1](docs/composed_statement_renderer_v1_20260904.md) にあります。

Claim Body v1は後の監査で、提案を完了事実へ変える文と名詞断片を通していたためsupersededにしました。[v1記録](docs/claim_body_weight_v1_20260904.md)と[旧昇格記録](promotions/qwen3-1.7b-claim-body-v1-step64.json)は履歴として残しています。

現行のClaim Body v3 step128は、17 train topic・585件のclean targetでBaseから学習しました。完全除外したemail / EV / bike 15ケースでcontract valid `15/15`、strict schema `15/15`、競合claim `0`です。3 topicの1 round実走はいずれも公開6発言全てがWeight由来、fallback 0、hard gate通過でした。設定、v2/v4停止理由、会話全文、SHAは [Claim Body Weight v3](docs/claim_body_weight_v3_20260904.md)、運用境界は [現行昇格記録](promotions/qwen3-1.7b-claim-body-v3-step128.json) にあります。

Hugging Face向けには、ローカルpathを除いたAdapter設定、Model Card、Weight、SHA256SUMSだけのstaging packageを用意しています。公開前検証とupload境界は [Hugging Face release staging](docs/huggingface_release_claim_body_v3.md) を参照してください。

## bounded RSI shadow

RSIは、異なる固定ledgerを使ったdevelopmentとholdoutの両方で候補runがParentを上回るか検査します。

別domainの凍結holdoutとして、架空の8週間iOS開発要件を使う [`data/software_architecture_holdout/claim_ledger.json`](data/software_architecture_holdout/claim_ledger.json) を同梱しています。実在案件の事実ではなく、Swift/MLX先行とRust共有コア先行を異なる目的関数で議論させる再現用fixtureです。

第三domain候補として、架空の温室実験でヒーター効果・センサー誤差・換気交絡を検討する [`data/general_experiment_holdout/claim_ledger.json`](data/general_experiment_holdout/claim_ledger.json) も凍結しています。

```sh
python3.11 cod_model.py rsi-shadow \
  --parent-dev runs/parent-dev.json \
  --candidate-dev runs/candidate-dev.json \
  --parent-holdout runs/parent-holdout.json \
  --candidate-holdout runs/candidate-holdout.json \
  --round 1 --max-rounds 3 \
  --out runs/rsi-round1.json
```

評価対象はモデル主張率、証拠文・会話文成功率、reaction失敗率、fallback率、生出力保存率、異なるclaim間の会話同文率、発言イベントの多様性です。Parentがhard gate不合格でも候補は評価できますが、候補自身はdevelopmentと別ledger holdoutの両方ですべてのhard gateを通る必要があります。holdoutへ改善が移らない場合は停止します。

2026-08-26のローカルcross-domain smokeではweather development `+25.00`、software holdout `+13.75` で `research_shadow_candidate` になりました。これは昇格ではなく、引き続き `promotion_allowed=false`、`parent_replacement_allowed=false` です。人間確認なしにWeight、コード、GitHub、Hugging Faceを変更しません。

続くRound 2では機械的な初期発言を減らす候補がweather `+1.25`、software `+0.83` でした。softwareが必須の`+1.00`へ届かなかったため `parent_retained`、`continue_allowed=false` で停止し、候補profileは公開コードへ残していません。第三domainの追加実走も行っていません。

## テスト

```sh
python3.11 -m py_compile cod_model.py test_cod_model.py
python3.11 -m unittest -v
```

## ライセンス

コードは [MIT License](LICENSE) です。外部モデル、学習済みWeight、データソースには、それぞれのライセンス・利用条件が適用されます。
