# Multiple Personality CoD (Chain of Discussion)

[![tests](https://github.com/FreeBSE-Osaka/mp_cod/actions/workflows/tests.yml/badge.svg)](https://github.com/FreeBSE-Osaka/mp_cod/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一つのローカルLLMを、学派・方法論・リスク選好が異なる複数の専門家人格として独立に呼び出し、反論と擦り合わせを経て結論を作るCoD実験です。

ここでいう「Personality」はソフトウェア上の専門家ロールであり、精神医学的な概念を指しません。また、これはニューラルネットワーク内部のMoEではなく、複数人格の推論を調停するオーケストレーターです。人格別LoRA Weightは任意で追加できます。

内部の逐語的な思考（Chain of Thought）は扱わず、外部検証できる主張・前提・反論・証拠IDだけを記録します。

## 特徴

- 各人格は、他者の文章を見ずに初期見解を独立生成
- 通常討論は、相互反証 → 司会 → 独立監査まで実行
- イベント討論は、順番制ではなく異議・賛同などの反応を新規主張より優先して発言
- Generalは8つの専門観点を用意。賛否を役に固定せず、同じ意見の複数人支持・指摘だけの反論・賛同と改善を許可
- イベント討論では、固定台帳にある主張コードと `D01` 形式の証拠IDだけを許可
- 内部の証拠文 `statement` と、表示専用の会話文 `utterance` を分離し、LoRAは後者だけに限定
- 異議・賛同は相手の原文ではなく構造化主張だけを見て会話調で応答
- renderer失敗時も、異議・賛同・維持・変更ごとの複数templateをローテーションして自然文表示
- `--no-renderer`で、すり合わせを残したまま検証済みstatementから会話文を合成
- `--body-adapter`で、LoRAはclaim本文だけを生成し、発話行為はコードで安全に合成
- 証拠のない主張は自動失格し、対立は最大ラウンド内で3/4に達した時点で終了
- 人格ごとに効用と損失を分け、反論は前提否定・反例・トレードオフの型を使用
- prompt/configだけを比較するbounded RSI shadow gateを同梱
- 複数人格の提案・批評、実験の実測、反省と修正を自律反復する`pdca`を同梱
- 全呼び出しをJSON保存し、人間が承認した発言だけをLoRA用JSONLへ変換

## 必要環境

- Python 3.11
- 通常討論: 起動済みの [Ollama](https://ollama.com/) とローカルモデル
- MLX討論・Weight評価: Apple Silicon と [MLX-LM](https://github.com/ml-explore/mlx-lm)

通常討論・データ生成・テストに追加Pythonパッケージはありません。

## dotsなどの秘書役へ渡す補佐官レポート

`tools/advisor_brief.py`で、完了済みのportable討論ログを元台帳と照合し、候補・異論・根拠・未解決対立・発言の由来をJSONへ整理できます。推論を追加起動せず、保存済み合格フラグも再計算します。検査失敗や未解決対立は`HOLD`、それ以外も意味の確認が必要な`REVIEW_REQUIRED`とし、実行権限は付与しません。dots実接続は未確認です。[使い方と会議調整の腕試し](docs/advisor_brief_20261006.md)

会話と理由の再生成は、本人の凍結主張・選択済み資料だけを使い、1発言ずつ分離しています。2026-10-07の8人Base試験は673.957秒・87呼び出しで完了し、会話36/36、公開説明35/36がモデル由来でした。ただし説明の補完1件、未解決対立、資料にない参加者不在の断定、「30分化」と「30分短縮」の混同等が残り、HOLDです。各1回の実測を安定した高速化や汎用の意味保持とは数えません。[独立描画と原文点検の全記録](docs/advisor_meeting_evaluation_20261006.md)

General v5は320例・18,197正解tokensの追加LoRA学習と固定4候補の開発48入力生成を完了しました。事前規則で選んだstep160は固定事実保持20/48で親25/48を下回り、Base合格5件・親合格8件の後退、対案0/6のため研究HOLDです。未学習72入力のモデル出力は生成せず、既定Weightの置換やWeight公開もしていません。各Adapterは754,632 bytesでBaseは別途必要です。[v5の教材と全結果](docs/qwen35_full_utterance_own_only_v5_20261006.md)、[非昇格記録](promotions/qwen3.5-4b-full-utterance-v5-step160.json)を参照してください。

数字の検査は台帳由来の主張と資料本文だけを出典にし、生成理由や内部IDの数字を除外しています。明示的な数式を既存計算器で再計算し、公開説明と会話の合計不一致があれば補佐レポートをHOLDにします。公開文に漏れる採決のLEFT・RIGHT等も拒否し、レポートの表示文を再検査します。全162テストが通過しましたが、不一致0件は意味の合格ではありません。旧8人ログの公開説明5件は新検査で拒否されるため、元点数を保存したまま再変換を拒否しています。[検証範囲と描画入力の比較](docs/semantic_fact_check_20261007.md)

事実保持の指示追加と描画入力の削減を、固定要求・Baseのみで実生成比較しました。既存の本人限定再描画と同じ入力は9件中7件の直接描画検査に通りましたが、参加時刻や提案の再評価の問題が残り、初回描画へは採用していません。研究候補の形式合格を、通常経路の改善やWeight昇格とは扱いません。別の事実照合AIも新しい合成18例で誤承認1件があり、通常経路へ採用していません。

既存のQwen・Gemma・Nemotronも同じOllama条件で各9要求を比較しましたが、意味の課題が残り、既定モデルは変更していません。「検討するのはどうでしょうか」「いかがでしょうか」等の自然な条件付き対案を認識する共通検査の修正は採用しました。別題材の新しい6要求も実生成し、保存原文を新検査で6/6合格としましたが、検査変更をモデルの学習向上とは数えません。否定・過去の支持・相手案への反転は拒否を維持し、General v5の保存済み判定も維持しました。

公開文検査修正後の8人実走は733.422秒・91呼び出し、説明34/36・会話36/36でHOLDでした。局所的な3/4票で停止しても統合上の対立が残る問題を確認し、停止条件を統合結果へ合わせました。164テストが通過し、8人の実モデルも2ラウンド・1471.978秒・154呼び出しで完了しました。会話60/60・説明57/60でHOLD、選択変更4件中2件の理由は補完です。モデル由来の理由にも意味の問題が残り、停止条件の実走成功と内容の品質は区別します。[停止条件と全体試験](docs/advisor_meeting_evaluation_20261006.md)

変更理由の内部選択コード漏れを、初回・修復・補佐レポートで共通して拒否します。モデル理由が得られない補完は、その不足を明示し、新観測が増えたようには表示しません。167テストが通過しました。旧主張を渡す2項目指示や理由専用指示は、4Bで意味の後退が残るため未採用です。既存の27Bは固定8要求で状態・範囲を保持する教師候補となりましたが、追加学習、教材採用、既定モデルの置換、Weight昇格は行っていません。[変更理由と教師候補の全記録](docs/advisor_change_reason_20261007.md)

Generalの8演者×8発話行為について、学習用10テーマだけから教師候補64件を生成・点検しました。意味レビュー、現行検査、768-token制約を合わせて満たす候補は26件で、残る38組の教材は未完成です。Raw・旧点数・固定評価入力を保持し、追加学習はまだ開始していません。[教師教材の点検と不足する組](docs/qwen35_general_teacher_v6_20261007.md)

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

従来のstructuredモードでは、異議に代案・修正版・採用条件を必須とし、賛同は追加観点を伴います。Generalの既定はflexibleモードで、指摘だけの反論や同意だけも許可し、同じ意見を複数人が支持しても減点しません。相手の原文は見せず、対象claimのlabelを渡します。同意人数を独立した証拠の数とは扱いません。

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

Generalは仮説構築・批判的設計・実証監査・実行設計・利用者体験・資源制約・長期影響・横断的設計の8観点です。イベント討論の参加者は台帳の`role_preferences`で指定し、既存の3人・4人台帳も維持します。有効なモデル主張を持つ役が最低2人必要で、欠けた意見をコードで作って人数や主張数を埋めません。合意閾値は参加者数の3/4切り上げ（4人なら3票、8人なら6票）です。

`--discussion-style structured`で従来の発話方針を選べます。8人の架空の傘相談、`challenges` / `extends`関係、他プロジェクトとの互換性は [Generalの柔軟な討論](docs/general_flexible_discussion_20260908.md) にあります。台帳にない主張も生成する通常討論は `python3.11 cod_model.py debate --domain general "相談テーマ"` ですが、固定台帳の証拠検証と同等ではありません。

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

`--body-adapter`は1発言ずつ本文だけをtemperature 0で生成します。入力は固定ID`B01`、話者名、凍結claimだけで、evidenceをrendererへ渡しません。証拠と投票はAdapterを外したBaseが担当し、D番号付きログへ保持します。object / agree / maintain / reviseはコード合成します。本文はstrict schema、丁寧完全文、時制、制限、数字、競合claimを検査し、不合格ならstatementへ戻します。valid本文はclaim単位でrun内cacheし、同じclaimを選んだラウンド発言へ再利用します。`--adapter-map`、`--renderer-adapter`、`--no-renderer`とは排他的です。

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

2026-09-08には別方式のQwen3.5-4B本文専用LoRAをBaseから新規学習しました。step32（約4.07MB）は既存15問がBase 1/15→15/15、新規12問が6/12→7/12（語尾補正込み9/12）。ただし新規テーマの意味変化が残り、研究用HOLDです。96ステップ予定の学習は88の報告後にMetal内部エラーで終了し、保存済み32/64を比較しました。[学習・検証・全制約](docs/qwen35_claim_body_v1_20260908.md)に記録しています。既定Weightの置換、iPhone配備、Weight公開はしていません。

2026-10-02のv2では、32件の追加教材と復習を混ぜ、継続方式とMLPのみの方式を64ステップずつ学習しました。MLP step48（約0.755MB）は新規16問11/16・語尾補正込み13/16、既存15問15/15を確認。8人の傘相談は22発言全てWeight本文でpassしましたが、別の対案討論で現在進行形への変化が残りHOLDです。[実測・全文確認・失敗した実走](docs/qwen35_claim_body_v2_20261002.md)を参照してください。

続くv3は時制・確実性の32件で追加学習し、新しい16問が9/16→14/16、既存15問15/15を維持しました。ただし未確認の行為を「検証済み」とする1件が残りHOLDです。`--body-cache-scope speaker`で話者別生成を試せますが、今回の8人実走では呼び出し増に対して文体の差が小さく、時制の不安定さも残りました。[比較・制約・再現](docs/qwen35_claim_body_v3_20261002.md)を参照してください。

v4は行為と効果確認を区別する教材で追加学習し、step16（約0.755MB）を最終検証前に選定しました。「試している→検証済み」は修正できましたが、新規16問は旧構成と同じ14/16、確認済み事実8問は8/8→5/8、共通の改訂guardで以前の12問は11/12→9/12となりHOLDです。`--body-system-file`で実験promptを指定でき、不正活用「出す、です」も共通検査で拒否します。[プロンプト対Weight比較・元raw・制約](docs/qwen35_claim_body_v4_20261002.md)に記録しています。既定Weightの置換やWeight公開はしていません。

2026-10-04〜05のQwen3.5本文v5は、実MLXラッパーで思考なしの学習境界を揃え、v3から128ステップを正常完走しました。step32（約0.755MB）は開発8問が6/8→7/8、以前の時制16問が14/16→16/16、確認済み8/8と既存15/15を維持。一方、新規16問は15/16同点、初期12問は11/12→10/12、8人実走のWeight本文は19/23→17/23で **研究HOLD・既定置換なし** です。欠落した数量上限や比較境界を共通guardで拒否し、元rawを変更せず再採点しました。容量20GiBの監視付き学習CLI、除外試験、全非回帰と実発言は [v5学習と検証](docs/qwen35_claim_body_v5_20261004.md)、SHAと未達理由は [非昇格記録](promotions/qwen3.5-4b-claim-body-v5-step32.json) を参照してください。

2026-10-05のQwen3.5本文v6は、原文由来のkind・数量比較を追加する入力で128ステップ学習し、step64を最終検証前に選定しました。新規16問は従来の親14/16・新入力だけの親11/16に対し16/16、8人実走のWeight本文は19/23→23/23・代替0になりました。ただし旧行為効果は14/16→9/16、確認済み8/8→7/8等の後退があり **研究HOLD・既定置換なし** です。実験指定は`--body-constraint-hints`（討論）、`--constraint-hints`（dataset/evaluate）で、既定OFF。補助情報へtargetやアンカーを入れません。[v6入力と全検証](docs/qwen35_claim_body_v6_20261005.md)、[SHAと未達理由](promotions/qwen3.5-4b-claim-body-v6-step64.json)を参照してください。

同日のQwen3.5本文v7は、同じ正解を通常・補助入力の両方で復習し、256ステップを正常完走しました。step256（約0.755MB）を最終生成前に固定し、新規16問は通常15/16同点・補助13/16→15/16。旧115例は通常104/115→110/115で親の合格を全て保持、補助107/115で集計上改善しましたが、除外条件反転など親の合格4例が後退し **研究HOLD・既定置換なし** です。8人実走はWeight本文19/23→23/23・代替0、Baseの独立判断・投票rawは一致したものの、根拠誤読で討論全体は不合格。明示的な対象の包含・除外を共通guardで検査し、凍結rawを別ファイルへ再採点しました。[v7の学習と全検証](docs/qwen35_claim_body_v7_20261005.md)、[SHAと未達理由](promotions/qwen3.5-4b-claim-body-v7-step256.json)を参照してください。

Qwen3.5本文v8は、包含・除外、時制、確認状態、修飾節の対照教材で親v7から192ステップを学習し、step144を最終生成前に固定しました。新規16問は通常12/16→16/16、補助13/16→15/16。旧131例は通常125/131同点・補助122/131→123/131ですが、包含反転と個別後退が残り **研究HOLD・既定置換なし** です。親・候補の8人実走は本文23/23・代替0を維持、判断・投票rawは一致しましたが、Baseの根拠誤読で討論全体は不合格。全799出力を再採点し、固定基準や元rawを変更していません。[v8学習と全検証](docs/qwen35_claim_body_v8_20261005.md)、[SHAと未達理由](promotions/qwen3.5-4b-claim-body-v8-step144.json)を参照してください。

v8後の入力診断では、同じWeightの72生成でsystem文と補助情報を分離しました。既知の除外反転は追加systemだけでも発生し、元system＋補助情報では保持。空欄省略は全12例で同一raw、キー順変更も他例の後退があり既定へ採用しません。文末だけでなく文中の試す／試みるの時制を共通guardで検査し、旧799出力を別ファイルへ再採点しました。直接合格数を変更せず、Weight改善・全体昇格には数えていません。[入力依存・文中時制の診断](docs/qwen35_input_diagnostic_20261005.md)に全条件と限界を記録しています。

Qwen3.5本文v9は元systemを両入力で固定し、MLPのみ／attention追加の2条件を各48ステップ実学習しました。開発の対応32出力は一致し、事前規則で小さいMLP48（約0.755MB）を固定。新規16問は親・候補とも両入力13/16、旧147例は通常141/147→139/147・補助139/147→140/147で個別後退が残り **研究HOLD・既定置換なし** です。8人の親／候補・通常／補助4実走は本文23/23・代替0を維持しましたが、Baseの根拠誤読で全体は不合格。過去試行や計測項目の意味変化を集計と区別し、本文とコード側の接続句も分けて記録しています。[v9学習対象層と全検証](docs/qwen35_layer_comparison_v9_20261005.md)、[SHAと未達理由](promotions/qwen3.5-4b-claim-body-v9-mlp-step48.json)を参照してください。

Generalの実験指定`--prompt-profile source_grounded`では、否定・未確認を保つ判断指示と、本人の候補理由・専門観点を全文rendererへ渡す入力を使います。既定OFF、本文Adapterの入力は変更しません。新規の確認済み／不確定対照、8人格の全文比較で対象・比較方向の誤読が残り **研究HOLD** です。JSONの具体形で最上位objectは9/9、行キー・IDを含む完全契約は8/9になりましたが、直接の全文生成0/24・補正後7/24で、Weight昇格には数えていません。[根拠解釈と理由付き発言の検証](docs/qwen35_source_reason_context_20261006.md)に原文保存先と未達を記録しています。

2026-10-06のQwen3.5全文発言LoRAは、8人格・8種の発話を含む9学習topicで128ステップ完走しました。step32（約0.75MB、Baseは別途必要）は未学習24例が12/24→13/24でしたが、旧全文18例は10/18→0/18へ後退。8人の同条件実走も直接発言0/24、補正12・代替12のままで **研究HOLD・既定置換なし・Weight公開なし** です。通常の「根拠」という語で説明の後半が消える表示バグだけは共通処理で修正し、学習の改善とは区別しています。[全文Weightの学習と全検証](docs/qwen35_full_utterance_weight_v1_20261006.md)、[非昇格記録](promotions/qwen3.5-4b-full-utterance-v1-step32.json)を参照してください。

全文発言v2は3入力形式・前案ラベル・確認済み事実と未確認条件を含む教材で64ステップを完走しました。固定選定のstep16は開発の事実保持6/48で親7/48を上回らず **研究HOLD** です。未学習72入力のモデル出力はまだ生成していません。一方、具体的なJSON配列を共通指示へ追加する比較では、既知plain32入力の完全契約がBase・候補とも9/32→32/32になりました。これは指示の改善でありWeightの性能向上とは区別しています。既に具体例のある理由付き形式は元の指示を維持します。保存済み討論を追加推論なしで点検する`tools/discussion_quality.py`も追加し、同じ立場の異なる演者の反復とラベル復唱を診断できます。正当な同意は自動失格にしません。[v2学習と共通指示の検証](docs/qwen35_full_utterance_weight_v2_20261006.md)、[非昇格記録](promotions/qwen3.5-4b-full-utterance-v2-step16.json)を参照してください。

全文発言v3は、同じ教材・学習率・32ステップの新規LoRAと継続を完走し、4候補を同じ48入力で比較しました。固定選定の継続32は事実保持がBase16/48→20/48でしたが、Base合格の6入力が落ち、対案・異議も固定直接検査0/6のため **研究HOLD** です。未学習72入力のモデル出力は生成していません。学習配分25/64組の偏りと、末尾padding1 tokenをlossへ含める境界を診断し、共有学習器はEOSを残してpaddingだけを除外するよう修正しました。117テスト、小さなCPUモデル、実Qwenの修正後1ステップを確認しました。Qwenでは正解49 tokensだけを学習し保存まで成功しましたが、発言品質の改善とは区別します。学習器は`--parent-adapter`省略でBaseから新規開始でき、容量監視と例外時の復元も維持します。[初期化比較の全結果](docs/qwen35_full_utterance_initialization_v3_20261006.md)、[非昇格記録](promotions/qwen3.5-4b-full-utterance-v3-resumed-step32.json)を参照してください。

全文発言v4は、8演者×8発話行為×4表現の256ステップを、実入力を切らず未使用tailだけを除く経路で完走しました。固定4候補の開発48入力を比較したstep192はstudy合格がBase16/48→22/48でしたが、Base合格2入力の限定条件が欠落し、対案0/6・異議2/6で **研究HOLD・既定置換なし・Weight公開なし** です。未学習72入力のモデル出力は生成していません。失敗ログ・旧条件・選定基準も維持しています。[均等学習の条件と全結果](docs/qwen35_full_utterance_balanced_v4_20261006.md)、[非昇格記録](promotions/qwen3.5-4b-full-utterance-v4-step192.json)を参照してください。

Natural specialist v5ではQwen3-14B teacherから実行役event-agreeの直接合格自然文を3件得ましたが、親step160からの専用継続はholdout 1/3のまま、Base specialistは0/3でした。move別few-shotもobjectへ賛同例が混入したためruntimeへ採用せず、全結果を [General Dialogue natural specialist v5実験記録](docs/general_dialogue_weight_v5_20260904.md) に残しています。

Natural specialists v6では12の異なるtopicから仮説object 12件・実行event-agree 13件を集め、人格・phase・move別LoRAとrepair LoRAを評価しました。数値創作を拒否するgrounding guardは採用しましたが、全Weightが未学習holdoutで親同等以下だったため非昇格です。全target、評価、SHA、停止理由は [General Dialogue natural specialists v6実験記録](docs/general_dialogue_weight_v6_20260904.md) にあります。

発話行為をコードで確定し、検証済みstatement本文だけを接続する`--no-renderer`経路は、4人格のすり合わせを残したまま39.2秒へ短縮しました。設計、実測、会話全文は [Composed statement renderer v1](docs/composed_statement_renderer_v1_20260904.md) にあります。

Claim Body v1は後の監査で、提案を完了事実へ変える文と名詞断片を通していたためsupersededにしました。[v1記録](docs/claim_body_weight_v1_20260904.md)と[旧昇格記録](promotions/qwen3-1.7b-claim-body-v1-step64.json)は履歴として残しています。

現行のClaim Body v3 step128は、17 train topic・585件のclean targetでBaseから学習しました。完全除外したemail / EV / bike 15ケースでcontract valid `15/15`、strict schema `15/15`、競合claim `0`です。3 topicの1 round実走はいずれも公開6発言全てがWeight由来、fallback 0、hard gate通過でした。設定、v2/v4停止理由、会話全文、SHAは [Claim Body Weight v3](docs/claim_body_weight_v3_20260904.md)、運用境界は [現行昇格記録](promotions/qwen3-1.7b-claim-body-v3-step128.json) にあります。

2026-09-08にGeneral本文用v5/v5bを実際に追加学習しました。v5b step256は新しい未学習12問の直接合格が親v3の4件から7件へ改善しましたが、語尾補正込みでは親9件・候補8件で、意味の変化も残るため **HOLD・既定置換なし** です。上記v3の`15/15`は当時の条件での値で、今回の明示的な丁寧語promptと追加guardによる既存15問の再評価は親・候補とも`13/15`です。実Weight、8人実走の不合格箇所、Qwen3.5-4B Base比較は [General Claim Body v5実験記録](docs/general_weight_v5_20260908.md) を参照してください。

未学習の競馬システム改善ledgerでもClaim Body v3はfull runの本文`20/20`、reaction失敗0、矛盾summary 0でhard gateを通過しました。転移で見つかった対立グラフ整合性と凍結claim-aware validationの修正、判断権限の境界は [Horse Claim Body transfer v2](docs/horse_claim_body_transfer_v2_20260905.md) にあります。

同一claim本文のvalidated cacheにより、EV 2 event / 1 roundは公開内容を変えず、Body call `6→2`、総model call `14→10`、実行時間`52.5→41.9秒`へ短縮しました。

選択claimへ整合したstatementの不正D番号は既存sanitizerで正規化し、email 2 event / 1 roundのreconciliation repairを`4→1`、総model callを`14→11`、実行時間を`49.6→45.1秒`へ短縮しました。公開発言・投票・変更理由・summaryは同一です。

EV 8 event / 最大2 roundは16 model call・54.2秒で完走しました。Weightが凍結claimと完全一致するplain formを返した1件だけ文末を安全に丁寧語化し、公開12発言すべてWeight由来、fallback 0、hard gate通過です。

`--portable-context`を指定したrun JSONにはledger snapshot、人格順、表示名を埋め込めます。ledger本文を複製するため既定はOFFです。台風18号weather runは16/16 Weight発言・hard gate通過後、ExtremeWeatherの純Swift importer、無関係fixtureからの復元、改変ledger拒否、iPhone 16 Simulatorでの画面表示まで確認しました。[実装・画面証跡](docs/extremeweather_portable_import_20260904.md)

物理iPhone 13 Pro / A15 / iOS 17.6.1では、`Qwen3-1.7B-4bit + Claim Body v3`を端末内MLXで直接ロード・生成・unloadできました。warm runはTTFT 2.306秒、25.732 tok/s、total 4.558秒、thermal nominalです。これは本文1件の実機スモークであり、完全な多人数CoDの実機完走ではありません。[実装・実測・再現手順](docs/iphone13_a15_claim_body_v3_smoke_20260904.md)

2026-10-06にはMacと物理iPhone 13 Proの独立job分担と、実験patch付きBackburnerのQwen3.5-4B末尾4層分担を両方実行しました。独立jobは起動・回収込み11.821秒でMac単独4.458秒より遅く、baselineの本文契約も未達です。層分担は全probeの生成token一致、42 remote chunk、fallback0を確認。batch調整後の小さな固定入力では中央値5.072秒→4.349秒でしたが、ばらつきがあり **研究候補・既定経路への採用なし** です。分散学習ではありません。[全測定と再現手順](docs/distributed_iphone13_backburner_20261006.md)、[A15互換patch](patches/backburner-a15/README.md)を参照してください。

続く4人格の独立本文soakは11.904秒、4/4 contract valid、exact polite 2/4、thermal nominalでした。各session後にMLX buffer cacheを解放し、peak footprintを2,510.879→1,478.894 MiB（-41.1%）、memory-limit headroomを561.137→2,058.169 MiBへ改善しました。生成中cancel後のAdapter unload / cache 0も実機確認済みです。このsoak単体ではBaseによる主張選択とreconciliationを実行していません。

その後、物理A15内で`Qwen3-0.6B`の盲検claim/evidence選択、異論側だけの再投票、Base選択D番号からの変更理由合成、Claim Body v3由来本文による1 round Native CoDを実行しました。架空の均衡fixtureで初期3案、異議・変更・賛同、最終2対2の未解決保持までhard gate pass。現行runnerの回帰runは構造6 call、本文cache 7 hit、26.846秒、peak 895.893 MiB、thermal nominalです。Weight本文は同じ実機で生成・検証済みのSHA付き永続cacheを利用します。[実測・全HOLD・会話全文](docs/iphone13_a15_native_cod_20260904.md)

実データshadowとして、台風18号の2026-08-25 15:50 JST固定packetをsource URL・観測/有効時刻付きの[歴史的replay ledger](data/typhoon18_20260825/native_cod_replay_ledger.json)へ縮約し、物理iPhoneで実行しました。4つの直交見解を保持し、北東転向外れの扱いだけを再討論して`unresolved_plurality`を記録。検証済み3 runは13.631〜13.735秒、peak 1,063.425〜1,164.175 MiB、thermal nominal、fallback 0、semantic完全一致です。[会話・性能・失敗履歴](docs/iphone13_a15_native_cod_20260904.md)

物理A15では、ExtremeWeatherのMetal雲・雨shaderを固定試験雲場で描画しながらNative CoDを実行するshadowも通過しました。384×256 / 24 steps / 20 fps目標で、CoD 14.834秒、描画平均19.20 fps、単独実行との意味出力一致。3D切替cancelは1.413秒、模擬メモリ警告cancelは1.384秒で、MLX解放後の描画回復を確認しました。Unity・地図・通信を含む本体全体は未検証です。[実測・停止検証・再現手順](docs/iphone13_a15_3d_shadow_20260907.md)

続く本文cache miss試験では、0.6B判断後に1.7B + Claim Body v3から全6本文を新規生成し、3D同時実行36.041秒・19.56 fps、peak 1,549 MiBで外部gateを通過しました。本文生成中のcancelは3D要求2.241秒、模擬memory warning 0.935秒。既存cacheは変更せず、Base判断と公開発言の一致も確認しています。[本文新規生成・解放・再現手順](docs/iphone13_a15_fresh_body_20260908.md)

Hugging Face向けには、ローカルpathを除いたAdapter設定、Model Card、Weight、SHA256SUMSだけのstaging packageを用意しています。公開前検証とupload境界は [Hugging Face release staging](docs/huggingface_release_claim_body_v3.md) を参照してください。

## 議論と実測のPDCA

複数人格が改善案を出し、実験コマンドの測定結果を使って次の案を修正します。
同梱例はCoD本文プロンプトの改善で、開発用の失敗を議論し、最終候補だけ別topicでも検証します。

```sh
python3.11 cod_model.py pdca \
  --task configs/pdca-general-body.json --out runs/pdca-body-001 \
  --model qwen3.5:4b --min-rounds 2 --max-rounds 3
```

Generalは既定8人格。参加者・モデル・回数を変更でき、他プロジェクトの測定器にも接続できます。
[使い方・実験JSON・採用ルール・実測記録](docs/pdca_20261004.md)を参照してください。

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
