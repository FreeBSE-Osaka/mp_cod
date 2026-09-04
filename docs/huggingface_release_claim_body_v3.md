# Hugging Face release staging — Claim Body v3

## Status

公開前のstaging packageを作成済み。Hugging Face repositoryはまだ作成・更新していない。

```text
/Volumes/data4/cod_model_weight/releases/mp-cod-claim-body-v3/
  README.md
  adapter_config.json
  adapters.safetensors
  SHA256SUMS
```

元のMLX学習用`adapter_config.json`にはローカルmodel・dataset・出力pathが含まれる。配布版はMLX推論に必要な`fine_tune_type`、`num_layers`、`lora_parameters`と、base model・contract・source commitだけへ縮めた。

検証結果:

- Hugging Face `ModelCard.validate()`: pass
- package内ローカル絶対path: 0
- sanitized configからのMLX load: pass
- strict claim-body生成: pass
- staging packageを使ったemail 1 round: Weight 6/6、repair 0、fallback 0、hard gate pass
- 同じ配布config / Weightを物理iPhone 13 Pro / A15で直接load・4人格本文生成・cancel・unload: pass
- 同じ実機Weight rawの検証済み永続cacheを使った架空fixture Native CoD: 7 event、17.245秒、hard gate pass
- `hf auth whoami`: not logged in（upload未実行）

```text
adapter_config.json  4e9dc5ffbe79d381b58e931e0a4a4ec5dc5b879786d733fd1212ef7880460e48
adapters.safetensors 4ce21e64af220f0ee309599e189fd136e10c4c5cd11440c3d60fd306749a9a92
README.md            a451287aa71fcf210b0f852730f9813aa636e85fc853b51bd2c93e6116b4c5c5
package smoke raw    fdd712c06427b7c61b86d016c6c4861cf73e2117ac20f03e09b7b894e31f6276
iPhone result raw    c3008004f01ddf941f9b6080e1f8df869ca5c35e8f432a67616a4bebcf9bd8ba
iPhone 4-body raw    8205792a922733004623534431f7f68d8332ae49c4e39aa1dd96d7f018a61fba
iPhone cancel raw    869313c595f312f97c42417ae2b8d196e85485e8cdd05f02eee5098e45d6f1cd
iPhone Native CoD   b38cbcaf46deaaf2d9309149849261daf463c98e882e5bfb6493cebdc47c2d75
iPhone CoD repeat   f1509870b71d6ed514017774db22858100acfc3f5ee9561a064295dc0f26179f
```

package smoke rawは`/Volumes/data4/cod_model_weight/evaluations/claim-body-v3/hf_package_email_smoke.json`へ保存した。

## Pre-publication verification

```sh
cd /Volumes/data4/cod_model_weight/releases/mp-cod-claim-body-v3
shasum -a 256 -c SHA256SUMS

/Volumes/data4/cod_model_weight/venv/bin/python3.11 \
  /Users/osaka/src/cod_model/cod_model.py event-debate \
  --ledger /Users/osaka/src/cod_model/data/general_v2_email_triage_holdout/claim_ledger.json \
  --domain general \
  --backend mlx \
  --model-path /Volumes/data4/cod_model_weight/models/Qwen3-1.7B-4bit \
  --body-adapter /Volumes/data4/cod_model_weight/releases/mp-cod-claim-body-v3 \
  --max-turns 2 \
  --reconcile-rounds 1
```

## Upload

repository IDを決め、公開操作を行う時だけ実行する。

```sh
HF_REPO_ID=YOUR_HF_ACCOUNT/mp-cod-claim-body-v3
HF_PACKAGE=/Volumes/data4/cod_model_weight/releases/mp-cod-claim-body-v3

/Volumes/data4/cod_model_weight/venv/bin/hf auth whoami
/Volumes/data4/cod_model_weight/venv/bin/hf upload \
  "$HF_REPO_ID" "$HF_PACKAGE" . \
  --commit-message "Publish MP-CoD Claim Body v3 MLX adapter"
```

`hf upload`はrepositoryが存在しない場合に作成し得る外部変更なので、repo IDと公開意思が確定するまで実行しない。

## Boundaries

- Base Weightを同梱しない
- datasetと評価rawを同梱しない
- ローカル絶対pathを`adapter_config.json`へ含めない
- Adapter単独のchat利用を推奨しない
- `cod_model.py`のvalidatorとfallbackを必須とする
- fused model、GGUF、iPhone bundleはこのreleaseへ含めない
- 物理iPhoneのNative CoD PASSは架空fixture・1 round・検証済み本文cacheの範囲で、実データや長い多人数CoDの性能保証にしない
