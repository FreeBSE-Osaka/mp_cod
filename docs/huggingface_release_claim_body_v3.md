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
- `hf auth whoami`: not logged in（upload未実行）

```text
adapter_config.json  078a2db9d9eb4d7df1c5fb2db1386d47427aaa05e330cbfdf107b7a46532622f
adapters.safetensors 4ce21e64af220f0ee309599e189fd136e10c4c5cd11440c3d60fd306749a9a92
README.md            d2cfdc0a0d8b352b65827c66e09b9b62d6f994989ec2c0f2a0f8efb69db632f2
package smoke raw    fdd712c06427b7c61b86d016c6c4861cf73e2117ac20f03e09b7b894e31f6276
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
