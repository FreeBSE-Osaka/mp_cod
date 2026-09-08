# General: multiple viewpoints and flexible discussion

Generalを8つの専門観点へ広げ、反論役・賛成役を固定しない討論に変更しました。
同じ意見を複数人が支持すること、問題点だけを指摘すること、賛同しつつ改良することを許可します。
実Weightの学習と8人実走の結果は [実験記録](general_weight_v5_20260908.md) を参照してください。

## Behavior

General has eight available viewpoints: hypothesis construction, critical design,
empirical audit, practical execution, user experience, resource constraints,
long-term effects, and cross-disciplinary design. These are evaluation methods,
not assigned yes/no positions. Several speakers may reach the same conclusion,
with the same or different evidence. Agreement does not create independent evidence.

`event-debate --domain general` selects `--discussion-style flexible` automatically.
Other domains retain structured behavior. Pass `--discussion-style structured`
to reproduce the previous alternative-required objection policy.

Flexible discussion supports:

- a critique without an invented alternative;
- a counterproposal;
- agreement without a forced addendum;
- improvement of a supported proposal;
- an explanation added to one's own earlier claim;
- conditions, observations, retained views, and changed views.

The event queue still gives reactions priority and does not rotate speakers mechanically.
The actor count is the active `role_preferences` subset, with a minimum of two.
There is no quota forcing all speakers to disagree. Flexible scoring does not reward
distinct conclusions or speech-act diversity; those counts remain diagnostics.
The initial claim count is a maximum, not a quota. Missing or invalid model choices
are not filled from the catalog. At least two actors must have valid model claims.
The same claim from several actors is excluded from near-duplicate penalties, but
this support does not establish that they are statistically independent experts.

The free-topic `debate --domain general` route also removes the instruction to attack
someone even when all participants agree. It prints each model's `response` and
allows support, improvement, conditions, or a problem-only critique.

## Ledger relationships

`contradicts` means mutually incompatible claims for reconciliation voting.
`challenges` means a directed critique of a premise; it does not make both claims
mutually exclusive. `extends` means a supporting/additional claim, including an
improvement. All targets must be known claim codes; self-references are rejected.

Optional claim `kind` values are `proposal`, `critique`, `improvement`, `condition`,
`observation`, and `support`. They describe the claim's content, not the speaker's
permanent role. A critique must not be labelled a counterproposal just to satisfy
a schema. Same-claim support by different speakers is retained; same-speaker
duplicate claim entries remain invalid.

The Base chooses claims, evidence and votes. The body-only LoRA renders the selected
claim after that decision. Display prefixes are code-composed, not hidden model
reasoning. The event log retains both `action` and `dialogue_move`.
Reconciliation transports pair choices as `LEFT` / `RIGHT`, counterbalances their
display order across actors, and restores canonical codes into the saved votes.
Each vote records `choice_transport`; invalid choices abstain instead of guessing
from substrings. This reduces a formatting/position-bias hazard, not all model bias.

This evidence-led entry point still rejects claims outside the fixed ledger.
"Flexible" does not mean unsupported facts may be introduced. The separate free-topic
debate route has a broader generation scope and requires independent review.

## Eight-speaker example

The umbrella fixture is synthetic, based on a short outing and uncertain rain timing.
It is not a current weather forecast. All eight actors can consider all eight claims;
ordering expresses focus but does not forbid any position.

```sh
<mlx-python> cod_model.py event-debate \
  --domain general --backend mlx \
  --model-path /path/to/Qwen3-1.7B-4bit \
  --body-adapter /path/to/validated-body-adapter \
  --ledger data/general_flexible_umbrella/claim_ledger.json \
  --max-turns 24 --reconcile-rounds 1
```

Use an evaluated adapter only. Eight viewpoints share the same loaded Base and renderer
weights; this does not load eight full model copies. More independent calls still take
more inference time.
For a renderer-free smoke, replace `--body-adapter ...` with `--no-renderer`; its
display text is composed from validated statements, not evidence of learned Weight
quality. The v5b candidate in the linked experiment is HOLD, not a validated release.

Without a fixed claim ledger, use the separate open-topic entry point:

```sh
python3.11 cod_model.py debate --domain general --model qwen3.5:4b \
  "架空の相談: 雨が降るか分からない日の短い外出に傘を持つべきか"
```

That entry point still uses staged blind/review/moderator calls rather than the
event scheduler. Its schema is checked, but novel claims are not verified against
the fixed evidence ledger. The eight-speaker live test in this revision exercised
`event-debate`, not this broader open-topic route.

## Compatibility and review

The run records `discussion_style`; RSI comparisons require matching styles.
The metric policy is `flexible_evidence_v1` for General and `structured_v1` otherwise.
Older clients may enforce the former objection wording or ignore new dialogue moves.
Use structured mode for those clients until their importer is updated. This change
does not claim that the native iPhone importer supports every flexible move.

The body validator additionally rejects multiple sentences for the one-body contract
and loss of a negative conditional such as `増えなければ` becoming `人数に応じて`.
It also checks maximum-duration language, loan/cancellation/storage/disposal polarity,
plan-to-completed-action changes, and speaker-name attribution absent from the claim.
Known plain-verb plus `です` endings such as `伝えるです` are not valid conjugations.
Safe terminal conjugations such as `置く` → `置きます` are separately logged as repairs;
they do not count as direct model success. Claim-level body caching remains enabled,
so speakers sharing a claim may use the same sentence. That is a latency tradeoff,
not eight independently generated styles.
These are conservative checks, not a proof of full semantic equivalence.
