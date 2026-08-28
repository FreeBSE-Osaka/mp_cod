# 近畿地域の活断層評価 2026-08-28

地震調査研究推進本部が2026年8月28日に公表した「近畿地域の活断層の長期評価（第一版）」を、CoDの根拠ID付き議論へ渡すための固定packetです。

## Sources

- [近畿地域の活断層の地域評価](https://www.jishin.go.jp/evaluation/long_term_evaluation/regional_evaluation/kinki-detail/)
- [近畿地域の活断層の長期評価（第一版）のポイント](https://www.jishin.go.jp/main/chousa/26aug_chi_kinki/kinki_point.pdf)
- [近畿地域の活断層の長期評価（第一版）の概要](https://www.jishin.go.jp/main/chousa/26aug_chi_kinki/kinki_gaiyo.pdf)

## Boundary

- 30年確率の解釈を議論するpacketであり、地震の日時、直近の切迫度、特定地点の揺れを予測するものではありません。
- 2016年熊本地震前の布田川断層帯が「ほぼ0〜0.9%」だった教訓は、低い確率を安全情報と誤読しないための根拠としてのみ使用します。
- 地域50〜60%と個別断層のS/A/Z/Xは異なる尺度です。
- 実データrunは全発言を人間が再確認し、hard gate合格だけでは公開結論や学習承認にしません。
- 2026-08-28のQwen3.5-4B実走は、1件の証拠ID違反と「マグニチュード」を「最大震度」とした誤表現があったため、学習用にrejectしました。
