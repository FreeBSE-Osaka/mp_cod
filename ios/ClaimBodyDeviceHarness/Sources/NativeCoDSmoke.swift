import Darwin
import CryptoKit
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

struct NativeCoDDataItem: Codable {
    let id: String
    let text: String
    let sourceIDs: [String]?
    let observedAt: String?
    let validUntil: String?

    init(
        id: String,
        text: String,
        sourceIDs: [String]? = nil,
        observedAt: String? = nil,
        validUntil: String? = nil
    ) {
        self.id = id
        self.text = text
        self.sourceIDs = sourceIDs
        self.observedAt = observedAt
        self.validUntil = validUntil
    }

    enum CodingKeys: String, CodingKey {
        case id, text
        case sourceIDs = "source_ids"
        case observedAt = "observed_at"
        case validUntil = "valid_until"
    }
}

struct NativeCoDClaim: Codable {
    let code: String
    let label: String
    let supportedBy: [String]
    let contradicts: [String]
    let requiredTerms: [String]

    init(
        code: String,
        label: String,
        supportedBy: [String],
        contradicts: [String] = [],
        requiredTerms: [String] = []
    ) {
        self.code = code
        self.label = label
        self.supportedBy = supportedBy
        self.contradicts = contradicts
        self.requiredTerms = requiredTerms
    }

    enum CodingKeys: String, CodingKey {
        case code, label, contradicts
        case supportedBy = "supported_by"
        case requiredTerms = "required_terms"
    }
}

struct NativeCoDPersona: Codable {
    let name: String
    let objective: String
}

struct NativeCoDSource: Codable {
    let id: String
    let name: String
    let url: String
    let authority: String
}

struct NativeCoDProvenance: Codable {
    let liveRetrieval: Bool
    let parentLedger: String
    let parentLedgerSHA256: String
    let dataPacket: String
    let dataPacketSHA256: String

    enum CodingKeys: String, CodingKey {
        case liveRetrieval = "live_retrieval"
        case parentLedger = "parent_ledger"
        case parentLedgerSHA256 = "parent_ledger_sha256"
        case dataPacket = "data_packet"
        case dataPacketSHA256 = "data_packet_sha256"
    }
}

struct NativeCoDLedger: Codable {
    let scenarioID: String
    let fixtureKind: String
    let historicalReplay: Bool
    let cutoff: String?
    let topic: String
    let provenance: NativeCoDProvenance?
    let sources: [NativeCoDSource]
    let data: [NativeCoDDataItem]
    let claims: [NativeCoDClaim]
    let personas: [NativeCoDPersona]
    let rolePreferences: [String: [String]]

    init(
        scenarioID: String,
        fixtureKind: String,
        historicalReplay: Bool,
        cutoff: String? = nil,
        topic: String,
        provenance: NativeCoDProvenance? = nil,
        sources: [NativeCoDSource] = [],
        data: [NativeCoDDataItem],
        claims: [NativeCoDClaim],
        personas: [NativeCoDPersona],
        rolePreferences: [String: [String]]
    ) {
        self.scenarioID = scenarioID
        self.fixtureKind = fixtureKind
        self.historicalReplay = historicalReplay
        self.cutoff = cutoff
        self.topic = topic
        self.provenance = provenance
        self.sources = sources
        self.data = data
        self.claims = claims
        self.personas = personas
        self.rolePreferences = rolePreferences
    }

    enum CodingKeys: String, CodingKey {
        case cutoff, topic, provenance, sources, data, claims, personas
        case scenarioID = "scenario_id"
        case fixtureKind = "fixture_kind"
        case historicalReplay = "historical_replay"
        case rolePreferences = "role_preferences"
    }
}

struct NativeCoDPosition: Encodable {
    let persona: String
    let claim: String
    let dataIDs: [String]
    let confidence: Int
    let changeReason: String?
    let origin: String

    enum CodingKeys: String, CodingKey {
        case persona, claim, confidence, origin
        case dataIDs = "data_ids"
        case changeReason = "change_reason"
    }
}

struct NativeCoDStructuralCall: Encodable {
    let phase: String
    let persona: String
    let attempt: Int
    let adapterActive: Bool
    let rawOutput: String
    let jsonExtracted: Bool
    let valid: Bool
    let validationError: String?
    let ttftSeconds: Double
    let promptTokens: Int
    let generationTokens: Int
    let tokensPerSecond: Double
}

struct NativeCoDBodyCall: Encodable {
    let claim: String
    let speaker: String
    let rawOutput: String
    let body: String
    let valid: Bool
    let sanitized: Bool
    let fallback: Bool
    let validationError: String?
    let ttftSeconds: Double
    let promptTokens: Int
    let generationTokens: Int
    let tokensPerSecond: Double
}

struct NativeCoDEvent: Encodable {
    let id: String
    let phase: String
    let persona: String
    let move: String
    let targetEventID: String?
    let claim: String
    let dataIDs: [String]
    let confidence: Int
    let changeReason: String?
    let body: String
    let bodyOrigin: String
    let utterance: String

    enum CodingKeys: String, CodingKey {
        case id, phase, persona, move, claim, confidence, body, utterance
        case targetEventID = "target_event_id"
        case dataIDs = "data_ids"
        case changeReason = "change_reason"
        case bodyOrigin = "body_origin"
    }
}

struct NativeCoDMemorySample: Encodable {
    let stage: String
    let footprintBytes: UInt64
    let footprintPeakBytes: UInt64
    let limitBytesRemaining: UInt64
    let mlxActiveBytes: UInt64
    let mlxCacheBytes: UInt64
    let mlxPeakBytes: UInt64
}

struct NativeCoDResult: Encodable {
    let schemaVersion: Int
    let createdAt: Date
    let mode: String
    let resultFilename: String
    let model: String
    let bodyModel: String
    let bodyModelLoaded: Bool
    let adapterWeightsSHA256: String
    let ledgerSHA256: String
    let ledger: NativeCoDLedger
    let personas: [NativeCoDPersona]
    let structuralCalls: [NativeCoDStructuralCall]
    let initialPositions: [NativeCoDPosition]
    let reconciliationPositions: [NativeCoDPosition]
    let initialTally: [String: Int]
    let finalTally: [String: Int]
    let consensusClaim: String?
    let outcomeStatus: String
    let events: [NativeCoDEvent]
    let bodyCalls: [NativeCoDBodyCall]
    let structuralRepairs: Int
    let structuralJSONExtractions: Int
    let bodyRendererModelCalls: Int
    let bodyRendererCacheHits: Int
    let persistentBodyCacheEntries: Int
    let persistentBodyCacheSourceSHA256: String?
    let persistentBodyCache: NativeBodyCacheFile
    let bodyPolitenessSanitizations: Int
    let bodyFallbacks: Int
    let reconciliationModelSpeakers: [String]
    let retainedInitialVotes: Int
    let structuralAdapterActive: Bool
    let bodyAdapterLoadRequired: Bool
    let bodyAdapterLoaded: Bool
    let bodyAdapterLoadedAfterStructuralCalls: Bool
    let bodyCachePrimeMode: Bool
    let adapterUnloaded: Bool
    let hardGatePass: Bool
    let baseLoadSeconds: Double
    let bodyBaseLoadSeconds: Double
    let adapterLoadSeconds: Double
    let totalSeconds: Double
    let startThermalState: String
    let thermalState: String
    let peakFootprintBytes: UInt64
    let minimumLimitBytesRemaining: UInt64
    let memorySamples: [NativeCoDMemorySample]

    var transcript: String {
        events.map { "\($0.persona):\n\($0.utterance)" }.joined(separator: "\n\n")
    }
}

private struct NativeClaimRuntime {
    let claim: NativeCoDClaim
    let requiredTerms: [String]
}

private struct NativeChoice {
    let claim: String
    let dataIDs: [String]
    let confidence: Int
    let changeReason: String?
}

private struct NativeSelection {
    let choice: NativeChoice
    let calls: [NativeCoDStructuralCall]
}

private struct NativeGeneration {
    let raw: String
    let ttftSeconds: Double
    let promptTokens: Int
    let generationTokens: Int
    let tokensPerSecond: Double
}

private struct NativeParsedChoice {
    let choice: NativeChoice
    let extracted: Bool
}

private struct NativeEventDraft {
    let phase: String
    let persona: String
    let move: String
    let targetEventID: String?
    let choice: NativeChoice
}

private struct NativePromptDataItem: Encodable {
    let id: String
    let text: String
}

private struct NativePromptClaim: Encodable {
    let code: String
    let label: String
    let supportedBy: [String]
}

private struct NativeBodyValue {
    let body: String
    let origin: String
}

struct NativeBodyCacheEntry: Codable {
    let claimLabel: String
    let rawOutput: String
    let body: String
    let sanitized: Bool
    let origin: String

    enum CodingKeys: String, CodingKey {
        case body, sanitized, origin
        case claimLabel = "claim_label"
        case rawOutput = "raw_output"
    }
}

struct NativeBodyCacheFile: Codable {
    let schemaVersion: Int
    let adapterWeightsSHA256: String
    let sourcePayloadSHA256: String
    var entries: [String: NativeBodyCacheEntry]

    enum CodingKeys: String, CodingKey {
        case entries
        case schemaVersion = "schema_version"
        case adapterWeightsSHA256 = "adapter_weights_sha256"
        case sourcePayloadSHA256 = "source_payload_sha256"
    }
}

enum NativeCoDScenarioKind {
    case syntheticBalanced
    case typhoon18Replay
}

private struct NativeScenarioRuntime {
    let ledger: NativeCoDLedger
    let ledgerSHA256: String
    let claimRuntimes: [NativeClaimRuntime]
    let resultFilename: String
    let cacheFilename: String
}

@MainActor
enum NativeCoDSmokeRunner {
    private static let modelID = "mlx-community/Qwen3-0.6B-4bit"
    private static let bodyModelID = "mlx-community/Qwen3-1.7B-4bit"
    private static let adapterSHA = "4ce21e64af220f0ee309599e189fd136e10c4c5cd11440c3d60fd306749a9a92"

    private static let data = [
        NativeCoDDataItem(
            id: "D01",
            text: "30人pilotでは作業時間が22%短縮し、crashは0件だった。"
        ),
        NativeCoDDataItem(
            id: "D02",
            text: "旧端末では電池消費が8%増えた。"
        ),
        NativeCoDDataItem(
            id: "D03",
            text: "30件中2件の要約は公開前に修正が必要だった。"
        ),
        NativeCoDDataItem(
            id: "D04",
            text: "feature flagなら5分以内に既存経路へ戻せる。"
        ),
        NativeCoDDataItem(
            id: "D05",
            text: "運用担当は展開後のincidentを30分以内に確認できる。"
        ),
    ]

    private static let claimRuntimes = [
        NativeClaimRuntime(
            claim: NativeCoDClaim(
                code: "PILOT",
                label: "100人へpilotを拡大し修正率と電池影響を再評価する",
                supportedBy: ["D01", "D02", "D03"],
                requiredTerms: ["100人", "pilot", "修正率", "電池影響", "再評価"]
            ),
            requiredTerms: ["100人", "pilot", "修正率", "電池影響", "再評価"]
        ),
        NativeClaimRuntime(
            claim: NativeCoDClaim(
                code: "ROLLOUT",
                label: "feature flagを維持したまま全利用者へ段階展開する",
                supportedBy: ["D01", "D04", "D05"],
                requiredTerms: ["feature flag", "全利用者", "段階展開"]
            ),
            requiredTerms: ["feature flag", "全利用者", "段階展開"]
        ),
        NativeClaimRuntime(
            claim: NativeCoDClaim(
                code: "DEVICE_SPLIT",
                label: "新しい端末だけ自動要約を有効化し旧端末は既存経路を保つ",
                supportedBy: ["D01", "D02", "D04"],
                requiredTerms: ["新しい端末", "自動要約", "有効化", "旧端末", "既存経路"]
            ),
            requiredTerms: ["新しい端末", "自動要約", "有効化", "旧端末", "既存経路"]
        ),
    ]

    private static let personas = [
        NativeCoDPersona(
            name: "批判的設計者",
            objective: "未検証の障害と後戻りコストを最小化し、最も検証可能な次の一手を選ぶ。"
        ),
        NativeCoDPersona(
            name: "仮説構築者",
            objective: "便益を受ける利用者数と実環境の情報量を最大化し、可逆な時にpilotだけを繰り返す遅延を損失とする。"
        ),
        NativeCoDPersona(
            name: "実証監査者",
            objective: "証拠の強さと再現可能性を最大化し、未検証項目を既成事実にしない。"
        ),
        NativeCoDPersona(
            name: "実行設計者",
            objective: "旧端末の電池回帰を避けつつ新端末へ届ける運用価値を最大化し、全端末を同じ経路で扱うことを損失とする。"
        ),
    ]

    private static let rolePreferences = [
        "批判的設計者": ["PILOT", "DEVICE_SPLIT"],
        "仮説構築者": ["ROLLOUT", "DEVICE_SPLIT"],
        "実証監査者": ["PILOT", "DEVICE_SPLIT"],
        "実行設計者": ["DEVICE_SPLIT", "ROLLOUT"],
    ]

    private static var ledger: NativeCoDLedger {
        NativeCoDLedger(
            scenarioID: "synthetic_balanced_v1",
            fixtureKind: "synthetic_balanced",
            historicalReplay: false,
            topic: "架空の災害通知アプリの自動要約機能をどう段階導入するか",
            data: data,
            claims: claimRuntimes.map(\.claim),
            personas: personas,
            rolePreferences: rolePreferences
        )
    }

    private static func loadScenario(_ kind: NativeCoDScenarioKind) throws -> NativeScenarioRuntime {
        switch kind {
        case .syntheticBalanced:
            let syntheticLedger = ledger
            try validateScenario(syntheticLedger, historicalReplayRequired: false)
            let encoded = try jsonString(syntheticLedger)
            return NativeScenarioRuntime(
                ledger: syntheticLedger,
                ledgerSHA256: sha256(Data(encoded.utf8)),
                claimRuntimes: claimRuntimes,
                resultFilename: "mp_cod_a15_native_cod.json",
                cacheFilename: "mp_cod_claim_body_cache.json"
            )
        case .typhoon18Replay:
            guard let url = Bundle.main.url(
                forResource: "native_cod_replay_ledger",
                withExtension: "json"
            ) else {
                throw NativeCoDError.scenarioMissing
            }
            let payload = try Data(contentsOf: url)
            let digest = sha256(payload)
            guard digest == "53683c7ebdba75861981e0d8ee2729965d09bb97d4d829432c0ed15e83b1349c" else {
                throw NativeCoDError.scenarioInvalid("台風18号replay ledger SHA不一致")
            }
            let ledger = try JSONDecoder().decode(NativeCoDLedger.self, from: payload)
            try validateScenario(ledger, historicalReplayRequired: true)
            return NativeScenarioRuntime(
                ledger: ledger,
                ledgerSHA256: digest,
                claimRuntimes: ledger.claims.map {
                    NativeClaimRuntime(claim: $0, requiredTerms: $0.requiredTerms)
                },
                resultFilename: "mp_cod_a15_typhoon18_replay.json",
                cacheFilename: "mp_cod_typhoon18_body_cache.json"
            )
        }
    }

    private static func validateScenario(
        _ ledger: NativeCoDLedger,
        historicalReplayRequired: Bool
    ) throws {
        let dataIDs = ledger.data.map(\.id)
        let claimCodes = ledger.claims.map(\.code)
        let personaNames = ledger.personas.map(\.name)
        guard !ledger.scenarioID.isEmpty,
              !ledger.topic.isEmpty,
              Set(dataIDs).count == dataIDs.count,
              Set(claimCodes).count == claimCodes.count,
              ledger.personas.count == 4,
              Set(personaNames).count == personaNames.count,
              Set(ledger.rolePreferences.keys) == Set(personaNames),
              (!historicalReplayRequired || (ledger.historicalReplay && ledger.cutoff != nil)),
              ledger.claims.allSatisfy({ claim in
                  !claim.requiredTerms.isEmpty
                    && Set(claim.supportedBy).isSubset(of: Set(dataIDs))
                    && Set(claim.contradicts).isSubset(of: Set(claimCodes).subtracting([claim.code]))
              }),
              ledger.rolePreferences.allSatisfy({ _, codes in
                  codes.count >= 2 && Set(codes).isSubset(of: Set(claimCodes))
              }),
              ledger.data.allSatisfy({ item in
                  !historicalReplayRequired
                    || (item.sourceIDs?.isEmpty == false
                        && item.observedAt != nil
                        && item.validUntil != nil)
              }),
              ledger.sources.allSatisfy({ source in
                  URL(string: source.url)?.scheme == "https"
              }) else {
            throw NativeCoDError.scenarioInvalid("scenario schemaまたは参照整合性が不正")
        }
    }

    static func run(
        scenario kind: NativeCoDScenarioKind = .syntheticBalanced,
        allowBodyCachePrime: Bool = false,
        progress: (String) -> Void
    ) async throws -> NativeCoDResult {
        let scenario = try loadScenario(kind)
        let ledger = scenario.ledger
        let claimRuntimes = scenario.claimRuntimes
        let personas = ledger.personas
        let rolePreferences = ledger.rolePreferences
        let totalStart = ContinuousClock.now
        let startThermalState = thermalName(ProcessInfo.processInfo.thermalState)
        var persistentBodyCache = try loadOrSeedBodyCache(scenario)
        let cacheComplete = Set(claimRuntimes.map(\.claim.code)).isSubset(
            of: Set(persistentBodyCache?.entries.keys ?? Dictionary<String, NativeBodyCacheEntry>().keys)
        )
        guard startThermalState == "nominal"
                || (startThermalState == "fair" && (cacheComplete || allowBodyCachePrime)) else {
            throw NativeCoDError.deviceTooHot(startThermalState)
        }
        Memory.peakMemory = 0
        var memorySamples = [try memorySample(stage: "start")]

        var structuralCalls: [NativeCoDStructuralCall] = []
        var initialPositions: [NativeCoDPosition] = []
        var reconciliationPositions: [NativeCoDPosition] = []
        var reconciliationModelSpeakers: [String] = []
        var activeSpeakers: Set<String> = []
        var baseLoadSeconds = 0.0
        var initialTally: [String: Int] = [:]
        var leadingClaim: String?

        do {
            progress("Native CoD 0.6B Baseをロード中")
            let loadStart = ContinuousClock.now
            let structureContainer = try await #huggingFaceLoadModelContainer(
                configuration: LLMRegistry.qwen3_0_6b_4bit
            )
            baseLoadSeconds = seconds(since: loadStart)
            memorySamples.append(try memorySample(stage: "after_structure_base_load"))

            for persona in personas {
                try Task.checkCancellation()
                progress("盲検選択: \(persona.name)")
                let allowedCodes = rolePreferences[persona.name] ?? []
                let personaLedger = NativeInitialContext(
                    scenarioID: ledger.scenarioID,
                    historicalReplay: ledger.historicalReplay,
                    cutoff: ledger.cutoff,
                    topic: ledger.topic,
                    data: ledger.data.map {
                        NativePromptDataItem(id: $0.id, text: $0.text)
                    },
                    claims: allowedCodes.compactMap { code in
                        claimRuntimes.first(where: { $0.claim.code == code }).map {
                            NativePromptClaim(
                                code: $0.claim.code,
                                label: $0.claim.label,
                                supportedBy: $0.claim.supportedBy
                            )
                        }
                    }
                )
                let selection = try await selectChoice(
                    phase: "initial",
                    persona: persona,
                    context: "固定ledger:\n\(try jsonString(personaLedger))",
                    allowedCodes: allowedCodes,
                    claimRuntimes: claimRuntimes,
                    container: structureContainer
                )
                structuralCalls.append(contentsOf: selection.calls)
                let position = position(
                    persona: persona.name,
                    choice: selection.choice,
                    origin: "model"
                )
                initialPositions.append(position)
                memorySamples.append(
                    try memorySample(stage: "after_initial_\(initialPositions.count)")
                )
                print("MP_COD_NATIVE_BASE phase=initial persona=\(persona.name) claim=\(position.claim) data=\(position.dataIDs)")
            }

            initialTally = tally(initialPositions)
            leadingClaim = uniqueLeader(initialTally) ?? initialPositions.first?.claim
            let dissenters = initialPositions
                .filter { $0.claim != leadingClaim }
                .sorted { left, right in
                    let leftConflict = claimsConflict(
                        left.claim,
                        leadingClaim,
                        claimRuntimes: claimRuntimes
                    )
                    let rightConflict = claimsConflict(
                        right.claim,
                        leadingClaim,
                        claimRuntimes: claimRuntimes
                    )
                    if leftConflict != rightConflict { return leftConflict && !rightConflict }
                    if left.confidence != right.confidence { return left.confidence > right.confidence }
                    return personaIndex(left.persona, personas: personas)
                        < personaIndex(right.persona, personas: personas)
                }
            let conflictingDissenters = dissenters.filter {
                claimsConflict(
                    $0.claim,
                    leadingClaim,
                    claimRuntimes: claimRuntimes
                )
            }
            let activePool = conflictingDissenters.isEmpty ? dissenters : conflictingDissenters
            reconciliationModelSpeakers = Array(activePool.prefix(2).map(\.persona))
            activeSpeakers = Set(reconciliationModelSpeakers)
            let reconciliationContext = try jsonString(
                NativeReconciliationContext(
                    ledger: NativeReconciliationLedger(
                        fixtureKind: ledger.fixtureKind,
                        topic: ledger.topic,
                        data: ledger.data.map {
                            NativePromptDataItem(id: $0.id, text: $0.text)
                        },
                        claims: ledger.claims.map {
                            NativePromptClaim(
                                code: $0.code,
                                label: $0.label,
                                supportedBy: $0.supportedBy
                            )
                        }
                    ),
                    initialPositions: initialPositions,
                    tally: initialTally
                )
            )

            for persona in personas {
                if !activeSpeakers.contains(persona.name),
                   let prior = initialPositions.first(where: { $0.persona == persona.name }) {
                    reconciliationPositions.append(
                        NativeCoDPosition(
                            persona: prior.persona,
                            claim: prior.claim,
                            dataIDs: prior.dataIDs,
                            confidence: prior.confidence,
                            changeReason: nil,
                            origin: "retained_initial"
                        )
                    )
                    print("MP_COD_NATIVE_BASE phase=reconciliation persona=\(persona.name) retained=\(prior.claim)")
                    continue
                }
                try Task.checkCancellation()
                progress("すり合わせ: \(persona.name)")
                let priorClaim = initialPositions.first {
                    $0.persona == persona.name
                }?.claim
                let conflictCodes = claimRuntimes.map(\.claim.code).filter { code in
                    code == priorClaim
                        || claimsConflict(
                            code,
                            priorClaim,
                            claimRuntimes: claimRuntimes
                        )
                }
                let reconciliationCodes = conflictCodes.count >= 2
                    ? conflictCodes
                    : claimRuntimes.map(\.claim.code)
                let selection = try await selectChoice(
                    phase: "reconciliation",
                    persona: persona,
                    context: "構造化context:\n\(reconciliationContext)",
                    allowedCodes: reconciliationCodes,
                    claimRuntimes: claimRuntimes,
                    container: structureContainer
                )
                structuralCalls.append(contentsOf: selection.calls)
                let selectedChoice: NativeChoice
                if let priorClaim, selection.choice.claim != priorClaim {
                    selectedChoice = NativeChoice(
                        claim: selection.choice.claim,
                        dataIDs: selection.choice.dataIDs,
                        confidence: selection.choice.confidence,
                        changeReason: structuredChangeReason(
                            priorClaim: priorClaim,
                            selectedChoice: selection.choice
                        )
                    )
                } else {
                    selectedChoice = selection.choice
                }
                let position = position(
                    persona: persona.name,
                    choice: selectedChoice,
                    origin: "model"
                )
                reconciliationPositions.append(position)
                memorySamples.append(
                    try memorySample(stage: "after_reconciliation_\(reconciliationPositions.count)")
                )
                print("MP_COD_NATIVE_BASE phase=reconciliation persona=\(persona.name) claim=\(position.claim) data=\(position.dataIDs)")
            }
        }

        Memory.clearCache()
        memorySamples.append(try memorySample(stage: "after_structure_base_release"))
        let finalTally = tally(reconciliationPositions)
        let consensusClaim = finalTally
            .sorted { left, right in left.value == right.value ? left.key < right.key : left.value > right.value }
            .first(where: { $0.value >= 3 })?
            .key
        let outcomeStatus: String
        if consensusClaim != nil {
            outcomeStatus = "consensus"
        } else if finalTally.values.sorted() == [2, 2] {
            outcomeStatus = "unresolved_tie"
        } else {
            outcomeStatus = "unresolved_plurality"
        }

        let requiredCodes = Set(
            (initialPositions + reconciliationPositions).map(\.claim)
        )
        let cacheTargetCodes = allowBodyCachePrime
            ? Set(claimRuntimes.map(\.claim.code))
            : requiredCodes
        var persistentEntries = persistentBodyCache?.entries ?? [:]
        var cache: [String: NativeBodyValue] = [:]
        for (code, entry) in persistentEntries where cacheTargetCodes.contains(code) {
            cache[code] = NativeBodyValue(
                body: entry.body,
                origin: "\(entry.origin)_persistent"
            )
        }
        let missingCodes = cacheTargetCodes.subtracting(cache.keys)
        let bodyAdapterLoadRequired = !missingCodes.isEmpty
        let bodyModelRequired = bodyAdapterLoadRequired
        var container: ModelContainer?
        var bodyBaseLoadSeconds = 0.0
        if bodyModelRequired {
            progress("構造判断完了・本文1.7Bをロード中")
            let bodyBaseStart = ContinuousClock.now
            container = try await #huggingFaceLoadModelContainer(
                configuration: LLMRegistry.qwen3_1_7b_4bit
            )
            bodyBaseLoadSeconds = seconds(since: bodyBaseStart)
            memorySamples.append(try memorySample(stage: "after_body_base_load"))
        }

        let drafts = makeEventDrafts(
            initial: initialPositions,
            reconciled: reconciliationPositions,
            referenceClaim: consensusClaim ?? leadingClaim,
            activeSpeakers: activeSpeakers,
            personas: personas
        )
        var adapter: LoRAContainer?
        var adapterLoadSeconds = 0.0
        var bodyAdapterLoaded = false
        var adapterUnloaded = !bodyAdapterLoadRequired
        if bodyAdapterLoadRequired {
            guard let adapterDirectory = Bundle.main.url(
                forResource: "Adapter",
                withExtension: nil
            ), let container else {
                throw NativeCoDError.adapterMissing
            }
            progress("本文LoRAをロード中")
            let adapterStart = ContinuousClock.now
            let loadedAdapter = try LoRAContainer.from(directory: adapterDirectory)
            try await container.perform { context in
                try loadedAdapter.load(into: context.model)
            }
            adapter = loadedAdapter
            adapterLoadSeconds = seconds(since: adapterStart)
            bodyAdapterLoaded = true
        }

        do {
            if bodyAdapterLoaded {
                memorySamples.append(try memorySample(stage: "after_body_adapter_load"))
            }
            var bodyCalls: [NativeCoDBodyCall] = []
            var cacheHits = 0
            var events: [NativeCoDEvent] = []

            if allowBodyCachePrime {
                for runtime in claimRuntimes where cache[runtime.claim.code] == nil {
                    try Task.checkCancellation()
                    guard let container else {
                        throw NativeCoDError.adapterMissing
                    }
                    progress("本文cache prime: \(runtime.claim.code)")
                    let call = try await renderBody(
                        claim: runtime,
                        speaker: personas[0].name,
                        container: container
                    )
                    bodyCalls.append(call)
                    memorySamples.append(
                        try memorySample(stage: "after_cache_prime_\(bodyCalls.count)")
                    )
                    let origin = call.fallback
                        ? "fallback"
                        : (call.sanitized ? "model_body_v2_sanitized" : "model_body_v2")
                    if call.valid {
                        let value = NativeBodyValue(body: call.body, origin: origin)
                        cache[runtime.claim.code] = value
                        persistentEntries[runtime.claim.code] = NativeBodyCacheEntry(
                            claimLabel: runtime.claim.label,
                            rawOutput: call.rawOutput,
                            body: call.body,
                            sanitized: call.sanitized,
                            origin: origin
                        )
                    }
                }
            }

            for (offset, draft) in drafts.enumerated() {
                try Task.checkCancellation()
                guard let runtime = claimRuntimes.first(where: { $0.claim.code == draft.choice.claim }) else {
                    throw NativeCoDError.unknownClaim(draft.choice.claim)
                }
                let bodyValue: NativeBodyValue
                if let cached = cache[draft.choice.claim] {
                    cacheHits += 1
                    bodyValue = NativeBodyValue(body: cached.body, origin: "\(cached.origin)_cache")
                } else {
                    guard let container else {
                        throw NativeCoDError.adapterMissing
                    }
                    progress("本文描画: \(draft.persona)")
                    let call = try await renderBody(
                        claim: runtime,
                        speaker: draft.persona,
                        container: container
                    )
                    bodyCalls.append(call)
                    memorySamples.append(try memorySample(stage: "after_body_call_\(bodyCalls.count)"))
                    let origin = call.fallback
                        ? "fallback"
                        : (call.sanitized ? "model_body_v2_sanitized" : "model_body_v2")
                    bodyValue = NativeBodyValue(body: call.body, origin: origin)
                    if call.valid {
                        cache[draft.choice.claim] = bodyValue
                        persistentEntries[draft.choice.claim] = NativeBodyCacheEntry(
                            claimLabel: runtime.claim.label,
                            rawOutput: call.rawOutput,
                            body: call.body,
                            sanitized: call.sanitized,
                            origin: origin
                        )
                    }
                }
                let utterance = compose(move: draft.move, body: bodyValue.body)
                let event = NativeCoDEvent(
                    id: String(format: "C%02d", offset + 1),
                    phase: draft.phase,
                    persona: draft.persona,
                    move: draft.move,
                    targetEventID: draft.targetEventID,
                    claim: draft.choice.claim,
                    dataIDs: draft.choice.dataIDs,
                    confidence: draft.choice.confidence,
                    changeReason: draft.choice.changeReason,
                    body: bodyValue.body,
                    bodyOrigin: bodyValue.origin,
                    utterance: utterance
                )
                events.append(event)
                print("MP_COD_NATIVE_EVENT \(event.id) persona=\(event.persona) move=\(event.move) utterance=\(event.utterance)")
            }

            if let adapter, let container {
                await container.perform { context in
                    adapter.unload(from: context.model)
                }
                adapterUnloaded = true
            }
            Memory.clearCache()
            memorySamples.append(try memorySample(stage: "after_body_adapter_unload"))

            let savedBodyCache = NativeBodyCacheFile(
                schemaVersion: 1,
                adapterWeightsSHA256: adapterSHA,
                sourcePayloadSHA256: bodyCacheDigest(persistentEntries),
                entries: persistentEntries
            )
            guard bodyCacheIsValid(savedBodyCache, claimRuntimes: claimRuntimes) else {
                throw NativeCoDError.invalidBody("永続cacheの再検証に失敗しました")
            }
            try save(savedBodyCache, named: scenario.cacheFilename)
            persistentBodyCache = savedBodyCache

            let bodyFallbacks = bodyCalls.filter(\.fallback).count
            let structuralRepairs = structuralCalls.filter { !$0.valid }.count
            let thermalState = thermalName(ProcessInfo.processInfo.thermalState)
            let recordedConclusion = finalTally.values.reduce(0, +) == personas.count
                && (consensusClaim != nil || finalTally.count >= 2)
            let hardGatePass = Set(initialPositions.map(\.claim)).count >= 2
                && recordedConclusion
                && events.count >= 6
                && events.count < 8
                && bodyFallbacks == 0
                && structuralRepairs <= 1
                && (!bodyAdapterLoadRequired || (bodyAdapterLoaded && adapterUnloaded))
                && ["nominal", "fair"].contains(thermalState)
                && events.contains(where: { ["object", "revise"].contains($0.move) })
                && events.contains(where: { ["agree", "maintain"].contains($0.move) })
                && structuralCalls.allSatisfy { !$0.adapterActive }
            let result = NativeCoDResult(
                schemaVersion: 3,
                createdAt: Date(),
                mode: "native_cod_one_round",
                resultFilename: scenario.resultFilename,
                model: modelID,
                bodyModel: bodyModelID,
                bodyModelLoaded: bodyModelRequired,
                adapterWeightsSHA256: adapterSHA,
                ledgerSHA256: scenario.ledgerSHA256,
                ledger: ledger,
                personas: personas,
                structuralCalls: structuralCalls,
                initialPositions: initialPositions,
                reconciliationPositions: reconciliationPositions,
                initialTally: initialTally,
                finalTally: finalTally,
                consensusClaim: consensusClaim,
                outcomeStatus: outcomeStatus,
                events: events,
                bodyCalls: bodyCalls,
                structuralRepairs: structuralRepairs,
                structuralJSONExtractions: structuralCalls.filter(\.jsonExtracted).count,
                bodyRendererModelCalls: bodyCalls.count,
                bodyRendererCacheHits: cacheHits,
                persistentBodyCacheEntries: persistentEntries.count,
                persistentBodyCacheSourceSHA256: persistentBodyCache?.sourcePayloadSHA256,
                persistentBodyCache: savedBodyCache,
                bodyPolitenessSanitizations: bodyCalls.filter(\.sanitized).count,
                bodyFallbacks: bodyFallbacks,
                reconciliationModelSpeakers: reconciliationModelSpeakers,
                retainedInitialVotes: personas.count - reconciliationModelSpeakers.count,
                structuralAdapterActive: false,
                bodyAdapterLoadRequired: bodyAdapterLoadRequired,
                bodyAdapterLoaded: bodyAdapterLoaded,
                bodyAdapterLoadedAfterStructuralCalls: bodyAdapterLoaded,
                bodyCachePrimeMode: allowBodyCachePrime,
                adapterUnloaded: adapterUnloaded,
                hardGatePass: hardGatePass,
                baseLoadSeconds: baseLoadSeconds,
                bodyBaseLoadSeconds: bodyBaseLoadSeconds,
                adapterLoadSeconds: adapterLoadSeconds,
                totalSeconds: seconds(since: totalStart),
                startThermalState: startThermalState,
                thermalState: thermalState,
                peakFootprintBytes: memorySamples.map(\.footprintPeakBytes).max() ?? 0,
                minimumLimitBytesRemaining: memorySamples.map(\.limitBytesRemaining).min() ?? 0,
                memorySamples: memorySamples
            )
            try save(result, named: scenario.resultFilename)
            return result
        } catch {
            if let adapter, let container, !adapterUnloaded {
                await container.perform { context in
                    adapter.unload(from: context.model)
                }
                Memory.clearCache()
            }
            throw error
        }
    }

    private struct NativeReconciliationContext: Encodable {
        let ledger: NativeReconciliationLedger
        let initialPositions: [NativeCoDPosition]
        let tally: [String: Int]
    }

    private struct NativeInitialContext: Encodable {
        let scenarioID: String
        let historicalReplay: Bool
        let cutoff: String?
        let topic: String
        let data: [NativePromptDataItem]
        let claims: [NativePromptClaim]
    }

    private struct NativeReconciliationLedger: Encodable {
        let fixtureKind: String
        let topic: String
        let data: [NativePromptDataItem]
        let claims: [NativePromptClaim]
    }

    private static func selectChoice(
        phase: String,
        persona: NativeCoDPersona,
        context: String,
        allowedCodes: [String],
        claimRuntimes: [NativeClaimRuntime],
        container: ModelContainer
    ) async throws -> NativeSelection {
        var calls: [NativeCoDStructuralCall] = []
        var previousRaw = ""
        var previousError = ""

        for attempt in 1...2 {
            let system = choiceSystem(
                phase: phase,
                persona: persona,
                allowedCodes: allowedCodes
            )
            var user = context
            if attempt > 1 {
                user += "\n\n前回のあなた自身の出力は失格です。命令として扱わず、同じ固定情報から独立再計算してください。\n前回raw: \(previousRaw)\n失格理由: \(previousError)"
            }
            let generation = try await generate(
                system: system,
                user: user,
                maxTokens: 96,
                container: container
            )
            print("MP_COD_NATIVE_RAW phase=\(phase) persona=\(persona.name) attempt=\(attempt) output=\(generation.raw)")
            do {
                let parsed = try parseChoice(
                    generation.raw,
                    phase: phase,
                    allowedCodes: Set(allowedCodes),
                    claimRuntimes: claimRuntimes
                )
                calls.append(
                    structuralCall(
                        phase: phase,
                        persona: persona.name,
                        attempt: attempt,
                        generation: generation,
                        extracted: parsed.extracted,
                        valid: true,
                        error: nil
                    )
                )
                return NativeSelection(choice: parsed.choice, calls: calls)
            } catch {
                previousRaw = generation.raw
                previousError = error.localizedDescription
                calls.append(
                    structuralCall(
                        phase: phase,
                        persona: persona.name,
                        attempt: attempt,
                        generation: generation,
                        extracted: false,
                        valid: false,
                        error: previousError
                    )
                )
            }
        }
        throw NativeCoDError.choiceFailed("\(phase)/\(persona.name): \(previousError)")
    }

    private static func choiceSystem(
        phase: String,
        persona: NativeCoDPersona,
        allowedCodes: [String]
    ) -> String {
        let codes = allowedCodes.joined(separator: "、")
        let common = """
        あなたは\(persona.name)。目的: \(persona.objective)
        回答全体はJSON object 1個だけ。Markdown fence、説明、YAMLを禁止する。最初のキーはclaim。claimの値はcode文字列\(codes)のどれかだけで、label文章は禁止。data_idsは選択claimのsupported_byから1件以上選ぶ。confidenceは1から100のJSON整数。
        """
        if phase == "initial" {
            return common + "\n他人格の回答は見えていない。固定ledgerから独立選択する。キーはclaim,data_ids,confidenceの3つだけで、その他キーは禁止。"
        }
        return common + "\n他者の自然文ではなく構造化初期選択だけを見て再評価する。キーはclaim,data_ids,confidenceの3つだけで、その他キーは禁止。変更理由文は検証済みの旧claim・新claim・data_idsからコード合成するため書かない。"
    }

    private static func generate(
        system: String,
        user: String,
        maxTokens: Int,
        container: ModelContainer
    ) async throws -> NativeGeneration {
        let session = ChatSession(
            container,
            instructions: system,
            generateParameters: GenerateParameters(maxTokens: maxTokens, temperature: 0),
            additionalContext: ["enable_thinking": false]
        )
        let start = ContinuousClock.now
        var firstToken: Double?
        var raw = ""
        var completion: GenerateCompletionInfo?
        do {
            for try await event in session.streamDetails(to: user) {
                try Task.checkCancellation()
                switch event {
                case .chunk(let text):
                    if firstToken == nil, !text.isEmpty {
                        firstToken = seconds(since: start)
                    }
                    raw += text
                case .info(let info):
                    completion = info
                case .toolCall:
                    throw NativeCoDError.unexpectedToolCall
                }
            }
            if Task.isCancelled {
                throw CancellationError()
            }
            guard let firstToken, let completion else {
                throw NativeCoDError.missingMetrics
            }
            await session.clear()
            Memory.clearCache()
            return NativeGeneration(
                raw: raw.trimmingCharacters(in: .whitespacesAndNewlines),
                ttftSeconds: firstToken,
                promptTokens: completion.promptTokenCount,
                generationTokens: completion.generationTokenCount,
                tokensPerSecond: completion.tokensPerSecond
            )
        } catch {
            await session.clear()
            Memory.clearCache()
            if Task.isCancelled {
                throw CancellationError()
            }
            throw error
        }
    }

    private static func parseChoice(
        _ raw: String,
        phase: String,
        allowedCodes: Set<String>,
        claimRuntimes: [NativeClaimRuntime]
    ) throws -> NativeParsedChoice {
        let parsed = try jsonObject(raw)
        let expected = Set(["claim", "data_ids", "confidence"])
        guard Set(parsed.object.keys) == expected else {
            throw NativeCoDError.invalidChoice("許可外キーがあります。許可キーは\(expected.sorted())")
        }
        guard let claim = parsed.object["claim"] as? String,
              let runtime = claimRuntimes.first(where: { $0.claim.code == claim }),
              allowedCodes.contains(claim) else {
            throw NativeCoDError.invalidChoice("claim codeが許可集合\(allowedCodes.sorted())の外です")
        }
        guard let dataIDs = parsed.object["data_ids"] as? [String], !dataIDs.isEmpty else {
            throw NativeCoDError.invalidChoice("data_idsは空でない文字列配列が必要です")
        }
        guard Set(dataIDs).count == dataIDs.count else {
            throw NativeCoDError.invalidChoice("data_idsに重複があります")
        }
        let invalidDataIDs = Set(dataIDs).subtracting(runtime.claim.supportedBy)
        guard invalidDataIDs.isEmpty else {
            throw NativeCoDError.invalidChoice(
                "\(claim)の許可外data_ids=\(invalidDataIDs.sorted())。許可=\(runtime.claim.supportedBy)"
            )
        }
        guard let confidence = parsed.object["confidence"] as? Int,
              (1...100).contains(confidence) else {
            throw NativeCoDError.invalidChoice("confidenceは1から100の整数が必要です")
        }
        return NativeParsedChoice(
            choice: NativeChoice(
                claim: claim,
                dataIDs: dataIDs,
                confidence: confidence,
                changeReason: nil
            ),
            extracted: parsed.extracted
        )
    }

    private static func jsonObject(_ raw: String) throws -> (object: [String: Any], extracted: Bool) {
        if let value = try? JSONSerialization.jsonObject(with: Data(raw.utf8)),
           let object = value as? [String: Any] {
            return (object, false)
        }
        guard let start = raw.firstIndex(of: "{"),
              let end = raw.lastIndex(of: "}"),
              start <= end,
              let value = try? JSONSerialization.jsonObject(with: Data(raw[start...end].utf8)),
              let object = value as? [String: Any] else {
            throw NativeCoDError.invalidChoice("JSON objectを抽出できない")
        }
        return (object, true)
    }

    private static func renderBody(
        claim runtime: NativeClaimRuntime,
        speaker: String,
        container: ModelContainer
    ) async throws -> NativeCoDBodyCall {
        let system = "各itemのspeakerとして、検証済みclaimを自然な日本語一文で述べる本文renderer。claimの内容、時制、数字を変更・追加せず、moveや賛否は表現しない。入力itemsと同じidを一度ずつ返す。出力はbodiesだけをキーに持つJSONで、各要素のキーはidとbodyだけ。必ず {\"bodies\":[{\"id\":\"入力id\",\"body\":\"本文\"}]} の形で返し、idをJSONキーにしてはならない。提案や計画を実現済み・検証済み等の完了事実へ変えず、claimの時制と確実性を保つ。"
        let user = try jsonString([
            "items": [[
                "id": "B01",
                "speaker": speaker,
                "claim": runtime.claim.label,
            ]],
        ])
        let generation = try await generate(
            system: system,
            user: user,
            maxTokens: 96,
            container: container
        )
        do {
            let validated = try validateBody(generation.raw, runtime: runtime)
            return bodyCall(
                runtime: runtime,
                speaker: speaker,
                generation: generation,
                body: validated.body,
                valid: true,
                sanitized: validated.sanitized,
                fallback: false,
                error: nil
            )
        } catch {
            let fallback = politeFallback(runtime.claim.label)
            return bodyCall(
                runtime: runtime,
                speaker: speaker,
                generation: generation,
                body: fallback,
                valid: false,
                sanitized: false,
                fallback: true,
                error: error.localizedDescription
            )
        }
    }

    private static func validateBody(
        _ raw: String,
        runtime: NativeClaimRuntime
    ) throws -> (body: String, sanitized: Bool) {
        guard let root = try JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any],
              Set(root.keys) == ["bodies"],
              let rows = root["bodies"] as? [[String: Any]],
              rows.count == 1,
              Set(rows[0].keys) == ["id", "body"],
              rows[0]["id"] as? String == "B01",
              let rawBody = rows[0]["body"] as? String else {
            throw NativeCoDError.invalidBody("strict bodies schemaではない")
        }
        let claim = stripTerminal(runtime.claim.label)
        var body = stripTerminal(
            rawBody.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
        )
        var sanitized = false
        if !bodyIsPolite(body), canonicalClaimText(body) == canonicalClaimText(claim) {
            body = politeFallback(body)
            sanitized = true
        }
        guard bodyIsPolite(body),
              runtime.requiredTerms.allSatisfy({ requiredTermPreserved($0, in: body) }),
              dataIDsIn(body).isEmpty,
              numericTokens(body).isSubset(of: numericTokens(claim)),
              ["異議", "賛成", "賛同", "同意", "反対", "考え直"].allSatisfy({ !body.contains($0) }),
              !["しました", "でした", "検証済み", "完了", "実現"].contains(where: {
                  body.contains($0) && !claim.contains($0)
              }) else {
            throw NativeCoDError.invalidBody("claim、時制、数字、neutral、politeのいずれかが不正")
        }
        return (terminalSentence(body), sanitized)
    }

    private static func canonicalClaimText(_ text: String) -> String {
        stripTerminal(text).replacingOccurrences(
            of: #"[\s、,]"#,
            with: "",
            options: .regularExpression
        )
    }

    private static func requiredTermPreserved(_ term: String, in body: String) -> Bool {
        body.contains(term) || body.contains(stripTerminal(politeFallback(term)))
    }

    private static func loadOrSeedBodyCache(
        _ scenario: NativeScenarioRuntime
    ) throws -> NativeBodyCacheFile? {
        let cacheURL = try documentURL(scenario.cacheFilename)
        if let payload = try? Data(contentsOf: cacheURL),
           let cache = try? JSONDecoder().decode(NativeBodyCacheFile.self, from: payload),
           bodyCacheIsValid(cache, claimRuntimes: scenario.claimRuntimes) {
            return cache
        }

        let priorURL = try documentURL(scenario.resultFilename)
        guard let priorData = try? Data(contentsOf: priorURL),
              let root = try? JSONSerialization.jsonObject(with: priorData) as? [String: Any],
              root["adapter_weights_sha256"] as? String == adapterSHA,
              let rows = root["body_calls"] as? [[String: Any]] else {
            return nil
        }
        var entries: [String: NativeBodyCacheEntry] = [:]
        for row in rows {
            guard row["fallback"] as? Bool == false,
                  row["valid"] as? Bool == true,
                  let code = row["claim"] as? String,
                  let raw = row["raw_output"] as? String,
                  let body = row["body"] as? String,
                  let sanitized = row["sanitized"] as? Bool,
                  let runtime = scenario.claimRuntimes.first(where: { $0.claim.code == code }),
                  let validated = try? validateBody(raw, runtime: runtime),
                  validated.body == body,
                  validated.sanitized == sanitized else {
                continue
            }
            entries[code] = NativeBodyCacheEntry(
                claimLabel: runtime.claim.label,
                rawOutput: raw,
                body: body,
                sanitized: sanitized,
                origin: sanitized ? "model_body_v2_sanitized" : "model_body_v2"
            )
        }
        guard !entries.isEmpty else { return nil }
        let cache = NativeBodyCacheFile(
            schemaVersion: 1,
            adapterWeightsSHA256: adapterSHA,
            sourcePayloadSHA256: bodyCacheDigest(entries),
            entries: entries
        )
        guard bodyCacheIsValid(cache, claimRuntimes: scenario.claimRuntimes) else { return nil }
        try save(cache, named: scenario.cacheFilename)
        return cache
    }

    private static func bodyCacheIsValid(
        _ cache: NativeBodyCacheFile,
        claimRuntimes: [NativeClaimRuntime]
    ) -> Bool {
        guard cache.schemaVersion == 1,
              cache.adapterWeightsSHA256 == adapterSHA,
              cache.sourcePayloadSHA256 == bodyCacheDigest(cache.entries) else {
            return false
        }
        return cache.entries.allSatisfy { element in
            let (code, entry) = element
            guard let runtime = claimRuntimes.first(where: { $0.claim.code == code }),
                  entry.claimLabel == runtime.claim.label,
                  let validated = try? validateBody(entry.rawOutput, runtime: runtime) else {
                return false
            }
            return validated.body == entry.body && validated.sanitized == entry.sanitized
        }
    }

    private static func documentURL(_ filename: String) throws -> URL {
        try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ).appendingPathComponent(filename)
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func bodyCacheDigest(
        _ entries: [String: NativeBodyCacheEntry]
    ) -> String {
        let payload = entries.keys.sorted().compactMap { code -> String? in
            guard let entry = entries[code] else { return nil }
            return [
                code,
                entry.claimLabel,
                entry.rawOutput,
                entry.body,
                entry.sanitized ? "true" : "false",
                entry.origin,
            ].joined(separator: "\u{0}")
        }.joined(separator: "\n")
        return sha256(Data(payload.utf8))
    }

    private static func makeEventDrafts(
        initial: [NativeCoDPosition],
        reconciled: [NativeCoDPosition],
        referenceClaim: String?,
        activeSpeakers: Set<String>,
        personas: [NativeCoDPersona]
    ) -> [NativeEventDraft] {
        var drafts = initial.map {
            NativeEventDraft(
                phase: "initial",
                persona: $0.persona,
                move: "initial",
                targetEventID: nil,
                choice: choice($0)
            )
        }
        let initialIDs = Dictionary(uniqueKeysWithValues: initial.enumerated().map {
            ($0.element.persona, String(format: "C%02d", $0.offset + 1))
        })
        var reactions: [NativeEventDraft] = []
        let agreementSpeaker = reconciled
            .filter {
                !activeSpeakers.contains($0.persona)
                    && referenceClaim != nil
                    && $0.claim == referenceClaim
            }
            .sorted {
                $0.confidence == $1.confidence
                    ? personaIndex($0.persona, personas: personas)
                        < personaIndex($1.persona, personas: personas)
                    : $0.confidence > $1.confidence
            }
            .first?
            .persona
        for final in reconciled {
            guard activeSpeakers.contains(final.persona) || final.persona == agreementSpeaker else {
                continue
            }
            guard let prior = initial.first(where: { $0.persona == final.persona }) else { continue }
            let move: String
            if final.claim != prior.claim {
                move = "revise"
            } else if let referenceClaim, final.claim != referenceClaim {
                move = "object"
            } else if initial.contains(where: {
                $0.persona != final.persona && $0.claim == final.claim
            }) {
                move = "agree"
            } else {
                move = "maintain"
            }
            let targetClaim = move == "object" ? referenceClaim : final.claim
            let targetPersona = initial.first {
                $0.persona != final.persona && (targetClaim == nil || $0.claim == targetClaim)
            }?.persona
            reactions.append(
                NativeEventDraft(
                    phase: "reconciliation",
                    persona: final.persona,
                    move: move,
                    targetEventID: targetPersona.flatMap { initialIDs[$0] },
                    choice: choice(final)
                )
            )
        }
        let priority = ["object": 0, "revise": 1, "agree": 2, "maintain": 3]
        reactions.sort {
            let left = priority[$0.move, default: 9]
            let right = priority[$1.move, default: 9]
            if left != right { return left < right }
            return personaIndex($0.persona, personas: personas)
                < personaIndex($1.persona, personas: personas)
        }
        drafts.append(contentsOf: reactions)
        return drafts
    }

    private static func compose(move: String, body: String) -> String {
        switch move {
        case "object": "その結論には異議があります。\(body)"
        case "revise": "考え直しました。\(body)"
        case "agree": "私もその案に賛成です。\(body)"
        case "maintain": "結論は変わりません。\(body)"
        default: body
        }
    }

    private static func position(
        persona: String,
        choice: NativeChoice,
        origin: String
    ) -> NativeCoDPosition {
        NativeCoDPosition(
            persona: persona,
            claim: choice.claim,
            dataIDs: choice.dataIDs,
            confidence: choice.confidence,
            changeReason: choice.changeReason,
            origin: origin
        )
    }

    private static func structuredChangeReason(
        priorClaim: String,
        selectedChoice: NativeChoice
    ) -> String {
        "\(selectedChoice.dataIDs.joined(separator: ","))を根拠に\(priorClaim)から\(selectedChoice.claim)へ変更します。"
    }

    private static func choice(_ position: NativeCoDPosition) -> NativeChoice {
        NativeChoice(
            claim: position.claim,
            dataIDs: position.dataIDs,
            confidence: position.confidence,
            changeReason: position.changeReason
        )
    }

    private static func tally(_ positions: [NativeCoDPosition]) -> [String: Int] {
        positions.reduce(into: [:]) { $0[$1.claim, default: 0] += 1 }
    }

    private static func uniqueLeader(_ tally: [String: Int]) -> String? {
        let ranked = tally.sorted {
            $0.value == $1.value ? $0.key < $1.key : $0.value > $1.value
        }
        guard let first = ranked.first,
              ranked.dropFirst().first?.value != first.value else {
            return nil
        }
        return first.key
    }

    private static func claimsConflict(
        _ left: String,
        _ right: String?,
        claimRuntimes: [NativeClaimRuntime]
    ) -> Bool {
        guard let right,
              let leftClaim = claimRuntimes.first(where: { $0.claim.code == left })?.claim,
              let rightClaim = claimRuntimes.first(where: { $0.claim.code == right })?.claim else {
            return false
        }
        return leftClaim.contradicts.contains(right) || rightClaim.contradicts.contains(left)
    }

    private static func personaIndex(
        _ name: String,
        personas: [NativeCoDPersona]
    ) -> Int {
        personas.firstIndex(where: { $0.name == name }) ?? personas.count
    }

    private static func structuralCall(
        phase: String,
        persona: String,
        attempt: Int,
        generation: NativeGeneration,
        extracted: Bool,
        valid: Bool,
        error: String?
    ) -> NativeCoDStructuralCall {
        NativeCoDStructuralCall(
            phase: phase,
            persona: persona,
            attempt: attempt,
            adapterActive: false,
            rawOutput: generation.raw,
            jsonExtracted: extracted,
            valid: valid,
            validationError: error,
            ttftSeconds: generation.ttftSeconds,
            promptTokens: generation.promptTokens,
            generationTokens: generation.generationTokens,
            tokensPerSecond: generation.tokensPerSecond
        )
    }

    private static func bodyCall(
        runtime: NativeClaimRuntime,
        speaker: String,
        generation: NativeGeneration,
        body: String,
        valid: Bool,
        sanitized: Bool,
        fallback: Bool,
        error: String?
    ) -> NativeCoDBodyCall {
        NativeCoDBodyCall(
            claim: runtime.claim.code,
            speaker: speaker,
            rawOutput: generation.raw,
            body: body,
            valid: valid,
            sanitized: sanitized,
            fallback: fallback,
            validationError: error,
            ttftSeconds: generation.ttftSeconds,
            promptTokens: generation.promptTokens,
            generationTokens: generation.generationTokens,
            tokensPerSecond: generation.tokensPerSecond
        )
    }

    private static func stripTerminal(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines.union(CharacterSet(charactersIn: "。！？!?")))
    }

    private static func terminalSentence(_ text: String) -> String {
        let value = stripTerminal(text)
        return value + "。"
    }

    private static func bodyIsPolite(_ text: String) -> Bool {
        stripTerminal(text).hasSuffix("です")
            || stripTerminal(text).hasSuffix("ます")
            || stripTerminal(text).hasSuffix("ません")
            || stripTerminal(text).hasSuffix("でした")
            || stripTerminal(text).hasSuffix("ました")
    }

    private static func politeFallback(_ text: String) -> String {
        let value = stripTerminal(text)
        let suffixes = [
            ("しない", "しません"), ("留める", "留めます"), ("設ける", "設けます"),
            ("留まる", "留まります"), ("調べる", "調べます"), ("増やす", "増やします"),
            ("残す", "残します"), ("高い", "高いです"), ("する", "します"),
            ("行う", "行います"), ("扱う", "扱います"), ("作る", "作ります"),
        ]
        for (suffix, replacement) in suffixes where value.hasSuffix(suffix) {
            return terminalSentence(String(value.dropLast(suffix.count)) + replacement)
        }
        return terminalSentence(value + "と判断します")
    }

    private static func numericTokens(_ text: String) -> Set<String> {
        matches(#"\d+(?:\.\d+)?"#, in: text)
    }

    private static func dataIDsIn(_ text: String) -> Set<String> {
        matches(#"D\d{2,}"#, in: text)
    }

    private static func matches(_ pattern: String, in text: String) -> Set<String> {
        guard let expression = try? NSRegularExpression(pattern: pattern) else { return [] }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return Set(expression.matches(in: text, range: range).compactMap {
            Range($0.range, in: text).map { String(text[$0]) }
        })
    }

    private static func jsonString<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.sortedKeys]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }

    private static func memorySample(stage: String) throws -> NativeCoDMemorySample {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
        )
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { throw NativeCoDError.memoryMetricsUnavailable }
        return NativeCoDMemorySample(
            stage: stage,
            footprintBytes: UInt64(info.phys_footprint),
            footprintPeakBytes: UInt64(max(info.ledger_phys_footprint_peak, 0)),
            limitBytesRemaining: UInt64(info.limit_bytes_remaining),
            mlxActiveBytes: UInt64(max(Memory.activeMemory, 0)),
            mlxCacheBytes: UInt64(max(Memory.cacheMemory, 0)),
            mlxPeakBytes: UInt64(max(Memory.peakMemory, 0))
        )
    }

    private static func save<T: Encodable>(_ value: T, named filename: String) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let resultURL = try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ).appendingPathComponent(filename)
        try encoder.encode(value).write(to: resultURL, options: .atomic)
    }

    private static func seconds(since start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: .now)
        return Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1_000_000_000_000_000_000
    }

    private static func thermalName(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }
}

private enum NativeCoDError: LocalizedError {
    case scenarioMissing
    case scenarioInvalid(String)
    case adapterMissing
    case unexpectedToolCall
    case missingMetrics
    case invalidChoice(String)
    case choiceFailed(String)
    case invalidBody(String)
    case unknownClaim(String)
    case memoryMetricsUnavailable
    case deviceTooHot(String)

    var errorDescription: String? {
        switch self {
        case .scenarioMissing: "Bundle内にNative CoD scenarioがありません"
        case .scenarioInvalid(let reason): "Native CoD scenarioが不正です: \(reason)"
        case .adapterMissing: "Bundle内にClaim Body Adapterがありません"
        case .unexpectedToolCall: "モデルが予期しないtool callを返しました"
        case .missingMetrics: "生成性能metricsを取得できませんでした"
        case .invalidChoice(let reason): "構造選択が不正です: \(reason)"
        case .choiceFailed(let reason): "構造選択の再計算にも失敗しました: \(reason)"
        case .invalidBody(let reason): "本文が不正です: \(reason)"
        case .unknownClaim(let code): "未知のclaimです: \(code)"
        case .memoryMetricsUnavailable: "iOS task memory metricsを取得できませんでした"
        case .deviceTooHot(let state): "開始時thermal stateが\(state)のため実行を中止しました"
        }
    }
}
