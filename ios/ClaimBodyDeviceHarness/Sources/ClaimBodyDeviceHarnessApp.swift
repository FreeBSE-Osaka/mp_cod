import Darwin
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import SwiftUI
import Tokenizers
import UIKit

@main
struct ClaimBodyDeviceHarnessApp: App {
    var body: some Scene {
        WindowGroup {
            if ProcessInfo.processInfo.arguments.contains("--3d-shadow") {
                CoD3DShadowView()
            } else {
                ClaimBodyDeviceHarnessView()
            }
        }
    }
}

private struct SmokeItem {
    let persona: String
    let claim: String
    let requiredTerms: [String]
}

private let smokeItems = [
    SmokeItem(
        persona: "力学モデル研究者",
        claim: "進路予測は上層場と地上場の整合を確認して更新する",
        requiredTerms: ["進路予測", "上層場", "地上場", "更新"]
    ),
    SmokeItem(
        persona: "アンサンブル確率予報者",
        claim: "少数だが重大なシナリオも分布に残して比較する",
        requiredTerms: ["少数", "重大", "シナリオ", "比較"]
    ),
    SmokeItem(
        persona: "観測・ナウキャスト専門家",
        claim: "根拠データの観測時刻と出典を毎回確認する",
        requiredTerms: ["観測時刻", "出典", "確認"]
    ),
    SmokeItem(
        persona: "影響・リスク予報者",
        claim: "暴風が強まる前の安全確保を優先する",
        requiredTerms: ["暴風", "安全確保", "優先"]
    ),
]

private struct UtteranceResult: Encodable {
    let sequence: Int
    let persona: String
    let claim: String
    let body: String
    let rawOutput: String
    let ttftSeconds: Double
    let promptTokens: Int
    let generationTokens: Int
    let tokensPerSecond: Double
    let generationSeconds: Double
}

private struct MemorySample: Encodable {
    let stage: String
    let footprintBytes: UInt64
    let footprintPeakBytes: UInt64
    let residentBytes: UInt64
    let residentPeakBytes: UInt64
    let limitBytesRemaining: UInt64
    let mlxActiveBytes: UInt64
    let mlxCacheBytes: UInt64
    let mlxPeakBytes: UInt64
}

private struct SmokeResult: Encodable {
    let schemaVersion: Int
    let createdAt: Date
    let mode: String
    let model: String
    let adapterWeightsSHA256: String
    let baseLoadSeconds: Double
    let adapterLoadSeconds: Double
    let totalPromptTokens: Int
    let totalGenerationTokens: Int
    let aggregateTokensPerSecond: Double
    let endToEndTokensPerSecond: Double
    let totalSeconds: Double
    let thermalState: String
    let adapterUnloaded: Bool
    let outputsDistinct: Bool
    let peakFootprintBytes: UInt64
    let minimumLimitBytesRemaining: UInt64
    let memorySamples: [MemorySample]
    let utterances: [UtteranceResult]
    let distributedRequestID: String?
    let distributedJobIndices: [Int]?
    let modelWeightsSHA256: String?
}

private struct DistributedFailure: Encodable {
    let schemaVersion = 1
    let mode = "distributed_body_worker"
    let status: String
    let distributedRequestID: String
    let distributedJobIndices: [Int]
    let error: String
    let availableDiskBytes: UInt64?
}

private struct CancellationResult: Encodable {
    let schemaVersion: Int
    let createdAt: Date
    let model: String
    let completedUtterances: Int
    let adapterUnloaded: Bool
    let thermalState: String
    let memoryAfterUnload: MemorySample?
}

private struct ClaimBodyDeviceHarnessView: View {
    @Environment(\.scenePhase) private var scenePhase
    @State private var status = "実行待ち"
    @State private var log = "Baseは端末へ初回downloadします。"
    @State private var downloadProgress = 0.0
    @State private var isRunning = false
    @State private var didAutoRun = false
    @State private var smokeTask: Task<Void, Never>?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Qwen3 0.6B Base + 1.7B Claim Body v3")
                        .font(.headline)
                    Text("iPhone実機だけで構造化CoD、本文LoRA、memory、thermal、unloadを確認します。")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    if isRunning {
                        ProgressView(value: downloadProgress)
                    }

                    LabeledContent("状態", value: status)
                    Text(log)
                        .font(.caption.monospaced())
                        .textSelection(.enabled)

                    Button {
                        startNativeCoD(.typhoon18Replay)
                    } label: {
                        Label("台風18号Historical Replay", systemImage: "hurricane")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isRunning)
                    .accessibilityIdentifier("runA15TyphoonReplay")

                    Button {
                        startNativeCoD(.syntheticBalanced)
                    } label: {
                        Label("架空Fixture Native CoD", systemImage: "person.3.sequence")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .disabled(isRunning)
                    .accessibilityIdentifier("runA15NativeCoD")

                    Button {
                        startSmoke()
                    } label: {
                        Label("本文Renderer Soakを実行", systemImage: "cpu")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .disabled(isRunning)
                    .accessibilityIdentifier("runA15Smoke")

                    if isRunning {
                        Button(role: .destructive) {
                            smokeTask?.cancel()
                        } label: {
                            Label("生成をキャンセル", systemImage: "stop.circle")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .accessibilityIdentifier("cancelA15Smoke")
                    }
                }
                .padding()
            }
            .navigationTitle("MP-CoD Weight Smoke")
            .task {
                guard !didAutoRun,
                      ProcessInfo.processInfo.arguments.contains("--autorun") else { return }
                didAutoRun = true
                if ProcessInfo.processInfo.arguments.contains("--native-cod") {
                    startNativeCoD(
                        ProcessInfo.processInfo.arguments.contains("--weather-replay")
                            ? .typhoon18Replay
                            : .syntheticBalanced,
                        allowBodyCachePrime: ProcessInfo.processInfo.arguments.contains(
                            "--prime-body-cache"
                        )
                    )
                } else {
                    startSmoke()
                }
            }
            .onChange(of: scenePhase) { _, phase in
                if phase != .active, ProcessInfo.processInfo.arguments.contains("--distributed-worker") {
                    smokeTask?.cancel()
                }
            }
            .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in
                if ProcessInfo.processInfo.arguments.contains("--distributed-worker") {
                    smokeTask?.cancel()
                }
            }
        }
    }

    @MainActor
    private func startSmoke() {
        guard smokeTask == nil else { return }
        smokeTask = Task { @MainActor in
            await runSmoke()
            smokeTask = nil
        }
    }

    @MainActor
    private func startNativeCoD(
        _ scenario: NativeCoDScenarioKind,
        allowBodyCachePrime: Bool = false
    ) {
        guard smokeTask == nil else { return }
        smokeTask = Task { @MainActor in
            await runNativeCoD(
                scenario,
                allowBodyCachePrime: allowBodyCachePrime
            )
            smokeTask = nil
        }
    }

    @MainActor
    private func runNativeCoD(
        _ scenario: NativeCoDScenarioKind,
        allowBodyCachePrime: Bool
    ) async {
        guard !isRunning else { return }
        isRunning = true
        downloadProgress = 0
        status = "Native CoDを準備中"
        log = "structure_model=mlx-community/Qwen3-0.6B-4bit\nbody_model=mlx-community/Qwen3-1.7B-4bit\n"
        defer { isRunning = false }

        do {
            let result = try await NativeCoDSmokeRunner.run(
                scenario: scenario,
                allowBodyCachePrime: allowBodyCachePrime
            ) { phase in
                status = phase
            }
            let eventLog = result.events.map {
                "[\($0.id) \($0.move) claim=\($0.claim) data=\($0.dataIDs.joined(separator: ","))]\n\($0.persona): \($0.utterance)"
            }.joined(separator: "\n\n")
            log += eventLog
            log += "\n\ninitial_tally=\(result.initialTally)\n"
            log += "final_tally=\(result.finalTally)\n"
            log += "consensus=\(result.consensusClaim ?? "unresolved")\n"
            log += "outcome_status=\(result.outcomeStatus)\n"
            log += "structural_repairs=\(result.structuralRepairs)\n"
            log += "body_calls=\(result.bodyRendererModelCalls) cache_hits=\(result.bodyRendererCacheHits)\n"
            log += "body_sanitizations=\(result.bodyPolitenessSanitizations) fallbacks=\(result.bodyFallbacks)\n"
            log += "peak_footprint_mib=\(format(mebibytes(result.peakFootprintBytes)))\n"
            log += "minimum_limit_remaining_mib=\(format(mebibytes(result.minimumLimitBytesRemaining)))\n"
            log += "total_seconds=\(format(result.totalSeconds)) thermal_state=\(result.thermalState)\n"
            log += "result_file=\(result.resultFilename)\n"
            status = result.hardGatePass ? "CoD PASS" : "CoD HOLD"
            downloadProgress = 1
            print("MP_COD_A15_NATIVE_COD \(result.hardGatePass ? "PASS" : "HOLD")\n\(log)")
        } catch is CancellationError {
            status = "CANCELLED"
            log += "cancelled=true\n"
            print("MP_COD_A15_NATIVE_COD CANCELLED\n\(log)")
        } catch {
            status = "FAIL"
            log += "error=\(error.localizedDescription)\n"
            print("MP_COD_A15_NATIVE_COD FAIL \(error)")
        }
    }

    @MainActor
    private func runSmoke() async {
        guard !isRunning else { return }
        isRunning = true
        downloadProgress = 0
        status = "Baseを取得・ロード中"
        log = "model=mlx-community/Qwen3-1.7B-4bit\n"
        let totalStart = ContinuousClock.now
        var distributedRequest: DistributedBodyRequest?

        defer { isRunning = false }
        do {
            distributedRequest = try DistributedBodyRequest.parse(ProcessInfo.processInfo.arguments)
            let selectedItems = distributedRequest.map { request in
                request.jobIndices.map { smokeItems[$0] }
            } ?? smokeItems
            var configuration = LLMRegistry.qwen3_1_7b_4bit
            var modelWeightsSHA256: String?
            if let request = distributedRequest {
                try checkDistributedResources()
                let documents = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask,
                                                           appropriateFor: nil, create: true)
                guard !FileManager.default.fileExists(atPath: documents.appendingPathComponent(request.resultFilename).path) else {
                    throw DistributedBodyRequest.RequestError.existingResult
                }
                let modelDirectory = documents.appendingPathComponent("DistributedBodyModel", isDirectory: true)
                let weights = modelDirectory.appendingPathComponent("model.safetensors")
                guard FileManager.default.fileExists(atPath: weights.path) else {
                    throw DistributedBodyRequest.RequestError.localModelMissing
                }
                modelWeightsSHA256 = try await Task.detached(priority: .utility) {
                    try DistributedBodyRequest.fileSHA256(weights)
                }.value
                try Task.checkCancellation()
                try checkDistributedResources()
                configuration.id = .directory(modelDirectory)
                log += "distributed_request_id=\(request.requestID) jobs=\(request.jobIndices)\n"
            }
            Memory.peakMemory = 0
            var memorySamples = [try memorySample(stage: "start")]
            let container = try await #huggingFaceLoadModelContainer(
                configuration: configuration,
                progressHandler: { progress in
                    Task { @MainActor in
                        downloadProgress = progress.fractionCompleted
                        status = "Base取得 \(Int(progress.fractionCompleted * 100))%"
                    }
                }
            )
            let loadSeconds = seconds(since: totalStart)
            log += "base_load_seconds=\(format(loadSeconds))\n"
            memorySamples.append(try memorySample(stage: "after_base_load"))

            guard let adapterDirectory = Bundle.main.url(
                forResource: "Adapter",
                withExtension: nil
            ) else {
                throw HarnessError.adapterMissing
            }
            if distributedRequest != nil {
                guard try DistributedBodyRequest.fileSHA256(adapterDirectory.appendingPathComponent("adapters.safetensors"))
                        == "4ce21e64af220f0ee309599e189fd136e10c4c5cd11440c3d60fd306749a9a92" else {
                    throw HarnessError.invalidContract("Adapter SHA mismatch")
                }
            }
            let adapterStart = ContinuousClock.now
            let adapter = try LoRAContainer.from(directory: adapterDirectory)
            try await container.perform { context in
                try adapter.load(into: context.model)
            }
            let adapterSeconds = seconds(since: adapterStart)
            log += "adapter_load_seconds=\(format(adapterSeconds))\n"

            var adapterUnloaded = false
            var utterances: [UtteranceResult] = []
            do {
                memorySamples.append(try memorySample(stage: "after_adapter_load"))
                if ProcessInfo.processInfo.arguments.contains("--cancel-during-generation") {
                    Task { @MainActor in
                        try? await ContinuousClock().sleep(for: .milliseconds(250))
                        smokeTask?.cancel()
                    }
                }

                for (offset, item) in selectedItems.enumerated() {
                    try Task.checkCancellation()
                    if distributedRequest != nil { try checkDistributedResources() }
                    status = "\(offset + 1)/\(selectedItems.count) \(item.persona)"
                    let utterance = try await renderBody(
                        item,
                        sequence: offset + 1,
                        container: container
                    )
                    utterances.append(utterance)
                    memorySamples.append(try memorySample(stage: "after_utterance_\(offset + 1)"))
                    log += "\(item.persona): \(utterance.body)\n"
                    log += "  ttft=\(format(utterance.ttftSeconds))s speed=\(format(utterance.tokensPerSecond))tok/s\n"
                }

                guard Set(utterances.map(\.body)).count == utterances.count else {
                    throw HarnessError.duplicateBodies
                }

                await container.perform { context in
                    adapter.unload(from: context.model)
                }
                adapterUnloaded = true
                Memory.clearCache()
                memorySamples.append(try memorySample(stage: "after_adapter_unload"))

                let totalSeconds = seconds(since: totalStart)
                let thermalState = thermalName(ProcessInfo.processInfo.thermalState)
                let totalPromptTokens = utterances.reduce(0) { $0 + $1.promptTokens }
                let totalGenerationTokens = utterances.reduce(0) { $0 + $1.generationTokens }
                let generationSeconds = utterances.reduce(0.0) { $0 + $1.generationSeconds }
                let decodeSeconds = utterances.reduce(0.0) {
                    $0 + Double($1.generationTokens) / max($1.tokensPerSecond, 0.001)
                }
                let aggregateTokensPerSecond = Double(totalGenerationTokens) / max(decodeSeconds, 0.001)
                let endToEndTokensPerSecond = Double(totalGenerationTokens) / max(generationSeconds, 0.001)
                log += "peak_footprint_mib=\(format(mebibytes(memorySamples.map(\.footprintPeakBytes).max() ?? 0)))\n"
                log += "minimum_limit_remaining_mib=\(format(mebibytes(memorySamples.map(\.limitBytesRemaining).min() ?? 0)))\n"
                log += "total_seconds=\(format(totalSeconds))\n"
                log += "thermal_state=\(thermalState)\n"

                let result = SmokeResult(
                    schemaVersion: distributedRequest == nil ? 2 : 3,
                    createdAt: Date(),
                    mode: distributedRequest == nil ? "four_persona_body_soak" : "distributed_body_worker",
                    model: "mlx-community/Qwen3-1.7B-4bit",
                    adapterWeightsSHA256: "4ce21e64af220f0ee309599e189fd136e10c4c5cd11440c3d60fd306749a9a92",
                    baseLoadSeconds: loadSeconds,
                    adapterLoadSeconds: adapterSeconds,
                    totalPromptTokens: totalPromptTokens,
                    totalGenerationTokens: totalGenerationTokens,
                    aggregateTokensPerSecond: aggregateTokensPerSecond,
                    endToEndTokensPerSecond: endToEndTokensPerSecond,
                    totalSeconds: totalSeconds,
                    thermalState: thermalState,
                    adapterUnloaded: adapterUnloaded,
                    outputsDistinct: true,
                    peakFootprintBytes: memorySamples.map(\.footprintPeakBytes).max() ?? 0,
                    minimumLimitBytesRemaining: memorySamples.map(\.limitBytesRemaining).min() ?? 0,
                    memorySamples: memorySamples,
                    utterances: utterances,
                    distributedRequestID: distributedRequest?.requestID,
                    distributedJobIndices: distributedRequest?.jobIndices,
                    modelWeightsSHA256: modelWeightsSHA256
                )
                let resultFilename = distributedRequest?.resultFilename ?? "mp_cod_a15_soak.json"
                try save(result, named: resultFilename)
                log += "result_file=\(resultFilename)\n"
                status = "PASS"
                downloadProgress = 1
                print("MP_COD_A15_SOAK PASS\n\(log)")
            } catch {
                if !adapterUnloaded {
                    await container.perform { context in
                        adapter.unload(from: context.model)
                    }
                    adapterUnloaded = true
                    Memory.clearCache()
                }
                let afterUnload = try? memorySample(stage: "cancel_after_adapter_unload")
                if error is CancellationError {
                    let result = CancellationResult(
                        schemaVersion: 1,
                        createdAt: Date(),
                        model: "mlx-community/Qwen3-1.7B-4bit",
                        completedUtterances: utterances.count,
                        adapterUnloaded: adapterUnloaded,
                        thermalState: thermalName(ProcessInfo.processInfo.thermalState),
                        memoryAfterUnload: afterUnload
                    )
                    do {
                        try save(result, named: "mp_cod_a15_cancel.json")
                        log += "result_file=mp_cod_a15_cancel.json\n"
                    } catch {
                        log += "cancel_result_error=\(error.localizedDescription)\n"
                    }
                }
                throw error
            }
        } catch is CancellationError {
            status = "CANCELLED"
            log += "cancelled=true\n"
            print("MP_COD_A15_SOAK CANCELLED\n\(log)")
            if let request = distributedRequest { saveDistributedFailure(request, status: "cancelled", error: "cancelled") }
        } catch {
            status = "FAIL"
            log += "error=\(error.localizedDescription)\n"
            print("MP_COD_A15_SOAK FAIL \(error)")
            if let request = distributedRequest {
                saveDistributedFailure(request, status: "failed", error: error.localizedDescription)
            }
        }
    }

    private func checkDistributedResources() throws {
        guard ProcessInfo.processInfo.thermalState == .nominal || ProcessInfo.processInfo.thermalState == .fair,
              try memorySample(stage: "distributed_guard").limitBytesRemaining >= 512 * 1_048_576 else {
            throw DistributedBodyRequest.RequestError.unsafeResources
        }
    }

    private func saveDistributedFailure(_ request: DistributedBodyRequest, status: String, error: String) {
        do {
            let attributes = try? FileManager.default.attributesOfFileSystem(forPath: NSHomeDirectory())
            let available = (attributes?[.systemFreeSize] as? NSNumber)?.uint64Value
            try save(DistributedFailure(status: status, distributedRequestID: request.requestID,
                                        distributedJobIndices: request.jobIndices, error: error,
                                        availableDiskBytes: available),
                     named: request.resultFilename)
        } catch { print("MP_COD_DISTRIBUTED failure_record_error=\(error)") }
    }

    @MainActor
    private func renderBody(
        _ item: SmokeItem,
        sequence: Int,
        container: ModelContainer
    ) async throws -> UtteranceResult {
        let system = "各itemのspeakerとして、検証済みclaimを自然な日本語一文で述べる本文renderer。claimの内容、時制、数字を変更・追加せず、moveや賛否は表現しない。入力itemsと同じidを一度ずつ返す。出力はbodiesだけをキーに持つJSONで、各要素のキーはidとbodyだけ。必ず {\"bodies\":[{\"id\":\"入力id\",\"body\":\"本文\"}]} の形で返し、idをJSONキーにしてはならない。提案や計画を実現済み・検証済み等の完了事実へ変えず、claimの時制と確実性を保つ。"
        let promptData = try JSONSerialization.data(
            withJSONObject: [
                "items": [[
                    "id": "B01",
                    "speaker": item.persona,
                    "claim": item.claim,
                ]],
            ],
            options: [.sortedKeys]
        )
        let session = ChatSession(
            container,
            instructions: system,
            generateParameters: GenerateParameters(maxTokens: 96, temperature: 0),
            additionalContext: ["enable_thinking": false]
        )
        let generationStart = ContinuousClock.now
        var firstTokenSeconds: Double?
        var output = ""
        var completion: GenerateCompletionInfo?

        do {
            for try await event in session.streamDetails(
                to: String(decoding: promptData, as: UTF8.self)
            ) {
                try Task.checkCancellation()
                switch event {
                case .chunk(let text):
                    if firstTokenSeconds == nil, !text.isEmpty {
                        firstTokenSeconds = seconds(since: generationStart)
                    }
                    output += text
                case .info(let info):
                    completion = info
                case .toolCall:
                    throw HarnessError.unexpectedToolCall
                }
            }
            if Task.isCancelled {
                throw CancellationError()
            }
            guard let completion, let firstTokenSeconds else {
                throw HarnessError.missingMetrics
            }
            print("MP_COD_A15_RAW persona=\(item.persona) output=\(output)")
            let body = try validatedBody(from: output, item: item)
            let result = UtteranceResult(
                sequence: sequence,
                persona: item.persona,
                claim: item.claim,
                body: body,
                rawOutput: output,
                ttftSeconds: firstTokenSeconds,
                promptTokens: completion.promptTokenCount,
                generationTokens: completion.generationTokenCount,
                tokensPerSecond: completion.tokensPerSecond,
                generationSeconds: seconds(since: generationStart)
            )
            await session.clear()
            Memory.clearCache()
            return result
        } catch {
            await session.clear()
            Memory.clearCache()
            if Task.isCancelled {
                throw CancellationError()
            }
            throw error
        }
    }

    private func validatedBody(from output: String, item: SmokeItem) throws -> String {
        guard let root = try JSONSerialization.jsonObject(with: Data(output.utf8)) as? [String: Any],
              Set(root.keys) == ["bodies"],
              let bodies = root["bodies"] as? [[String: Any]],
              bodies.count == 1,
              Set(bodies[0].keys) == ["id", "body"],
              bodies[0]["id"] as? String == "B01",
              let body = bodies[0]["body"] as? String,
              !body.isEmpty,
              item.requiredTerms.allSatisfy(body.contains),
              body.range(of: #"D\d{2,}"#, options: .regularExpression) == nil,
              body.range(
                of: #"(です|ます|ません|でしょう|ましょう|ください)[。！？]?$"#,
                options: .regularExpression
              ) != nil,
              ["異議", "賛成", "賛同", "同意", "反対", "考え直"].allSatisfy({ !body.contains($0) }) else {
            throw HarnessError.invalidContract("\(item.persona): \(output)")
        }
        return body
    }

    private func memorySample(stage: String) throws -> MemorySample {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
        )
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else {
            throw HarnessError.memoryMetricsUnavailable
        }
        return MemorySample(
            stage: stage,
            footprintBytes: UInt64(info.phys_footprint),
            footprintPeakBytes: UInt64(max(info.ledger_phys_footprint_peak, 0)),
            residentBytes: UInt64(info.resident_size),
            residentPeakBytes: UInt64(info.resident_size_peak),
            limitBytesRemaining: UInt64(info.limit_bytes_remaining),
            mlxActiveBytes: UInt64(max(Memory.activeMemory, 0)),
            mlxCacheBytes: UInt64(max(Memory.cacheMemory, 0)),
            mlxPeakBytes: UInt64(max(Memory.peakMemory, 0))
        )
    }

    private func save<T: Encodable>(_ value: T, named filename: String) throws {
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
        if filename.hasPrefix("mp_cod_distributed_"), FileManager.default.fileExists(atPath: resultURL.path) {
            throw DistributedBodyRequest.RequestError.existingResult
        }
        try encoder.encode(value).write(to: resultURL, options: .atomic)
    }

    private func seconds(since start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: .now)
        return Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1_000_000_000_000_000_000
    }

    private func format(_ value: Double) -> String {
        value.formatted(.number.precision(.fractionLength(3)))
    }

    private func mebibytes(_ bytes: UInt64) -> Double {
        Double(bytes) / 1_048_576
    }

    private func thermalName(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }
}

private enum HarnessError: LocalizedError {
    case adapterMissing
    case unexpectedToolCall
    case invalidContract(String)
    case missingMetrics
    case duplicateBodies
    case memoryMetricsUnavailable

    var errorDescription: String? {
        switch self {
        case .adapterMissing: "Bundle内にClaim Body Adapterがありません"
        case .unexpectedToolCall: "本文rendererがtool callを返しました"
        case .invalidContract(let detail): "Claim Body v2契約を満たさない出力です: \(detail)"
        case .missingMetrics: "生成性能metricsを取得できませんでした"
        case .duplicateBodies: "異なるclaimが同じ本文へ崩壊しました"
        case .memoryMetricsUnavailable: "iOS task memory metricsを取得できませんでした"
        }
    }
}
