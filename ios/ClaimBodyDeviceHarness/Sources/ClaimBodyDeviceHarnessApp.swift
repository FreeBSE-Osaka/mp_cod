import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import SwiftUI
import Tokenizers

@main
struct ClaimBodyDeviceHarnessApp: App {
    var body: some Scene {
        WindowGroup {
            ClaimBodyDeviceHarnessView()
        }
    }
}

private struct RendererResponse: Decodable {
    struct Body: Decodable {
        let id: String
        let body: String
    }

    let bodies: [Body]
}

private struct SmokeResult: Encodable {
    let schemaVersion: Int
    let createdAt: Date
    let model: String
    let adapterWeightsSHA256: String
    let baseLoadSeconds: Double
    let adapterLoadSeconds: Double
    let timeToFirstTokenSeconds: Double
    let promptTokens: Int
    let generationTokens: Int
    let tokensPerSecond: Double
    let totalSeconds: Double
    let thermalState: String
    let output: String

    enum CodingKeys: String, CodingKey {
        case model, output
        case schemaVersion = "schema_version"
        case createdAt = "created_at"
        case adapterWeightsSHA256 = "adapter_weights_sha256"
        case baseLoadSeconds = "base_load_seconds"
        case adapterLoadSeconds = "adapter_load_seconds"
        case timeToFirstTokenSeconds = "ttft_seconds"
        case promptTokens = "prompt_tokens"
        case generationTokens = "generation_tokens"
        case tokensPerSecond = "tokens_per_second"
        case totalSeconds = "total_seconds"
        case thermalState = "thermal_state"
    }
}

private struct ClaimBodyDeviceHarnessView: View {
    @State private var status = "実行待ち"
    @State private var log = "Baseは端末へ初回downloadします。"
    @State private var downloadProgress = 0.0
    @State private var isRunning = false
    @State private var didAutoRun = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Qwen3 1.7B 4bit + Claim Body v3")
                        .font(.headline)
                    Text("iPhone実機だけでBase load、LoRA適用、1発言生成、unloadを確認します。")
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
                        Task { await runSmoke() }
                    } label: {
                        Label("A15 Smokeを実行", systemImage: "cpu")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isRunning)
                    .accessibilityIdentifier("runA15Smoke")
                }
                .padding()
            }
            .navigationTitle("MP-CoD Weight Smoke")
            .task {
                guard !didAutoRun,
                      ProcessInfo.processInfo.arguments.contains("--autorun") else { return }
                didAutoRun = true
                await runSmoke()
            }
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

        defer { isRunning = false }
        do {
            let container = try await #huggingFaceLoadModelContainer(
                configuration: LLMRegistry.qwen3_1_7b_4bit,
                progressHandler: { progress in
                    Task { @MainActor in
                        downloadProgress = progress.fractionCompleted
                        status = "Base取得 \(Int(progress.fractionCompleted * 100))%"
                    }
                }
            )
            let loadSeconds = seconds(since: totalStart)
            log += "base_load_seconds=\(format(loadSeconds))\n"

            guard let adapterDirectory = Bundle.main.url(
                forResource: "Adapter",
                withExtension: nil
            ) else {
                throw HarnessError.adapterMissing
            }
            let adapterStart = ContinuousClock.now
            let adapter = try LoRAContainer.from(directory: adapterDirectory)
            try await container.perform { context in
                try adapter.load(into: context.model)
            }
            let adapterSeconds = seconds(since: adapterStart)
            log += "adapter_load_seconds=\(format(adapterSeconds))\n"

            let system = "各itemのspeakerとして、検証済みclaimを自然な日本語一文で述べる本文renderer。claimの内容、時制、数字を変更・追加せず、moveや賛否は表現しない。入力itemsと同じidを一度ずつ返す。出力はbodiesだけをキーに持つJSONで、各要素のキーはidとbodyだけ。必ず {\"bodies\":[{\"id\":\"入力id\",\"body\":\"本文\"}]} の形で返し、idをJSONキーにしてはならない。提案や計画を実現済み・検証済み等の完了事実へ変えず、claimの時制と確実性を保つ。"
            let prompt = #"{"items":[{"id":"B01","speaker":"影響・リスク予報者","claim":"暴風が強まる前の安全確保を優先する"}]}"#
            let session = ChatSession(
                container,
                instructions: system,
                generateParameters: GenerateParameters(maxTokens: 96, temperature: 0),
                additionalContext: ["enable_thinking": false]
            )
            status = "Claim Bodyを生成中"
            let generationStart = ContinuousClock.now
            var firstTokenSeconds: Double?
            var output = ""
            var completion: GenerateCompletionInfo?
            for try await event in session.streamDetails(to: prompt) {
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

            let response = try JSONDecoder().decode(
                RendererResponse.self,
                from: Data(output.utf8)
            )
            guard response.bodies.count == 1,
                  response.bodies[0].id == "B01",
                  response.bodies[0].body.contains("安全確保"),
                  response.bodies[0].body.contains("優先"),
                  response.bodies[0].body.range(
                    of: #"D\d{2,}"#,
                    options: .regularExpression
                  ) == nil else {
                throw HarnessError.invalidContract
            }

            log += "ttft_seconds=\(format(firstTokenSeconds ?? .nan))\n"
            if let completion {
                log += "prompt_tokens=\(completion.promptTokenCount)\n"
                log += "generation_tokens=\(completion.generationTokenCount)\n"
                log += "tokens_per_second=\(format(completion.tokensPerSecond))\n"
            }
            log += "output=\(output)\n"

            await container.perform { context in
                adapter.unload(from: context.model)
            }
            await session.clear()
            let totalSeconds = seconds(since: totalStart)
            let thermalState = thermalName(ProcessInfo.processInfo.thermalState)
            log += "total_seconds=\(format(totalSeconds))\n"
            log += "thermal_state=\(thermalState)\n"
            guard let completion, let firstTokenSeconds else {
                throw HarnessError.missingMetrics
            }
            let result = SmokeResult(
                schemaVersion: 1,
                createdAt: Date(),
                model: "mlx-community/Qwen3-1.7B-4bit",
                adapterWeightsSHA256: "4ce21e64af220f0ee309599e189fd136e10c4c5cd11440c3d60fd306749a9a92",
                baseLoadSeconds: loadSeconds,
                adapterLoadSeconds: adapterSeconds,
                timeToFirstTokenSeconds: firstTokenSeconds,
                promptTokens: completion.promptTokenCount,
                generationTokens: completion.generationTokenCount,
                tokensPerSecond: completion.tokensPerSecond,
                totalSeconds: totalSeconds,
                thermalState: thermalState,
                output: output
            )
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let resultURL = try FileManager.default.url(
                for: .documentDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            ).appendingPathComponent("mp_cod_a15_smoke.json")
            try encoder.encode(result).write(to: resultURL, options: .atomic)
            log += "result_file=mp_cod_a15_smoke.json\n"
            status = "PASS"
            downloadProgress = 1
            print("MP_COD_A15_SMOKE PASS\n\(log)")
        } catch {
            status = "FAIL"
            log += "error=\(error.localizedDescription)\n"
            print("MP_COD_A15_SMOKE FAIL \(error)")
        }
    }

    private func seconds(since start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: .now)
        return Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1_000_000_000_000_000_000
    }

    private func format(_ value: Double) -> String {
        value.formatted(.number.precision(.fractionLength(3)))
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
    case invalidContract
    case missingMetrics

    var errorDescription: String? {
        switch self {
        case .adapterMissing: "Bundle内にClaim Body Adapterがありません"
        case .unexpectedToolCall: "本文rendererがtool callを返しました"
        case .invalidContract: "Claim Body v2契約を満たさない出力です"
        case .missingMetrics: "生成性能metricsを取得できませんでした"
        }
    }
}
