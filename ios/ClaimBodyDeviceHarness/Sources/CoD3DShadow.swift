import CryptoKit
import MetalKit
import MLX
import SwiftUI
import UIKit
import simd

private struct ShadowShader: Decodable {
    let schemaVersion: Int
    let renderer: String
    let inputKind: String
    let sourceSha256: String
    let shaderSha256: String
    let shader: String
}

private struct ShadowFrame: Encodable {
    let phase: String
    let submittedAt: Double
    let completedAt: Double
    let gpuSeconds: Double
    let succeeded: Bool
}

private struct ShadowPhase: Encodable {
    let name: String
    let startedAt: Double
    let endedAt: Double
}

private struct ShadowSample: Encodable {
    let at: Double
    let thermal: String
    let memory: NativeCoDMemorySample
}

private enum ShadowError: LocalizedError {
    case invalid(String)
    var errorDescription: String? {
        switch self { case .invalid(let message): message }
    }
}

// Reuses the two production shader entry points with fixed synthetic inputs.
// This exercises GPU contention, not Unity/ArcGIS app-wide memory consumption.
@MainActor
private final class ShadowRenderer: NSObject, MTKViewDelegate {
    let view: MTKView
    let packet: ShadowShader
    let queue: MTLCommandQueue
    let volume: MTLRenderPipelineState
    let rain: MTLRenderPipelineState
    let sampler: MTLSamplerState
    let noiseSampler: MTLSamplerState
    let textures: [MTLTexture]
    let scene: MTLTexture
    let wind: MTLTexture
    var phase = "setup"
    var frames: [ShadowFrame] = []
    var skippedFrames = 0
    private var pending: MTLCommandBuffer?
    private var busy = false

    init(bundle: Bundle = .main) throws {
        guard let url = bundle.url(forResource: "shadow_renderer", withExtension: "json", subdirectory: "Shadow") else {
            throw ShadowError.invalid("prepare_iphone_3d_shadow.pyで試験shaderを準備してください")
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        packet = try decoder.decode(ShadowShader.self, from: Data(contentsOf: url))
        let digest = SHA256.hash(data: Data(packet.shader.utf8)).map { String(format: "%02x", $0) }.joined()
        guard packet.schemaVersion == 1, digest == packet.shaderSha256,
              let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw ShadowError.invalid("shader SHAまたはMetal deviceが不正です")
        }
        self.queue = queue
        let library = try device.makeLibrary(source: packet.shader, options: nil)
        func pipeline(_ fragment: String) throws -> MTLRenderPipelineState {
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.vertexFunction = library.makeFunction(name: "regional_volume_vertex")
            descriptor.fragmentFunction = library.makeFunction(name: fragment)
            descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            return try device.makeRenderPipelineState(descriptor: descriptor)
        }
        volume = try pipeline("regional_volume_fragment")
        rain = try pipeline("regional_precipitation_fragment")
        let sampling = MTLSamplerDescriptor()
        sampling.minFilter = .linear
        sampling.magFilter = .linear
        sampling.sAddressMode = .clampToEdge
        sampling.tAddressMode = .clampToEdge
        guard let linear = device.makeSamplerState(descriptor: sampling) else {
            throw ShadowError.invalid("samplerを作成できません")
        }
        sampler = linear
        sampling.sAddressMode = .repeat
        sampling.tAddressMode = .repeat
        sampling.rAddressMode = .repeat
        guard let repeating = device.makeSamplerState(descriptor: sampling) else {
            throw ShadowError.invalid("noise samplerを作成できません")
        }
        noiseSampler = repeating
        func texture(_ index: Int) throws -> MTLTexture {
            let size = 32
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba8Unorm, width: size, height: size, mipmapped: false
            )
            descriptor.storageMode = .shared
            descriptor.usage = .shaderRead
            if index == 11 {
                descriptor.textureType = .type3D
                descriptor.depth = size
            }
            guard let texture = device.makeTexture(descriptor: descriptor) else {
                throw ShadowError.invalid("試験textureを確保できません")
            }
            var bytes = [UInt8](repeating: 0, count: size * size * descriptor.depth * 4)
            for z in 0..<descriptor.depth {
                for y in 0..<size {
                    for x in 0..<size {
                        let offset = ((z * size + y) * size + x) * 4
                        let wave = 0.5 + 0.5 * sin(Float(x) * 0.23) * cos(Float(y) * 0.27)
                        let channel: UInt8 = index == 11
                            ? UInt8((x * 73 + y * 151 + z * 199) % 200 + 28)
                            : UInt8(80 + wave * 160)
                        bytes[offset] = index == 4 ? 24 : channel
                        bytes[offset + 1] = index == 4 ? 64 : channel
                        bytes[offset + 2] = index == 4 ? 92 : channel
                        bytes[offset + 3] = 255
                    }
                }
            }
            bytes.withUnsafeBytes {
                texture.replace(region: MTLRegionMake3D(0, 0, 0, size, size, descriptor.depth),
                    mipmapLevel: 0, slice: 0, withBytes: $0.baseAddress!, bytesPerRow: size * 4,
                    bytesPerImage: size * size * 4)
            }
            return texture
        }
        textures = try (0...11).map(texture)
        wind = try texture(12)
        let sceneDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: 384, height: 256, mipmapped: false
        )
        sceneDescriptor.storageMode = .shared
        sceneDescriptor.usage = [.shaderRead, .renderTarget]
        guard let scene = device.makeTexture(descriptor: sceneDescriptor) else {
            throw ShadowError.invalid("scene textureを確保できません")
        }
        self.scene = scene
        view = MTKView(frame: CGRect(x: 0, y: 0, width: 384, height: 256), device: device)
        view.colorPixelFormat = .bgra8Unorm
        view.autoResizeDrawable = false
        view.drawableSize = CGSize(width: 384, height: 256)
        view.preferredFramesPerSecond = 20
        view.isPaused = true
        super.init()
        view.delegate = self
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard !busy else { skippedFrames += 1; return }
        guard let drawable = view.currentDrawable, let output = view.currentRenderPassDescriptor,
              let command = queue.makeCommandBuffer() else { return }
        let started = ProcessInfo.processInfo.systemUptime
        let framePhase = phase
        let yaw = Float(started.truncatingRemainder(dividingBy: 120)) * 0.08
        let position = SIMD3<Float>(sin(yaw) * 2.2, 1.5, cos(yaw) * 2.2)
        let forward = simd_normalize(SIMD3<Float>(0, 0.15, 0) - position)
        let right = simd_normalize(simd_cross(forward, SIMD3<Float>(0, 1, 0)))
        let up = simd_cross(right, forward)
        // Same nine float4 slots as Regional3DMetalVolumeView.Uniforms.
        let uniforms: [SIMD4<Float>] = [
            SIMD4(384, 256, 4, 3), SIMD4(position, 0.55), SIMD4(forward, 0),
            SIMD4(right, 1), SIMD4(up, 1), SIMD4(24, 2, 0, 0), .zero,
            SIMD4(1, 0, 1, 0), SIMD4(Float(started), -0.48, 0.72, -0.30),
        ]
        let firstPass = MTLRenderPassDescriptor()
        firstPass.colorAttachments[0].texture = scene
        firstPass.colorAttachments[0].loadAction = .clear
        firstPass.colorAttachments[0].storeAction = .store
        guard let encoder = command.makeRenderCommandEncoder(descriptor: firstPass) else { return }
        encoder.setRenderPipelineState(volume)
        uniforms.withUnsafeBytes { encoder.setFragmentBytes($0.baseAddress!, length: $0.count, index: 0) }
        for (index, texture) in textures.enumerated() { encoder.setFragmentTexture(texture, index: index) }
        encoder.setFragmentSamplerState(sampler, index: 0)
        encoder.setFragmentSamplerState(noiseSampler, index: 1)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        guard let second = command.makeRenderCommandEncoder(descriptor: output) else { return }
        second.setRenderPipelineState(rain)
        uniforms.withUnsafeBytes { second.setFragmentBytes($0.baseAddress!, length: $0.count, index: 0) }
        second.setFragmentTexture(scene, index: 0)
        second.setFragmentTexture(textures[10], index: 1)
        second.setFragmentTexture(wind, index: 2)
        second.setFragmentSamplerState(sampler, index: 0)
        second.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        second.endEncoding()
        command.present(drawable)
        busy = true
        pending = command
        command.addCompletedHandler { @Sendable [weak self] buffer in
            let frame = ShadowFrame(phase: framePhase, submittedAt: started,
                completedAt: ProcessInfo.processInfo.systemUptime,
                gpuSeconds: max(0, buffer.gpuEndTime - buffer.gpuStartTime),
                succeeded: buffer.status == .completed)
            Task { @MainActor in
                self?.frames.append(frame)
                self?.busy = false
                self?.pending = nil
            }
        }
        command.commit()
    }

    func pause() async throws {
        view.isPaused = true
        // Wait asynchronously so cancellation and UI updates remain responsive.
        let deadline = ProcessInfo.processInfo.systemUptime + 5
        while busy {
            guard ProcessInfo.processInfo.systemUptime < deadline else {
                throw ShadowError.invalid("GPU終了待ちが5秒を超えました")
            }
            try await Task.sleep(for: .milliseconds(10))
        }
    }

    func saveImage(named name: String) throws {
        var bytes = [UInt8](repeating: 0, count: scene.width * scene.height * 4)
        scene.getBytes(&bytes, bytesPerRow: scene.width * 4,
            from: MTLRegionMake2D(0, 0, scene.width, scene.height), mipmapLevel: 0)
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let image = CGImage(width: scene.width, height: scene.height, bitsPerComponent: 8,
                bitsPerPixel: 32, bytesPerRow: scene.width * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)
                    .union(.byteOrder32Little), provider: provider, decode: nil,
                shouldInterpolate: false, intent: .defaultIntent),
              let png = UIImage(cgImage: image).pngData() else {
            throw ShadowError.invalid("描画証跡を保存できません")
        }
        let url = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true).appendingPathComponent(name)
        try png.write(to: url, options: .atomic)
    }
}

private struct ShadowResult: Encodable {
    let schemaVersion = 1
    let createdAt = Date()
    let mode: String
    let physicalDevice: Bool
    let freshBody: Bool
    let renderer: String
    let inputKind: String
    let sourceSHA256: String
    let shaderSHA256: String
    let renderWidth = 384
    let renderHeight = 256
    let targetFPS = 20
    let raySteps = 24
    let productionIntegrationAllowed = false
    let cancelled: Bool
    let stopReason: String?
    let stopRequestedAt: Double?
    let stopInferenceProgress: String?
    let inferenceFinishedAt: Double
    let codResult: NativeCoDResult?
    let error: String?
    let phases: [ShadowPhase]
    let samples: [ShadowSample]
    let frames: [ShadowFrame]
    let skippedFrames: Int
}

private struct ShadowFailure: Encodable {
    let schemaVersion = 1
    let createdAt = Date()
    let status = "hold"
    let mode: String
    let error: String
}

@MainActor
private final class ShadowController: ObservableObject {
    @Published var renderer: ShadowRenderer?
    @Published var status = "準備中"
    @Published var log = ""
    @Published var running = false
    private var inference: Task<NativeCoDResult, Error>?
    private var phase = "setup"
    private var samples: [ShadowSample] = []
    private var stopReason: String?
    private var stopRequestedAt: Double?
    private var inferenceProgress = ""
    private var stopInferenceProgress: String?

    func stop(_ reason: String) {
        guard running, stopReason == nil else { return }
        stopReason = reason
        stopRequestedAt = ProcessInfo.processInfo.systemUptime
        stopInferenceProgress = inferenceProgress
        renderer?.view.isPaused = true
        inference?.cancel()
        log += "stop_requested=\(reason)\n"
    }

    private func sample() {
        if let memory = try? NativeCoDSmokeRunner.memorySample(stage: phase) {
            let thermal = NativeCoDSmokeRunner.thermalName(ProcessInfo.processInfo.thermalState)
            samples.append(ShadowSample(at: ProcessInfo.processInfo.systemUptime, thermal: thermal, memory: memory))
            if ["serious", "critical"].contains(thermal) { stop("thermal_\(thermal)") }
            if memory.limitBytesRemaining < 512 * 1_048_576 { stop("memory_headroom") }
        }
    }

    func run() async {
        guard !running else { return }
        running = true
        UIApplication.shared.isIdleTimerDisabled = true
        defer { running = false; UIApplication.shared.isIdleTimerDisabled = false }
        let arguments = ProcessInfo.processInfo.arguments
        let freshBody = arguments.contains("--fresh-body")
        let mode = arguments.contains("--3d-handoff") ? "handoff"
            : arguments.contains("--simulate-memory-warning") ? "memory_warning_simulated" : "concurrent"
        let suffix = freshBody ? "_fresh_body" : ""
        let filename = "mp_cod_a15_3d_\(mode)\(suffix).json"
        var monitoring: Task<Void, Never>?
        defer { monitoring?.cancel() }
        do {
            #if targetEnvironment(simulator)
            throw ShadowError.invalid("この試験は物理iPhone専用です")
            #endif
            guard ProcessInfo.processInfo.thermalState == .nominal else {
                throw ShadowError.invalid("thermal nominalへ戻ってから実行してください")
            }
            let renderer = try ShadowRenderer()
            self.renderer = renderer
            log = "mode=\(mode) fresh_body=\(freshBody)\nshader_sha256=\(renderer.packet.shaderSha256)\n"
            var phases: [ShadowPhase] = []
            func begin(_ name: String) -> Double {
                phase = name
                renderer.phase = name
                status = name
                sample()
                return ProcessInfo.processInfo.systemUptime
            }
            func finish(_ name: String, _ start: Double) {
                phases.append(ShadowPhase(name: name, startedAt: start, endedAt: ProcessInfo.processInfo.systemUptime))
            }
            monitoring = Task { @MainActor in
                while !Task.isCancelled {
                    self.sample()
                    try? await Task.sleep(for: .milliseconds(250))
                }
            }
            // Give SwiftUI time to attach the MTKView before measuring frames.
            try await Task.sleep(for: .milliseconds(500))
            renderer.view.isPaused = false
            let baselineStart = begin("baseline_3d")
            try await Task.sleep(for: .seconds(3))
            try await renderer.pause()
            finish("baseline_3d", baselineStart)
            try renderer.saveImage(named: "mp_cod_a15_3d_\(mode)\(suffix).png")
            if let stopReason { throw ShadowError.invalid("事前gate: \(stopReason)") }
            let inferenceStart = begin("inference")
            renderer.view.isPaused = mode != "concurrent"
            var trigger: Task<Void, Never>?
            defer { trigger?.cancel() }
            inference = Task { @MainActor in
                try await NativeCoDSmokeRunner.run(scenario: .typhoon18Replay, bypassBodyCache: freshBody) { progress in
                    self.status = progress
                    self.inferenceProgress = progress
                    print("MP_COD_3D_PHASE \(progress)")
                    let triggerPrefix = freshBody ? "本文cache prime:" : "盲検選択:"
                    if mode != "concurrent", trigger == nil, progress.hasPrefix(triggerPrefix) {
                        trigger = Task { @MainActor in
                            try? await Task.sleep(for: .milliseconds(250))
                            guard !Task.isCancelled else { return }
                            if mode == "handoff" {
                                self.stop("3d_requested")
                            } else {
                                NotificationCenter.default.post(name: UIApplication.didReceiveMemoryWarningNotification,
                                    object: nil, userInfo: ["mp_cod_simulated": true])
                            }
                        }
                    }
                }
            }
            var result: NativeCoDResult?
            var cancelled = false
            var errorText: String?
            do { result = try await inference?.value }
            catch is CancellationError { cancelled = true }
            catch { errorText = String(describing: error) }
            let finishedAt = ProcessInfo.processInfo.systemUptime
            inference = nil
            Memory.clearCache()
            try await renderer.pause()
            finish("inference", inferenceStart)
            phase = "after_inference_release"
            sample()
            if let result { log += result.transcript + "\n" }
            let recoveryStart = begin("recovery_3d")
            if stopReason == nil || ["3d_requested", "memory_warning_simulated"].contains(stopReason!) {
                renderer.view.isPaused = false
                try await Task.sleep(for: .seconds(3))
            }
            try await renderer.pause()
            finish("recovery_3d", recoveryStart)
            sample()
            let report = ShadowResult(mode: mode, physicalDevice: true, freshBody: freshBody,
                renderer: renderer.packet.renderer, inputKind: renderer.packet.inputKind,
                sourceSHA256: renderer.packet.sourceSha256, shaderSHA256: renderer.packet.shaderSha256,
                cancelled: cancelled, stopReason: stopReason, stopRequestedAt: stopRequestedAt,
                stopInferenceProgress: stopInferenceProgress,
                inferenceFinishedAt: finishedAt, codResult: result, error: errorText,
                phases: phases, samples: samples, frames: renderer.frames, skippedFrames: renderer.skippedFrames)
            try NativeCoDSmokeRunner.save(report, named: filename)
            status = "計測完了・外部検証待ち"
            log += "result=\(filename) frames=\(renderer.frames.count) cancelled=\(cancelled)\n"
            print("MP_COD_3D_SHADOW \(log)")
        } catch {
            try? await renderer?.pause()
            status = "HOLD"
            log += "error=\(error)\n"
            do {
                try NativeCoDSmokeRunner.save(
                    ShadowFailure(mode: mode, error: String(describing: error)), named: filename
                )
            } catch {
                log += "result_write_error=\(error)\n"
            }
            print("MP_COD_3D_SHADOW HOLD \(error)")
        }
    }

    func memoryWarning(_ notification: Notification) {
        let simulated = notification.userInfo?["mp_cod_simulated"] as? Bool == true
        stop(simulated ? "memory_warning_simulated" : "memory_warning_os")
    }
}

private struct ShadowMetalView: UIViewRepresentable {
    let view: MTKView
    func makeUIView(context: Context) -> MTKView { view }
    func updateUIView(_ view: MTKView, context: Context) {}
}

struct CoD3DShadowView: View {
    @StateObject private var controller = ShadowController()
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("CoD × 3D 実機Shadow").font(.headline)
            Text("ExtremeWeather Metal shader・固定試験雲場\n384×256 / 24 steps / 20 fps目標。本体全体の負荷は別途検証。")
                .font(.caption)
            if let renderer = controller.renderer {
                ShadowMetalView(view: renderer.view).frame(height: 256)
            }
            Text(controller.status).accessibilityIdentifier("shadowStatus")
            ScrollView { Text(controller.log).font(.caption.monospaced()).textSelection(.enabled) }
            if controller.running {
                Button("停止") { controller.stop("user") }
                    .accessibilityIdentifier("cancel3DShadow")
            }
        }
        .padding()
        .task { await controller.run() }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { notification in
            controller.memoryWarning(notification)
        }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.willResignActiveNotification)) { _ in
            controller.stop("app_inactive")
        }
    }
}
