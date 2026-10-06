import CryptoKit
import Foundation

/// A bounded, foreground-only experiment, not a general remote execution API.
struct DistributedBodyRequest: Sendable {
    let requestID: String
    let jobIndices: [Int]

    var resultFilename: String { "mp_cod_distributed_\(requestID).json" }

    static func parse(_ arguments: [String]) throws -> Self? {
        guard arguments.contains("--distributed-worker") else { return nil }
        func value(_ key: String) throws -> String {
            let positions = arguments.indices.filter { arguments[$0] == key }
            guard positions.count == 1, let index = positions.first,
                  index + 1 < arguments.count else { throw RequestError.invalidArguments }
            return arguments[index + 1]
        }
        guard let uuid = UUID(uuidString: try value("--request-id")) else {
            throw RequestError.invalidArguments
        }
        let values = try value("--job-indices").split(separator: ",", omittingEmptySubsequences: false)
        let indices = values.compactMap { Int($0) }
        guard indices.count == values.count, (1...4).contains(indices.count),
              Set(indices).count == indices.count, indices.allSatisfy({ (0...3).contains($0) }) else {
            throw RequestError.invalidArguments
        }
        return Self(requestID: uuid.uuidString.lowercased(), jobIndices: indices)
    }

    static func fileSHA256(_ url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var digest = SHA256()
        while let data = try file.read(upToCount: 1_048_576), !data.isEmpty {
            digest.update(data: data)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    enum RequestError: LocalizedError {
        case invalidArguments
        case localModelMissing
        case unsafeResources
        case existingResult

        var errorDescription: String? {
            switch self {
            case .invalidArguments: "試行はUUIDと重複のない0〜3のjob番号が必要です"
            case .localModelMissing: "USBで配置したローカル1.7Bモデルがありません。自動downloadはしません"
            case .unsafeResources: "thermalまたは512 MiBのmemory余裕条件を満たしていません"
            case .existingResult: "同じrequestの結果を上書きしません"
            }
        }
    }
}
