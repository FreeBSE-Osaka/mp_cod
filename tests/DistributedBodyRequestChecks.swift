import Foundation

@main
struct DistributedBodyRequestChecks {
    static func main() throws {
        let id = UUID().uuidString.lowercased()
        let args = ["app", "--distributed-worker", "--request-id", id, "--job-indices", "1,3"]
        let request = try DistributedBodyRequest.parse(args)!
        precondition(request.requestID == id && request.jobIndices == [1, 3])
        precondition(request.resultFilename == "mp_cod_distributed_\(id).json")
        let absent = try DistributedBodyRequest.parse(["app"])
        precondition(absent == nil)
        for bad in [
            ["--distributed-worker"],
            ["--distributed-worker", "--request-id", "../../bad", "--job-indices", "1"],
            ["--distributed-worker", "--request-id", id, "--job-indices", "1,1"],
            ["--distributed-worker", "--request-id", id, "--job-indices", "4"],
            ["--distributed-worker", "--request-id", id, "--job-indices", "1,"],
            ["--distributed-worker", "--request-id", id, "--request-id", id, "--job-indices", "1"],
        ] {
            do {
                _ = try DistributedBodyRequest.parse(bad)
                preconditionFailure("invalid request was accepted")
            } catch {}
        }
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("mp-cod-request-check-\(UUID())")
        defer { try? FileManager.default.removeItem(at: file) }
        try Data("abc".utf8).write(to: file, options: .withoutOverwriting)
        let hash = try DistributedBodyRequest.fileSHA256(file)
        precondition(hash == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        print("distributed request and streaming SHA checks passed")
    }
}
