import CoreML
import XCTest

@testable import FluidAudio

final class LocalOnlyModelLoadingTests: XCTestCase {
    func testClassicModelsLoadFromExplicitLocalDirectory() async throws {
        let directory = try modelDirectory(environmentKey: "FLUIDAUDIO_CLASSIC_MODEL_DIR")
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly

        _ = try await AsrModels.loadLocalOnly(
            from: directory,
            configuration: configuration,
            version: .v3,
            encoderPrecision: .int8,
            encoderComputeUnits: .cpuOnly
        )
    }

    func testStreamingEouModelsLoadFromExplicitLocalDirectory() async throws {
        let directory = try modelDirectory(environmentKey: "FLUIDAUDIO_EOU_MODEL_DIR")
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        let manager = StreamingEouAsrManager(
            configuration: configuration,
            chunkSize: .ms320,
            eouDebounceMs: 1_280
        )

        try await manager.loadModels(from: directory)
        await manager.cleanup()
    }

    func testCtcVocabularyModelsLoadFromExplicitLocalDirectory() async throws {
        let directory = try modelDirectory(environmentKey: "FLUIDAUDIO_CTC_MODEL_DIR")
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly

        let models = try await CtcModels.loadDirect(
            from: directory,
            variant: .ctc110m,
            configuration: configuration
        )

        XCTAssertEqual(models.variant, .ctc110m)
        XCTAssertEqual(models.configuration.computeUnits, .cpuOnly)
        XCTAssertFalse(models.vocabulary.isEmpty)
        _ = try await CtcTokenizer.load(from: directory)
    }

    private func modelDirectory(environmentKey: String) throws -> URL {
        guard let path = ProcessInfo.processInfo.environment[environmentKey], !path.isEmpty else {
            throw XCTSkip("Set \(environmentKey) to run the local model integration test.")
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }
}
