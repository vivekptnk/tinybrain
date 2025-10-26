import XCTest
import Combine
@testable import TinyBrainRuntime

final class CombinePublisherTests: XCTestCase {
    private var cancellables: Set<AnyCancellable> = []

    override func tearDown() {
        cancellables.removeAll()
        super.tearDown()
    }

    func testGeneratePublisherEmitsAndCompletes() {
        let config = ModelConfig(numLayers: 1, hiddenDim: 32, numHeads: 4, vocabSize: 64, maxSeqLen: 64)
        let runner = ModelRunner(config: config)

        let genConfig = GenerationConfig(maxTokens: 5)

        let expectationFinished = expectation(description: "Publisher finished")
        var received: [TokenOutput] = []

        runner.generatePublisher(prompt: [1, 2, 3], config: genConfig)
            .sink(receiveCompletion: { completion in
                switch completion {
                case .finished:
                    expectationFinished.fulfill()
                case .failure(let error):
                    XCTFail("Unexpected error: \(error)")
                }
            }, receiveValue: { value in
                received.append(value)
            })
            .store(in: &cancellables)

        wait(for: [expectationFinished], timeout: 5.0)

        // Should have received exactly maxTokens values
        XCTAssertEqual(received.count, 5)
        // Validate fields are sane
        for output in received {
            XCTAssertGreaterThanOrEqual(output.tokenId, 0)
            XCTAssertLessThan(output.tokenId, 64)
            XCTAssertGreaterThan(output.probability, 0)
            XCTAssertLessThanOrEqual(output.probability, 1)
            XCTAssertTrue(output.entropy.isFinite)
        }

        runner.reset()
    }
}


