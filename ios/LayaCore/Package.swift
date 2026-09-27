// swift-tools-version: 5.9
// The platform-independent half of the iOS app: tokenizer, token sequence, image resize and answers, ported from
// web-demo/laya.js (itself a port of laya/vlm.py). No Apple-only frameworks, so `swift build` works on Linux and
// tests/test_ios_core.py can check it against Python there.
import PackageDescription

let package = Package(
    name: "LayaCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [.library(name: "LayaCore", targets: ["LayaCore"])],
    targets: [
        .target(name: "LayaCore"),
        // `laya-core-check ids|pixels ...`, driven by tests/test_ios_core.py
        .executableTarget(name: "laya-core-check", dependencies: ["LayaCore"]),
    ]
)
