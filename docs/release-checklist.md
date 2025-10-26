# TinyBrain v0.1.0 Release Checklist

**Version:** 0.1.0  
**Release Date:** October 25, 2025  
**Completed By:** [Your Name]

---

## Pre-Release Validation

### Code Quality

- [x] **All tests passing** (179/179)
  ```bash
  swift test
  # Result: 179 tests, 0 failures ✅
  ```

- [x] **Python tests passing** (7/7)
  ```bash
  source .venv/bin/activate
  pytest Tests/test_convert_model.py -v
  # Result: 7 passed, 3 skipped ✅
  ```

- [ ] **Linting clean**
  ```bash
  make lint
  # Expected: No violations
  ```

- [ ] **Format check clean**
  ```bash
  make format-check
  # Expected: No formatting issues
  ```

- [x] **Build succeeds (debug)**
  ```bash
  swift build
  # Result: Build complete! ✅
  ```

- [x] **Build succeeds (release)**
  ```bash
  swift build -c release
  # Result: Build complete! ✅
  ```

### Documentation

- [x] **README.md updated**
  - [x] Version badge reflects v0.1.0
  - [x] Feature list complete
  - [x] Installation instructions accurate
  - [x] Quick start works

- [x] **AGENTS.md current**
  - [x] Section 8: Development Workflows (TB-007)
  - [x] All rules reflect current state

- [x] **docs/overview.md updated**
  - [x] Architecture diagrams current
  - [x] Test count updated (179)
  - [x] TB-006 completion noted

- [x] **docs/faq.md complete**
  - [x] TextField issue documented
  - [x] AMX findings explained
  - [x] Troubleshooting comprehensive

- [x] **docs/benchmarking.md exists**
  - [x] Methodology explained
  - [x] YAML format documented
  - [x] Metric interpretation guide

- [x] **Release notes complete**
  - [x] docs/releases/v0.1.0.md
  - [x] Features listed
  - [x] Known issues documented
  - [x] Performance metrics included
  - [x] What's next outlined

- [x] **Task files complete**
  - [x] TB-001 through TB-006 marked complete
  - [x] TB-007 marked in-progress (Phases 1-7 done)

- [ ] **DocC builds without errors**
  ```bash
  make docs
  # Expected: Documentation generated successfully
  ```

### Application Testing

#### macOS (Manual)

- [ ] **ChatDemo app runs**
  - Open `Package.swift` in Xcode
  - Select ChatDemo scheme
  - Edit Scheme → Disable sandbox
  - Run app (Cmd+R)
  - [ ] UI loads correctly
  - [ ] TextField accepts input ✅
  - [ ] Message bubbles display
  - [ ] Streaming works
  - [ ] Telemetry updates

- [ ] **tinybrain-bench runs**
  ```bash
  .build/release/tinybrain-bench --demo --tokens 10
  # Expected: Benchmark completes, shows results
  ```

- [ ] **Device info works**
  ```bash
  .build/release/tinybrain-bench --device-info
  # Expected: Shows device, OS, GPU info
  ```

- [ ] **YAML scenarios work**
  ```bash
  .build/release/tinybrain-bench \
    --scenario benchmarks/scenarios.yml \
    --dry-run
  # Expected: Parses scenarios successfully
  ```

#### iOS (If available)

- [ ] **ChatDemo on iPhone 16 Pro**
  - Deploy via Xcode
  - [ ] App launches
  - [ ] TextField works (no sandbox issue on iOS)
  - [ ] Streaming functional
  - [ ] No crashes

### Model Conversion

- [x] **TinyLlama downloaded**
  ```bash
  ls -lh Models/tinyllama-raw/model.safetensors
  # Result: 2.0GB file exists ✅
  ```

- [x] **Conversion works**
  ```bash
  source .venv/bin/activate
  python Scripts/convert_model.py \
    --input Models/tinyllama-raw/model.safetensors \
    --output Models/tinyllama-1.1b-int8.tbf \
    --quantize int8 \
    --auto-config
  # Result: 808MB TBF file created ✅
  ```

- [ ] **Converter tests pass**
  ```bash
  pytest Tests/test_convert_model.py
  # Expected: 7 passed, 3 skipped
  ```

### Benchmarks

- [ ] **Baseline benchmarks run**
  ```bash
  ./Scripts/run_benchmarks.sh
  # Expected: Creates timestamped results in benchmarks/results/
  ```

- [x] **Baseline docs exist**
  - [x] benchmarks/baseline-m4-max.md
  - [x] benchmarks/baseline-iphone16pro.md (placeholder)

- [ ] **Results validate**
  - [ ] < 150 ms/token target
  - [ ] < 2 GB memory usage
  - [ ] No crashes
  - [ ] Metrics reasonable

### Repository Hygiene

- [ ] **No sensitive data**
  ```bash
  # Check for API keys, tokens, credentials
  grep -r "API_KEY" . --exclude-dir=.git
  grep -r "sk-" . --exclude-dir=.git
  # Expected: No matches
  ```

- [ ] **LICENSE file exists**
  - [x] MIT License
  - [x] Copyright year correct (2025)
  - [x] Author attribution

- [ ] **.gitignore correct**
  - [x] Models/* ignored (except .gitkeep)
  - [x] .venv/ ignored
  - [x] .build/ ignored
  - [x] *.tbf, *.pt, *.safetensors ignored

- [ ] **No large files**
  ```bash
  find . -type f -size +10M | grep -v ".git" | grep -v "Models/"
  # Expected: No matches (models are gitignored)
  ```

### CI/CD

- [x] **GitHub Actions workflow exists**
  - [x] .github/workflows/ci.yml

- [ ] **CI pipeline runs**
  - Push to test branch
  - Check GitHub Actions tab
  - [ ] Lint passes
  - [ ] Format check passes
  - [ ] Build passes
  - [ ] Tests pass
  - [ ] Documentation builds
  - [ ] Benchmark smoke test passes

### Dependencies

- [x] **Swift dependencies resolved**
  ```bash
  swift package resolve
  cat Package.resolved
  # Expected: ArgumentParser, Yams
  ```

- [x] **Python dependencies documented**
  ```bash
  cat requirements.txt
  # Expected: torch, safetensors, numpy, pytest, tqdm
  ```

- [ ] **Dependency versions pinned**
  - Check Package.resolved for Swift
  - Check requirements.txt for Python
  - Verify no conflicts

---

## Release Preparation

### Version Bumps

- [x] **Package.swift version**
  - Check version comments reflect 0.1.0

- [x] **TinyBrainBench version**
  - Sources/TinyBrainBench/TinyBrainBench.swift: `version: "0.1.0"` ✅

- [ ] **README.md badge**
  - Update version badge to v0.1.0

### Git

- [ ] **All changes committed**
  ```bash
  git status
  # Expected: "nothing to commit, working tree clean"
  ```

- [ ] **Commit messages clean**
  - Review `git log --oneline -20`
  - Check for task IDs (TB-001 through TB-007)
  - No WIP or temp commits

- [ ] **Branch clean**
  - On main/master branch
  - Up to date with remote
  - No uncommitted changes

### Tagging

- [ ] **Create annotated tag**
  ```bash
  git tag -a v0.1.0 -m "TinyBrain v0.1.0: Foundation Release"
  ```

- [ ] **Verify tag**
  ```bash
  git tag -l -n9 v0.1.0
  # Should show tag message
  ```

- [ ] **Push tag**
  ```bash
  git push origin v0.1.0
  ```

### GitHub Release

- [ ] **Create GitHub Release**
  - Go to Releases → Draft new release
  - Tag: v0.1.0
  - Title: "TinyBrain v0.1.0 - Foundation"
  - Body: Copy from docs/releases/v0.1.0.md
  - [ ] Attach artifacts (optional):
    - Source code (auto)
    - Documentation archive (from CI)
    - Benchmark results

- [ ] **Release notes accurate**
  - Features match implementation
  - Known issues disclosed
  - Installation instructions tested

---

## Post-Release

### Verification

- [ ] **GitHub release published**
  - Visible at github.com/vivekptnk/tinybrain/releases
  - Assets downloadable
  - Release notes formatted correctly

- [ ] **Tag pushed**
  ```bash
  git ls-remote --tags origin
  # Should show v0.1.0
  ```

- [ ] **CI badge updates**
  - Check README.md CI badge shows passing

### Communication

- [ ] **Announcement drafted**
  - Twitter/X (optional)
  - GitHub Discussions
  - README updated with release link

- [ ] **Issues closed**
  - Close any "v0.1.0 blocker" issues
  - Link to release in comments

### Documentation

- [ ] **Update main branch README**
  - Point to v0.1.0 release
  - Update "Latest Release" section

- [ ] **Archive old docs**
  - Move previous summaries to archives/ if needed

---

## Rollback Plan

If critical issues found post-release:

1. **Delete GitHub release**
   - Mark as draft or delete

2. **Delete tag**
   ```bash
   git tag -d v0.1.0
   git push --delete origin v0.1.0
   ```

3. **Fix issues**
   - Create hotfix branch
   - Fix critical bugs
   - Re-test thoroughly

4. **Re-release**
   - Follow checklist again
   - Consider v0.1.1 instead

---

## Sign-Off

- [ ] **All checklist items complete**
- [ ] **No critical issues found**
- [ ] **Ready for public release**

**Signed off by:** _______________  
**Date:** _______________  
**Notes:**

---

## Evidence Log

### Test Results

```bash
# Swift tests
swift test
# → [PASTE RESULTS HERE]

# Python tests
pytest Tests/test_convert_model.py -v
# → [PASTE RESULTS HERE]

# Benchmark
.build/release/tinybrain-bench --demo --tokens 10
# → [PASTE RESULTS HERE]
```

### Screenshots

- [ ] ChatDemo on macOS (with TextField working)
- [ ] Benchmark results JSON
- [ ] Device info output

### Metrics

- **Test count:** 179 Swift + 7 Python = 186 total
- **Line count:** ~15,000+
- **File count:** ~100+
- **Documentation pages:** 25+
- **Tasks completed:** TB-001 through TB-007

---

**Checklist completed:** ______ / 50+ items  
**Status:** ⏳ In Progress  
**Target Release:** October 25, 2025

