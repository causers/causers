# Release Checklist for causers

This checklist ensures a consistent and thorough release process for the causers package.

## Pre-Release Validation

### 1. Code Quality ✓

- [ ] All tests passing locally
  ```bash
  cargo test
  pytest tests/ -v
  ```
  
- [ ] Test coverage meets requirements (>90% Python, >80% Rust)
  ```bash
  pytest tests/ --cov=causers --cov-report=html
  # Check coverage report in htmlcov/index.html
  ```

- [ ] Code formatting applied
  ```bash
  black python/ tests/
  cargo fmt
  ```

- [ ] Linting passes without warnings
  ```bash
  ruff check python/ tests/
  cargo clippy
  ```

- [ ] Type checking passes
  ```bash
  mypy python/
  ```

### 2. Performance Validation ✓

- [ ] Performance benchmarks meet requirements
  ```bash
  pytest tests/test_performance.py -v -s
  # Verify: 1M rows < 100ms (REQ-037)
  ```

- [ ] Memory usage within limits
  ```bash
  # Monitor memory during test runs
  # Should be < 1.5x input DataFrame size
  ```

### 3. Security Review ✓

- [ ] No new unsafe Rust blocks (except PyO3 required)
  ```bash
  grep -r "unsafe" src/ --include="*.rs"
  ```

- [ ] Dependency audit passes
  ```bash
  cargo audit  # Install with: cargo install cargo-audit
  ```

- [ ] No hardcoded secrets or credentials
  ```bash
  grep -r "password\|secret\|key\|token" . --exclude-dir=target
  ```

### 4. Documentation ✓

- [ ] README.md updated with latest features
- [ ] CHANGELOG.md entry for new version
- [ ] API documentation complete
- [ ] All public functions have docstrings
- [ ] Examples tested and working

### 5. Platform Testing ✓

- [ ] Linux build and tests pass
- [ ] macOS (Intel) build and tests pass
- [ ] macOS (ARM) build and tests pass
- [ ] Windows build and tests pass

## Version Bumping Process

### 1. Update Version Numbers

- [ ] Update version in [`Cargo.toml`](../Cargo.toml:7)
  ```toml
  [package]
  version = "0.1.0"  # Update this
  ```

- [ ] Update version in [`pyproject.toml`](../pyproject.toml:7)
  ```toml
  [project]
  version = "0.1.0"  # Update this
  ```

- [ ] Update version in [`python/causers/__init__.py`](../python/causers/__init__.py:8)
  ```python
  __version__ = "0.1.0"  # Update this
  ```

### 2. Update Documentation

- [ ] Update CHANGELOG.md with release date
- [ ] Update README.md badges if needed
- [ ] Review and update roadmap section

### 3. Commit Version Changes

```bash
git add Cargo.toml pyproject.toml python/causers/__init__.py CHANGELOG.md
git commit -m "chore: bump version to v0.1.0"
git tag -a v0.1.0 -m "Release v0.1.0"
```

## Build Process

### 1. Clean Previous Builds

```bash
cargo clean
rm -rf target/wheels/
rm -rf build/ dist/ *.egg-info
```

### 2. Build Release Artifacts

```bash
# Build optimized Rust library
maturin build --release

# Artifacts will be in target/wheels/
ls -la target/wheels/
```

### 3. Build for Multiple Platforms (CI/CD)

```yaml
# This should be handled by GitHub Actions
# See .github/workflows/release.yml
```

### 4. Test Built Wheel

```bash
# Create clean virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install the built wheel
pip install target/wheels/causers-*.whl

# Test import and basic functionality
python -c "import causers; causers.about()"

# Run smoke test
python examples/basic_regression.py

# Clean up
deactivate
rm -rf test_env
```

## PyPI Publishing Steps

### 1. Prerequisites

- [ ] PyPI account created
- [ ] API token generated (https://pypi.org/manage/account/token/)
- [ ] Token saved securely (not in repository)

### 2. Test PyPI Upload (Recommended First)

```bash
# Upload to Test PyPI first
maturin publish --repository testpypi \
  --username __token__ \
  --password <your-test-pypi-token>

# Test installation from Test PyPI
pip install -i https://test.pypi.org/simple/ causers
```

### 3. Production PyPI Upload

```bash
# Upload to PyPI
maturin publish \
  --username __token__ \
  --password <your-pypi-token>

# Or using maturin with stored credentials
maturin publish
```

### 4. Verify Publication

- [ ] Package visible on PyPI: https://pypi.org/project/causers/
- [ ] Installation works: `pip install causers`
- [ ] All wheel variants uploaded (Linux, macOS, Windows)
- [ ] Documentation links working

## GitHub Release Process

### 1. Create Release Notes

Use the template below for GitHub releases:

```markdown
# causers v0.1.0

## 🎉 Highlights
- First public release
- Linear regression with >3x performance vs NumPy
- 100% test coverage
- Cross-platform support

## 📊 Performance
- 1M rows: ~45ms (requirement: <100ms ✅)

## 📦 Installation
```bash
pip install causers
```

## 📖 Documentation
- [README](https://github.com/causers/causers#readme)
- [CHANGELOG](https://github.com/causers/causers/blob/main/CHANGELOG.md)
- [API Docs](https://causers.readthedocs.io)

## 🔒 Security
- Rating: B+
- No critical vulnerabilities

## ⚠️ Known Issues
- NaN/Inf handling incomplete (see #xxx)
- Memory limits not enforced (see #xxx)

## 📈 What's Next
- v0.2.0: Correlation functions
- v0.3.0: Hypothesis testing

## Assets
- Source code (tar.gz)
- Source code (zip)
- causers-0.1.0-cp38-abi3-linux_x86_64.whl
- causers-0.1.0-cp38-abi3-macosx_10_12_x86_64.whl
- causers-0.1.0-cp38-abi3-macosx_11_0_arm64.whl
- causers-0.1.0-cp38-abi3-win_amd64.whl
```

### 2. Create GitHub Release

1. Go to: https://github.com/causers/causers/releases/new
2. Choose tag: `v0.1.0`
3. Release title: `causers v0.1.0 - Initial Release`
4. Paste release notes
5. Upload wheel files from `target/wheels/`
6. Check "Set as latest release"
7. Publish release

## Post-Release Tasks

### 1. Verification

- [ ] Installation from PyPI works
  ```bash
  pip install causers==0.1.0
  python -c "import causers; print(causers.__version__)"
  ```

- [ ] GitHub release page looks correct
- [ ] Documentation site updated (if applicable)
- [ ] CI/CD badges showing green

### 2. Announcements

- [ ] Update project status in README if needed
- [ ] Post on relevant forums/communities
- [ ] Update any dependent projects

### 3. Monitoring

- [ ] Monitor PyPI download statistics
- [ ] Check for installation issues on issue tracker
- [ ] Monitor for security advisories

## Rollback Plan

If critical issues are discovered post-release:

### 1. Yank from PyPI (if necessary)

```bash
# This marks the version as "yanked" but doesn't delete it
pip install twine
twine yank causers==0.1.0
```

### 2. Fix Issues

1. Create hotfix branch
2. Fix critical issues
3. Bump patch version (e.g., 0.1.1)
4. Follow emergency release process

### 3. Emergency Release Process

- [ ] Fix critical issue
- [ ] Run minimal test suite
- [ ] Update CHANGELOG with hotfix notes
- [ ] Build and publish patch version
- [ ] Notify users of the fix

## Release Schedule

- **Major releases** (x.0.0): Annually, with breaking changes
- **Minor releases** (0.x.0): Quarterly, with new features
- **Patch releases** (0.1.x): As needed for bug fixes
- **Security releases**: Within 72 hours of disclosure

## Automation Opportunities

Consider automating these steps in CI/CD:

1. Version bumping with commit hooks
2. Changelog generation from commit messages
3. Multi-platform wheel building
4. Automated PyPI publishing on tag
5. GitHub release creation
6. Documentation deployment

## Checklist Template

Copy this for each release:

```markdown
## Release v0.1.0 Checklist

**Release Manager**: [Name]
**Target Date**: 2025-12-21
**Status**: In Progress

### Pre-Release
- [ ] Code quality checks pass
- [ ] Performance requirements met
- [ ] Security review complete
- [ ] Documentation updated
- [ ] Platform testing complete

### Release
- [ ] Version bumped
- [ ] Artifacts built
- [ ] Test PyPI upload successful
- [ ] Production PyPI upload successful
- [ ] GitHub release created

### Post-Release
- [ ] Installation verified
- [ ] Announcements sent
- [ ] Monitoring in place

**Notes**: [Any special considerations]
```

## Contact Information

- **Release Manager**: [Your Name]
- **Security Issues**: security@example.com
- **General Support**: https://github.com/causers/causers/issues

---

Last Updated: 2025-12-21
Next Review: Before v0.2.0 release