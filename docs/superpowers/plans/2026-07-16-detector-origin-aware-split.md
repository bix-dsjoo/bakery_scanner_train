# Detector Origin-Aware Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Split detector data by origin so real scenes occur in both train and validation when possible, without provenance leakage.

**Architecture:** Retain _split_samples(samples, validation_fraction, seed) as the single interface used by generation and validation. Construct resource-connected components per origin, select validation components deterministically per origin, and merge assignments. A one-component origin is train-only; a dataset with no safe validation component fails.

**Tech Stack:** Python 3.11, random, Pillow, pytest 9, bakery-detector-data CLI.

## Global Constraints

- Never read datasets/base/test or datasets/incremental/test as training inputs.
- Preserve each real scene_e/m/h_<scene-id> group in one split.
- Preserve every reused synthetic source object, background, and source image in one split.
- Preserve original COCO category_id solely as provenance; output COCO has only bread.
- Do not change source data, COCO, class_registry, or synthetic runs.
- Write only under datasets/derived/detector/ using atomic publishing.
- Validation must recalculate the seeded assignment and reject divergence.

---

## File Structure

- Modify: src/bakery_scanner/detector_dataset.py — leakage components and assignment.
- Modify: tests/test_detector_dataset.py — mixed-origin regression.
- Modify: README.md — policy wording and working CLI commands.
- Create: docs/superpowers/specs/2026-07-16-detector-origin-aware-split-design.md — approved and committed.
- Create: docs/superpowers/plans/2026-07-16-detector-origin-aware-split.md — this plan.

### Task 1: Capture the mixed-origin regression

**Files:**

- Modify: tests/test_detector_dataset.py after test_split_is_deterministic_and_keeps_real_and_synthetic_resources_together.
- Consumes: build_detector_dataset and validate_detector_dataset.
- Produces: a regression proving real samples appear in both splits while synthetic provenance stays intact.

- [ ] **Step 1: Write the failing test**

    def test_split_keeps_real_scene_groups_in_both_splits_when_synthetic_is_indivisible(
        detector_inputs: Path,
    ) -> None:
        _make_real_images_unique(detector_inputs)
        report = build_detector_dataset(
            detector_inputs,
            "input",
            "origin-aware",
            DetectorDatasetConfig(seed=42, validation_fraction=0.2),
        )
        samples = _load_json(report.manifest_path)["samples"]
        real_samples = [item for item in samples if item["origin"] == "real"]
        synthetic_samples = [item for item in samples if item["origin"] == "synthetic"]
        assert {item["split"] for item in real_samples} == {"train", "validation"}
        assert len({item["split"] for item in synthetic_samples}) == 1

        real_splits_by_scene: dict[str, set[str]] = {}
        for item in real_samples:
            real_splits_by_scene.setdefault(item["provenance"]["scene_id"], set()).add(
                item["split"]
            )
        assert all(len(splits) == 1 for splits in real_splits_by_scene.values())
        assert validate_detector_dataset(detector_inputs, "origin-aware").image_count == 9

- [ ] **Step 2: Verify RED**

Run: python -m pytest tests/test_detector_dataset.py::test_split_keeps_real_scene_groups_in_both_splits_when_synthetic_is_indivisible -v

Expected: FAIL because the current global optimizer assigns every real sample to validation.

- [ ] **Step 3: Commit the red test**

    git add tests/test_detector_dataset.py
    git commit -m "test: cover origin-aware detector split"

### Task 2: Implement origin-aware assignment

**Files:**

- Modify: src/bakery_scanner/detector_dataset.py, replacing the current _split_samples body.
- Test: the new mixed-origin regression.
- Consumes: list[_Sample], each sample's origin, key, and resources.
- Produces: unchanged _split_samples(samples, validation_fraction, seed) -> dict[str, str].

- [ ] **Step 1: Add resource-component and component-assignment helpers**

    def _leakage_components(samples: list[_Sample]) -> list[list[_Sample]]:
        parents = list(range(len(samples)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        owners: dict[str, int] = {}
        for index, sample in enumerate(samples):
            for resource in sorted(sample.resources):
                union(index, owners.setdefault(resource, index))
        grouped: dict[int, list[_Sample]] = {}
        for index, sample in enumerate(samples):
            grouped.setdefault(find(index), []).append(sample)
        return sorted(
            (sorted(group, key=lambda sample: sample.key) for group in grouped.values()),
            key=lambda group: tuple(sample.key for sample in group),
        )

    def _assign_origin_components(
        components: list[list[_Sample]], validation_fraction: float, seed: int
    ) -> dict[str, str]:
        if len(components) == 1:
            return {sample.key: "train" for sample in components[0]}
        shuffled = list(components)
        random.Random(seed).shuffle(shuffled)
        choices: dict[int, tuple[int, ...]] = {0: ()}
        for component_index, component in enumerate(shuffled):
            for count, selected in list(choices.items()):
                choices.setdefault(count + len(component), (*selected, component_index))
        total = sum(len(component) for component in shuffled)
        valid_counts = [count for count in choices if 0 < count < total]
        if not valid_counts:
            raise DataValidationError("safe train/validation split is impossible")
        target = total * validation_fraction
        selected = set(
            choices[min(valid_counts, key=lambda count: (abs(count - target), count))]
        )
        return {
            sample.key: ("validation" if index in selected else "train")
            for index, component in enumerate(shuffled)
            for sample in component
        }

- [ ] **Step 2: Replace _split_samples**

    def _split_samples(
        samples: list[_Sample], validation_fraction: float, seed: int
    ) -> dict[str, str]:
        if len(samples) < 2:
            raise DataValidationError("safe train/validation split requires at least two images")
        by_origin: dict[str, list[_Sample]] = {}
        for sample in samples:
            by_origin.setdefault(sample.origin, []).append(sample)
        components = {
            origin: _leakage_components(origin_samples)
            for origin, origin_samples in sorted(by_origin.items())
        }
        if sum(len(groups) for groups in components.values()) < 2:
            raise DataValidationError(
                "safe train/validation split is impossible because all images share leakage resources"
            )
        assignments: dict[str, str] = {}
        for offset, groups in enumerate(components.values()):
            assignments.update(
                _assign_origin_components(groups, validation_fraction, seed + offset)
            )
        if "validation" not in assignments.values():
            raise DataValidationError("safe train/validation split is impossible")
        return assignments

- [ ] **Step 3: Verify GREEN**

Run: python -m pytest tests/test_detector_dataset.py -v

Expected: PASS, including the new regression, impossible-split, deterministic, and validator split-recalculation tests.

- [ ] **Step 4: Commit**

    git add src/bakery_scanner/detector_dataset.py tests/test_detector_dataset.py
    git commit -m "fix: balance detector split origins"

### Task 3: Document and exercise the CLI

**Files:**

- Modify: README.md detector split-policy paragraph and command example.
- Test: actual datasets/derived/synthetic/base_seed42 input.
- Consumes: bakery-detector-data generate and bakery-detector-data validate.
- Produces: documented, independently validated output below datasets/derived/detector/.

- [ ] **Step 1: Replace the origin-balance limitation with this policy**

    Split assignment is deterministic from seed and is calculated separately for
    real and synthetic origins. Real scene_e/m/h groups remain intact, and synthetic
    source objects, backgrounds, and source-image hashes remain in one split. An
    origin with one leakage component is train-only; when real scene groups are
    independently splittable, real scenes appear in both train and validation.

Add actual commands:

    bakery-detector-data generate `
      --dataset-root datasets `
      --synthetic-run base_seed42 `
      --run-name base_seed42_detector_origin_aware `
      --seed 42 `
      --validation-fraction 0.2

    bakery-detector-data validate `
      --dataset-root datasets `
      --run-name base_seed42_detector_origin_aware

- [ ] **Step 2: Generate and independently validate**

Run: python -m bakery_scanner.detector_cli generate --dataset-root datasets --synthetic-run base_seed42 --run-name base_seed42_detector_origin_aware --seed 42 --validation-fraction 0.2

Run: python -m bakery_scanner.detector_cli validate --dataset-root datasets --run-name base_seed42_detector_origin_aware

Expected: both commands return JSON status "ok"; real samples occur in both splits, the synthetic component is train-only, and source files remain unchanged.

- [ ] **Step 3: Run full checks**

Run: python -m pytest

Run: git diff --check; git status -sb

Expected: zero pytest failures, no whitespace errors, and generated detector output remains ignored.

- [ ] **Step 4: Commit**

    git add README.md docs/superpowers/plans/2026-07-16-detector-origin-aware-split.md
    git commit -m "docs: describe origin-aware detector split"

### Task 4: Publish and mark PR #4 ready

**Files:**

- Remote update only: PR #4, branch agent/detector-dataset-foundation.
- Consumes: verified commits and scope-checked worktree.
- Produces: pushed branch and non-draft PR recording the root cause and validation.

- [ ] **Step 1: Push after checks**

Run: git push origin agent/detector-dataset-foundation

Expected: remote branch advances with design, implementation, tests, plan, and README.

- [ ] **Step 2: Update description and mark ready**

Record the global component-count root cause, origin-aware policy, actual CLI output, and full pytest result. Run gh pr ready 4 only after pushing.

- [ ] **Step 3: Confirm remote state**

Run: gh pr view 4 --json state,isDraft,mergeStateStatus,url,title

Expected: state OPEN, isDraft false, and PR URL https://github.com/bix-dsjoo/bakery_scanner_train/pull/4.

## Plan Self-Review

- Spec coverage: Task 1 proves the regression; Task 2 implements deterministic origin-aware components and errors; Task 3 documents and validates the CLI; Task 4 publishes.
- Placeholder scan: no deferred implementation language or unspecified tests remain.
- Type consistency: every task retains _split_samples(list[_Sample], float, int) -> dict[str, str], used by generation and validation.

## Execution Adjustment

The implementation first builds global leakage components, then classifies each
whole component as real-containing or synthetic-only for origin-aware selection.
This preserves the existing error case where an image hash is shared across real
and synthetic inputs: that resource may never be separated merely because its
samples have different origins.  If neither origin group has multiple components,
the implementation falls back to the deterministic global component selection so
that existing safe datasets still receive a non-empty validation split.
