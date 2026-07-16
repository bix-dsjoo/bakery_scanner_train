# Detector Origin-Aware Split Design

## Goal

Keep the existing deterministic, leakage-safe detector split while ensuring that a
mixed real-plus-synthetic dataset has both train and validation real scene groups
when the real input contains at least two independent scene groups.  This change
does not train or infer a detector and does not modify input datasets or synthetic
runs.

## Problem

The current split optimizer builds leakage-connected components across every
sample, then selects the subset whose image count is nearest the requested
validation fraction.  A synthetic run can form one large component because its
scenes reuse source objects and backgrounds.  With `base_seed42`, that component
contains 100 images and the three real scene groups contain nine images.  The
global optimizer therefore assigns all real scenes to validation and all synthetic
scenes to train.

The assignment is leakage-safe but leaves no real scene in training, which makes
the assembled detector dataset unsuitable as a mixed-source training baseline.

## Decision

Build global leakage components first, then optimize the origin groups without
ever splitting a component:

- All samples first use one resource-component algorithm.  This preserves
  cross-origin safety: a real and synthetic sample sharing an image hash is one
  inseparable component.
- Components containing real samples form the real group.  Synthetic-only
  components form the synthetic group.  `real-scene:<id>` remains a resource, so
  every `scene_e/m/h_<id>` group remains in one split.
- Synthetic source objects, backgrounds, and image hashes remain component
  resources, so they cannot cross splits.
- Both origin subsets must produce non-empty train and validation assignments
  when that origin has at least two independent components.  An origin with one
  component is assigned to train; it cannot safely supply validation data.
- The requested validation fraction is applied within each origin that can be
  split.  The seeded component ordering retains deterministic selection.
- A dataset remains invalid if every input sample belongs to one leakage
  component, because neither split can be made safely.  If no origin group has
  multiple components, the existing global component assignment is used as a
  deterministic safe fallback.

For the checked-in `base_seed42` data and seed 42, the 100-scene synthetic
component remains in train.  The three real groups are split as two groups (six
images) into train and one group (three images) into validation.

## Interfaces and Manifest

`_split_samples(samples, validation_fraction, seed)` remains the internal
assignment interface.  Its behavior becomes origin-aware.  The public CLI and
manifest schema remain unchanged: `config.seed` and `config.validation_fraction`
are sufficient for independent validation to recompute the split exactly.

The validator continues to reconstruct the input samples, recalculate
assignments, and reject any divergence, source mutation, output mutation, invalid
bbox, or cross-split resource reuse.

## Error Handling

- Fewer than two total samples remains an error.
- A dataset with only one leakage component remains an error.
- A one-component origin is train-only rather than an error if another origin can
  safely populate validation.
- If no safe validation assignment exists after applying these rules, fail before
  publishing an output run.

## Tests and Verification

Add regression coverage for a large single synthetic component plus multiple real
scene groups.  It must prove that real data appears in both splits, real scene
groups stay intact, all synthetic data stays in one split, and validation of the
manifest reproduces the assignment.  Retain existing impossible-split and seeded
determinism coverage.

Run the focused detector tests, generate a new small real-data detector run from
`base_seed42`, validate it through the independent CLI, and run the full pytest
suite before publishing the PR update.
