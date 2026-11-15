# Round 4 architecture-equivalence verification

Requested check: confirm the concat+32-dim fusion architecture has no hidden asymmetry between baseline/real/zeroed conditions, independent of trained weights (no round-4 checkpoint was saved, so this uses a random-but-shared weight set -- a valid instance-level check, since the property being proven is architectural/mathematical, not learned).

## Check 1: weight-transplant equivalence

Same weights (random init, shared across both paths except man_proj's zero-padded extra columns), same underlying information as input, two implementation paths (64-dim baseline_no_image HeteroSAGE vs 97-dim zeroed HeteroSAGE). Max abs output diff: **0.000e+00** (tolerance 1e-05). **PASS: True**

## Check 2: feature-construction equivalence

feat_real vs feat_zeroed on the actual saved feature tensors: text dims and has_image flag are byte-identical; only the image slice differs (as intended -- that's the manipulated variable). **PASS: True**

## Conclusion

Both checks pass. The REAL-vs-ZEROED comparison in round 4's 3-seed result (ARCHIVE_round4_material_recovery_pilot.md) is architecturally sound: the zeroed condition is a true same-structure control (not an architecture with a hidden computational difference), and the only thing that varies between REAL and ZEROED is the image content itself. This does not re-run or change any reported number -- it closes the verification gap noted in USAGE_EVIDENCE_ONEPAGE.md ("REAL≈ZEROED landing close is an observation, not a verification -- round 4 never repeated round 2's weight-transplant check").
