# Manufacturer-image label sanity check (full-scale, not sampled)

**Corrected framing (2026-10-12):** this was originally called a "provenance
audit" and summarized as "clean" -- that overclaims what was actually
checked. What follows verifies only that the *set* of manufacturer domain
labels attached to images is not fabricated or orphaned. It does **NOT**
verify that any individual image is correctly attributed to its labeled
manufacturer -- that would require an image-level URL or scrape-log mapping
tying each specific image file to the specific page it was scraped from,
which does not exist for the official release (confirmed below: the
official parquet's path field carries no embedded URL/domain at all). A
label-set check cannot substitute for that; it only rules out one failure
mode (a label pointing to a directory entry that doesn't exist), not
mislabeling between two entries that both exist. The planned 100-200-image
human spot-check (see `QUALITY_AUDIT_PROTOCOL.md`) is what can actually add
image-level content evidence -- it has not been run yet.

## Official HF release (28,986 images / 4,484 manufacturers)
The official parquet's `image_path` field is a bare filename (no embedded
domain) — the manufacturer attribution comes solely from the `manufacturer`
class-label field, decoded via the README's `class_label` map to a domain
string. Checked: do all 4,484 unique domain labels used in the release
correspond to a real `Web` value in the source directory CSV?

- **Unique manufacturer domain labels in the release: 4,484**
- **Domain labels NOT found as any `Web` value in the source directory: 0**

This confirms the label set is internally valid (no fabricated/orphaned
domain strings) — it does **not** confirm any single image-to-manufacturer
assignment is correct. No image-level ground truth exists to check that
against for the official release.

## Legacy local subset (1,342 images / 433 manufacturers, dev-era leftover)
This subset's `image_path` field DOES embed the scraped domain directly
(e.g. `filtered_product_img/www.allmetalsinc.com/...`), so it allows an
independent cross-check: does the embedded domain match the `Web` field
recorded for that image's assigned company `Name`?

- **Unique (Name, path-domain) pairs checked: 434**
- **Mismatches: 1** — `"ACO Contactors注essages:"` (a corrupted/garbled
  `Name` string in the source CSV itself — a text-encoding artifact, not a
  case of an image genuinely linked to the wrong manufacturer).

## What this does and doesn't establish
Both the official release and the legacy subset pass a **label-sanity**
check at full population scale: no fabricated or orphaned manufacturer
labels, and (for the legacy subset only, 433/4,484 = 9.7% of image-bearing
manufacturers) no embedded-domain-vs-assigned-company mismatch beyond one
corrupted source string. This rules out gross mislabeling at the
dataset-construction level. It does **not** establish that each image is
genuinely sourced from the manufacturer it's attached to -- that claim would
need image-level URL/scrape-log evidence, which isn't available for the
90.3% of image-bearing manufacturers only covered by the official release's
bare-filename path field. The one confirmed content-quality issue found this
round (`apaa.wildapricot.org`, an arts association, carrying personal member
photos rather than product/facility images — see `audit_edge_fidelity_results.md`
§2) was found by inspecting that specific flagged row, not by this check,
and is a reminder that label-sanity passing does not mean image content is
necessarily on-topic. The planned human spot-check is the right tool for
that remaining question.
