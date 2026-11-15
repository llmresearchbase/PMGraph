# PMGraph — Dataset Card

**Authors**: Yunqing Li and Zixiang Tang (co-first authors), Lenovo ·
**Contact**: yli59@lenovo.com · **License**: CC BY 4.0

**Data**: https://huggingface.co/datasets/shawntzx/PMGraph (revision `v2`) · **Code**: https://github.com/llmresearchbase/PMGraph

## Summary
A manufacturer–product–attribute–image graph for 8,900 US manufacturing SMEs
(14,007 total directory records), built on the published C-MAG pipeline, with
a corrected entity-identity scheme and two released, leakage-checked
benchmark tasks. See the accompanying paper for full detail; this card gives
the facts a reuser needs without reading the paper end to end.

## Composition
- **Manufacturers**: 8,900 graph nodes (14,007 total directory records; most
  of the remainder never listed a product). 3 of the 8,900 are confirmed
  non-manufacturers from the upstream directory — flagged, not removed; see
  "Known issues" below.
- **Products**: 72,789 free-text catalog strings, 86.6% listed by exactly one
  manufacturer.
- **Edges**: 112,598 manufacturer–product edges.
- **Attributes**: Material_Capability (89.8% coverage), Process_Capability
  (33.9%), Certification, Industry, NAICS code, employee count, geography.
- **Images**: 4,484 manufacturers have website imagery (50.4% graph-relative
  coverage, observed under the current scrape+filter pipeline — not verified
  to be the true absence rate; see paper Section 3.2). **Not released as
  pixel data** — see "Image release format" below.

## Provenance
Merged from SUDOKN's SC-KG (NSF Proto-OKN, led by Farhad Ameri, ASU) and the
manufacturer directory and ontology of Li and Starly (2024). Both upstream
sources trace back to this submission's own author(s) rather than being
third-party data acquired independently: Ameri is a co-author of C-MAG (the
paper that first merged SUDOKN into PMGraph), and this submission's author
is a co-author of Li and Starly (2024) itself. A direct written record (an
email thread with Ameri, Feb–Mar 2026) confirms he personally supplied the
exact source file (`Cleaned_Master_CSV_v3.xlsx`, matching this release's own
`Cleaned_Master_CSV_v3.csv`) for this explicitly-stated benchmark/dataset-
paper purpose. See `PROVENANCE_AND_ETHICS.md` for the full authorization-
chain check, including what was and wasn't independently confirmable from
public license records (SUDOKN's own GitHub org: no license set; the Li &
Starly 2024 arXiv preprint: CC BY-NC-ND 4.0, the author's own prior choice
for that one paper; the prior HuggingFace release: no license set; C-MAG's
own arXiv paper: CC BY 4.0). Lenovo IP clearance is confirmed internally.
Zixiang Tang (co-first-author, owner of the existing public HuggingFace
listing) will personally handle republishing the dataset and GitHub repo.

## Image release format
Images are released as **derived Jina-CLIP-v1 embeddings (32-dim SVD-
compressed, and 768-dim raw) plus each manufacturer's website domain URL —
not redistributed pixel content.** This substantially reduces copyright/
redistribution exposure relative to hosting a copy of scraped images.
Tradeoffs, stated plainly:
- The URL is domain-level (e.g. `https://example-mfr.com`), not a per-image
  deep link — no per-image source URL was ever recorded during collection.
- A future user cannot re-encode the same images with a different vision
  encoder without independently re-scraping the manufacturer's site, which
  may have changed or gone offline since collection.
- The released embeddings are the frozen, versioned artifact the benchmark
  numbers in the paper depend on — they will not change even if the live
  website does.

## Benchmark tasks
1. **Manufacturer–product relation completion** — transductive edge
   completion (`RandomLinkSplit`, all manufacturers visible in every split;
   see paper Section 4.1). Reported under both random and degree-matched
   negative sampling — these give materially different numbers; use whichever
   matches your evaluation question, and report which one you used.
2. **Recorded-material-relation recovery** — Recall@K ranking over a frozen
   56-category vocabulary, manufacturer-level split (paper Section 4.2). The
   underlying label field is **not closed-world**: 17.4%/24.6% text-mention-
   vs-structured-field inconsistency rates were measured before this task was
   designed. Do not use this task's negatives as confirmed "manufacturer
   cannot do X" labels.

## Known issues (disclosed, not silently omitted)
- **3 confirmed non-manufacturer nodes** (a mortgage company, a residential
  community, a hobbyist arts association) inherited from the upstream
  directory, each mis-assigned a manufacturing NAICS code. Flagged by exact
  index in `manufacturer_validity_mask.pt` (indices 0, 5516, 6236); not
  removed from the graph shipped here (would require reindexing — see
  `README.md`).
- **3 extraction-artifact edges** on a 4th, otherwise-legitimate manufacturer
  (`brimleyskateboards`) whose listed "products" are tumblr blog-tag
  categories, not real products. Flagged at exact edge positions in the same
  mask file.
- These are two distinct findings from a targeted heuristic screen (not a
  random sample) — they establish that this failure mode exists, not its
  true population rate.
- **A separate, uniform random sample** of 20 (manufacturer, product) edges
  (not targeted, complementing the screen above) found 17/20 (85%)
  plausible, 1/20 (5%) borderline, 2/20 (10%) confirmed errors — both
  externally verified against the live manufacturer website (a generic
  scraped label mistaken for a product name; an equipment-for-X relation
  mistaken for a direct X relation). See `random_relation_audit_results.md`
  for the full per-item table and methodology; at n=20 this is a wide-
  interval estimate (95% CI for 2/20 ≈ 1–32%), not a precise population
  error rate.

## License
**CC BY 4.0 — confirmed.** Governs only PMGraph's own derived artifacts
(the graph, features, embeddings, splits, code) as compiled and released by
this submission — it does not attach a license to the two upstream sources
themselves beyond what their own authors/maintainers have already granted.
The Li & Starly (2024) arXiv preprint's CC BY-NC-ND tag was the author's own
prior choice for that one paper, not a third-party restriction, and the
constructed graph here is a separate artifact from that preprint. Full text
in `LICENSE`.

## Authors / contact
**Yunqing Li** (first author) and **Zixiang Tang** (co-first-author),
both Lenovo. Contact: yli59@lenovo.com

## Citation
The accompanying paper is under review. Until it has a published entry,
please cite the dataset release:
```
@misc{li2026pmgraph,
  title        = {PMGraph: A Multimodal Manufacturer--Product Knowledge Graph
                  with Audited Relation-Completion Benchmarks (v2)},
  author       = {Li, Yunqing and Tang, Zixiang},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/shawntzx/PMGraph}},
  note         = {Dataset version 2. Code: \url{https://github.com/llmresearchbase/PMGraph}}
}
```
