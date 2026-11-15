# PMGraph — Provenance and Ethics Statement

This document exists to answer, in one place, the two questions a WWW
Resource-track reviewer (or any reuser) is most likely to ask first: where
did this data come from and who authorized its use, and what are the
privacy/ethics implications of releasing it. Every claim below is either
directly checked against the released files or sourced to a publicly
verifiable record (cited inline); nothing here is asserted from memory.

## 1. Data provenance and authorization

PMGraph is a revised, audited version of the resource first released
alongside **C-MAG: Cascade Multimodal Attributed Graphs for Supply Chain
Link Prediction** (Li, Tang, Zhuang, Yang, Ameri, Zhang; KDD 2025 Workshop
on AI for Supply Chain; arXiv:2508.08071). It is built from two upstream
sources, both merged by this submission's own author(s), not acquired
third-party:

- **SUDOKN** (Supply and Demand Open Knowledge Network), an NSF Proto-OKN
  project led by Farhad Ameri (Arizona State University). **Farhad Ameri is
  a co-author of C-MAG** — the paper that first merged SUDOKN's data into
  PMGraph — which is direct, publicly verifiable evidence that this use of
  SUDOKN's data was made with the SUDOKN project lead's own participation
  and knowledge, not without the source team's awareness.
- **The manufacturer directory and ontology of Li and Starly (2024)**,
  "Building a Knowledge Graph to Enrich ChatGPT Responses in Manufacturing
  Service Discovery" (*J. Ind. Inf. Integr.* 40, 100612;
  arXiv:2404.06571). **This submission's author is a co-author of that
  paper** (Yunqing Li, with Binil Starly) — this upstream source is the
  author's own prior published work, not a third party's data being reused
  without the creator's involvement.

**What is and isn't independently confirmed by public records (checked this
round, not assumed):**
- SUDOKN's own GitHub organization (`github.com/SUDOKN`, repository
  `graph` — "Contains the latest data corresponding to the latest version
  of the graph") has **no license field set** in the GitHub API response,
  checked directly. SUDOKN being an NSF-funded, publicly described project
  is not the same as having a stated redistribution license.
- The Li and Starly (2024) arXiv preprint (2404.06571) is tagged
  **CC BY-NC-ND 4.0** (Non-Commercial, No-Derivatives) — checked directly
  against the arXiv listing. Standard practice is that a CC license on a
  *paper* (its text/figures as a scholarly work) does not by itself govern
  a separately constructed *dataset* merely described by that paper, and as
  a co-author the submission's author retains her own underlying rights to
  her own contribution independent of the license she chose for that one
  preprint — but this NC-ND tag is a real, found fact that should inform
  the license choice for this release, not be read past. **This specific
  point — whether NC-ND on that preprint constrains the redistribution
  terms of the derived graph/dataset — is a scope-of-use question the
  author needs to confirm, not one resolved by this check.**
- The existing public HuggingFace dataset (`shawntzx/PMGraph`, the prior
  C-MAG-era release) has **no `license` field set**, checked directly
  against the HuggingFace API. There is no existing data license this
  submission can simply "continue" — none was set on the prior release.
- The C-MAG arXiv paper itself (2508.08071) is tagged **CC BY 4.0** —
  checked directly. This is the clearest, most permissive license among the
  three upstream artifacts checked, but again covers the paper, not
  automatically the dataset.

**Direct written record (provided by the author, verified against the
released data, not just referenced)**: an email thread with Farhad Ameri
("ManufactureNC Dataset Request," Feb 27 – Mar 22, 2026) documents Ameri
personally providing the source data on repeated request, with the
request's purpose stated explicitly up front ("preparing a manufacturing
intelligence benchmark paper for submission to the NeurIPS Datasets and
Benchmarks Track") and reiterated later in reference to "A Web-Based
Information Extraction Approach for Building Manufacturing Capability
Knowledge Graph and KDD workshop paper." Critically, the file Ameri sent on
Mar 18, 2026 — `Cleaned_Master_CSV_v3.xlsx` — **has the exact filename of
this project's own source file**, `Cleaned_Master_CSV_v3.csv`, confirmed by
direct comparison, not assumed from the name alone. Ameri confirmed on Mar
22 that "the dataset you attached is accurate enough" after being asked
about version differences. Across the full thread, Ameri was told the
explicit benchmark/public-dataset-paper purpose multiple times and
continued supplying data without objection or any stated redistribution
restriction. (A separate file Ameri also shared, `NC-D&B.xlsx`, sourced
from D&B Hoovers — a commercial data provider with its own separate
licensing — was **not** incorporated into `Cleaned_Master_CSV_v3.csv` or
PMGraph; checked by scale alone, D&B's list is ~20,000 NC-only companies
vs. this resource's 14,007 multi-state records, consistent with the
SUDOKN-extraction source, not the D&B file.)

This is a real, good-faith research-collaboration record — not a formal
signed data-use or licensing agreement, and it does not contain an explicit
sentence granting a specific redistribution license. It does establish,
directly and in writing, that the SUDOKN data's use for this benchmark/
dataset-paper purpose was made with the source team's full knowledge and
active cooperation.

**Net position**: the authorization chain is substantially stronger than
"two unrelated third-party sources with no contactable owner" — both
upstream sources trace back to this submission's own author(s), the SUDOKN
lead is a named collaborator on the paper that created PMGraph in the first
place, and there is now a direct written record of him supplying the exact
source file for this exact benchmark-paper purpose. What remains a nuance
(not a blocker) is that the Li & Starly (2024) arXiv preprint's own NC-ND
tag was the author's personal licensing choice for that one paper, made
independently of this release — not a third party's restriction imposed on
her.

**Institutional and execution status (as of this round)**: Lenovo IP
clearance has been confirmed internally by the author. Zixiang Tang
(`shawntzx` on HuggingFace, the existing public listing's owner, and a
C-MAG co-author) is confirmed as co-first-author (both authors Lenovo) and will
personally handle republishing the dataset and the GitHub repository —
resolving the earlier open question of who has write access to the
existing public address.

## 2. What is released, and in what form

- **Structured attributes and graph relations**: manufacturer name, NAICS
  description, industry, employee count, certification, geography,
  Material_Capability / Process_Capability fields, and the
  manufacturer–product edge list. All derived from the two sources above.
- **Website images**: released only as **derived embeddings**
  (32-dim SVD-compressed and 768-dim raw Jina-CLIP-v1 vectors) plus each
  manufacturer's website **domain** (not a per-image URL — none was
  recorded during collection, and no per-image source link exists to
  release). Raw image pixels are **not** redistributed. This is a
  deliberate choice to reduce copyright/redistribution exposure relative to
  hosting a copy of scraped third-party website imagery (Section 6 of the
  paper; `DATA_CARD.md` "Image release format").

## 3. Privacy and human-subjects considerations

- **No personal data about private individuals is collected or released.**
  Every record describes a *business entity* (a manufacturing company) and
  its publicly listed, self-reported business attributes (name, industry,
  certifications, location, product catalog) — the kind of information a
  manufacturer itself publishes on its own public website or business
  directory listing, not information about an identifiable private person.
- **Website images** depict equipment, facilities, or product/process
  photography from company websites, not individuals as the subject of the
  evidence used in this work's tasks; no attempt is made to identify or
  profile people who may incidentally appear in a photo.
- **The human annotation conducted for this submission's data-quality
  audit** (Section 3.4 of the paper) involved two domain-literate
  annotators judging image relevance against manufacturer-level business
  records — it is a data-quality QA exercise, not human-subjects research;
  no data about the annotators themselves is collected or released beyond
  their judgments.
- **Three confirmed non-manufacturer entities** were found inherited from
  the upstream directory (a mortgage company, a residential community, a
  hobbyist arts association) and are flagged, not silently dropped, via
  `manufacturer_validity_mask.pt` (Section 3.5) — disclosed as a data-
  quality issue, not a privacy issue, since these too are business/
  organization-level records, not individuals.

## 4. Intended and unintended use

Per the paper's own framing (Section 1, Table 1): this resource supports
graph-learning, relation-recovery, and fixed-feature multimodal fusion
research on manufacturing-capability prediction. It does **not** support:
vision-encoder benchmarking or pixel-level image research (images are
embeddings-only); validated claims about any specific manufacturer's true,
complete capabilities (the benchmark recovers *recorded* relations, not
verified real-world capability); or re-identification/contact-targeting of
the listed manufacturers for purposes outside research use consistent with
how this directory data is already publicly available.
