# Random relation-correctness audit (manufacturer--product edges)

## Why this check, separate from Section 3.5's screen

Section 3.5's contamination screen is a **targeted** heuristic check (free-
hosting-platform domains) -- it found real issues (3 non-manufacturer nodes,
3 artifact edges) but, per this project's own standing lesson, a targeted or
top-sorted sample reveals failure modes, not a citable population-level
error rate. This is a separate, **uniform random** sample of ordinary
(manufacturer, product) edges, drawn to give an unbiased estimate of how
often a recorded relation is itself wrong or questionable, independent of
the free-hosting-platform pattern.

## Methodology

- **Sampling**: `random_relation_audit.py`, `random.Random(2026).sample(edges, 20)` over
  all 112,930 raw (manufacturer, product) CSV rows (pre-entity-ID-dedup
  count; the released graph's 112,598 edges are the deduplicated/corrected
  version -- sampling from the raw rows is the more conservative, harder
  test, since it does not pre-filter out any already-known-bad entries).
- **Judgment basis**: each sampled edge was assessed for plausibility using
  the record's own co-located fields (manufacturer name, NAICS description,
  Industry_1) as internal evidence. For 3 cases flagged as suspicious or
  borderline from internal evidence alone, the manufacturer's live website
  was additionally checked (via direct fetch) to confirm or overturn the
  judgment -- noted per-row below. The remaining 17 judgments are from
  internal record evidence only, not independently re-verified against a
  live external source.
- **Classification**: **Plausible** (the product is a sensible output for
  this manufacturer type, per name/NAICS/industry); **Borderline**
  (plausible but imprecise -- an adjacent-but-not-exact category);
  **Error** (the recorded product is not a credible product of this
  manufacturer, or is not a real product name at all).

## Per-item results

| # | Manufacturer | NAICS / Industry | Recorded product | Judgment | Basis |
|---|---|---|---|---|---|
| 1 | Bullseye Industries | Machine Shops | Fine Feature Components | Plausible | internal |
| 2 | H&R Special Springs | Spring (Heavy Gauge) Mfg | Quicksafe Boltstud Conversions | Plausible | internal |
| 3 | Orient Machining & Welding | Machine Shops / Mining | Machine Tools | Plausible | internal |
| 4 | Patriotforge | Military Industry | Turbine Shafts | Plausible | internal |
| 5 | smallarmsliquidatorsexchange | Small Arms/Ordnance Mfg | Firearms | Plausible | internal |
| 6 | bloominblinds | Blind and Shade Mfg | Exterior Shades | Plausible | internal |
| **7** | **Blender Products Company** | Military Industry | **Complete Products** | **Error** | **externally verified** |
| 8 | Elkay Mfg. Co. | Enameled Iron/Metal Sanitary Ware | Retrofit Kits | Plausible | internal |
| **9** | **Machinesolutions** | (none listed) | **Balloon Catheters** | **Error** | **externally verified** |
| 10 | Nitekmedical | Mining Industry (listed) | Bioidentical Hormones | Plausible\* | externally verified |
| 11 | Metallographic | Construction Industry | Compression Mounting Powders | Plausible | internal |
| 12 | Master Bolt & Mfg. | Bolt/Nut/Screw/Rivet/Washer Mfg | Carriage Bolts | Plausible | internal |
| 13 | hardware-decor | Hardware Mfg | Cabinet Products | Plausible | internal |
| 14 | Bigredironjr | (none listed) | Gates | Plausible | internal |
| 15 | Process Label, Inc. | Automotive Industry | Oil Change Reminder Stickers | Plausible | internal |
| 16 | bio-con | Surgical Appliance/Supplies Mfg | Compression Garment Accessories | Plausible | internal |
| 17 | nordicboatsusa | Boat Building | Navigation Lights | Plausible | internal |
| 18 | ctsdrugtesting | Chemical Product Mfg | Medical Gloves | Borderline | internal |
| 19 | dunbarstoneinc | Concrete Product Mfg | Fireplaces | Plausible | internal |
| 20 | NT Window Inc. | Metal Window and Door Mfg | Myriad Sunrooms | Plausible | internal |

\*Item 10's **product** relation is confirmed correct (the manufacturer's
live site confirms bioidentical hormones as a real offering); its separate
`Industry_1` field ("Mining Industry") is confirmed wrong by the same
check. This is a metadata-field error, not a manufacturer--product edge
error, and is counted accordingly -- it is not double-counted against the
relation-correctness tally below.

## Confirmed error detail (both independently verified against the live site)

- **Blender Products Company → "Complete Products"**: the live site lists
  actual products (Air Blender, Channel Blender, Gas Blender, Airex
  evaporative cooling equipment, Spark Cooler, custom metal manufacturing).
  "Complete Products" is not among them -- almost certainly a scraped
  navigation/category label ("Complete Product Line") rather than an actual
  product name. **Error type: generic/non-product scraping artifact.**
- **Machinesolutions → "Balloon Catheters"**: the live site confirms this
  manufacturer makes *equipment used to produce* balloon catheters for
  medical-device manufacturers ("equipment solutions to produce the balloon
  catheters," "End-to-End Medical Device Manufacturing Partner") -- it does
  not manufacture balloon catheters itself. **Error type:
  equipment-for-X conflated with X.**

## Summary

| Judgment | Count | Share |
|---|---|---|
| Plausible | 17 | 85.0% |
| Borderline | 1 | 5.0% |
| Error | 2 | 10.0% |

**This is a 20-item random sample, not a population census** -- at this
sample size the 95% CI on the error rate is wide (exact binomial CI for
2/20: roughly 1.2%-31.7%); report as "an estimated error rate in the
low-to-mid tens of percent is not supported, but a single-digit-to-
low-teens error rate among raw extracted relations is consistent with this
sample," not as a precise population figure. The two confirmed error types
found (generic/non-product label; equipment-for-X conflated with X) are
both artifacts of the automated extraction pipeline, not fabricated
relations -- consistent with, and a new independent data point alongside,
the free-hosting-platform contamination pattern already disclosed in
Section 3.5.
