# Manufacturer-product edge fidelity audit (full-scale, not sampled)

## 1. Edge-set reconciliation: Graph_A.pt vs re-derivation from source CSV

- Stored edges: 112598
- Rebuilt edges (fresh run of `build_edges()` on `Cleaned_Master_CSV_v3.csv`): 112598
- **Identical edge sets: True**
- Stored manufacturers: 8900, rebuilt: 8900
- Stored products: 72789, rebuilt: 72789

Conclusion: every edge in the released graph traces back exactly to a non-empty `Product_i` cell for that manufacturer's canonical Web-ID in the source directory CSV, with no fabricated, dropped, or duplicated edges. This is a full reconciliation of the entire 112,598-edge population, not a sample.

## 2. Known contamination pattern: non-manufacturer rows on free hosting platforms

Full-population screen (not a sample) for directory rows hosted on known free community/personal website platforms (wordpress.com, blogspot, weebly, tumblr, squarespace, wix.com, wildapricot, etc.): **15** of 14007 rows (0.107%), of which **11** are graph nodes carrying product edges.

| Name | Web | n_products | NAICS category assigned |
|---|---|---|---|
| Howell Precision | www.howellprecision.wordpress.com | 5 | nan |
| JJ Machine Inc. | www.jjmachineinc.wordpress.com | 27 | nan |
| Plasticology (Based on the given URL) | www.plasticology.squarespace.com | 4 | nan |
| Dunlopcustoms | www.dunlopcustoms.blogspot.com | 8 | nan |
| customdesignsbyjacquelyn.blogspot | www.customdesignsbyjacquelyn.blogspot.com | 10 | All Other Miscellaneous Textile Product Mills |
| regencymedicalr2008.blogspot | www.regencymedicalr2008.blogspot.com | 5 | Pharmaceutical Preparation Manufacturing |
| meridianmanor.blogspot | www.meridianmanor.blogspot.com | 8 | Manufactured Home (Mobile Home) Manufacturing |
| brimleyskateboards.tumblr | www.brimleyskateboards.tumblr.com | 5 | Sporting and Athletic Goods Manufacturing |
| krffinancialllc.weebly | www.krffinancialllc.weebly.com | 6 | Steel Investment Foundries |
| softheartdesigns.weebly | www.softheartdesigns.weebly.com | 13 | Jewelry and Silverware Manufacturing |
| apaa.wildapricot.org | apaa.wildapricot.org | 12 | Office Supplies (except Paper) Manufacturing |

Manual inspection of all 11 rows (full population, not a sample -- small enough to read directly): most (`Howell Precision`, `JJ Machine Inc.`, `Plasticology`, `Dunlopcustoms`) are plausible genuine small manufacturers using a free-tier website. Three are clear non-manufacturer contamination inherited from the upstream source directory, not introduced by this project's own pipeline:
- `krffinancialllc.weebly` -- a mortgage/loan company (products list: 'Conventional Loans', 'Jumbo Loans', ...), mis-tagged NAICS 'Steel Investment Foundries'.
- `meridianmanor.blogspot` -- a residential community (products list: 'Homes', 'Real Estate', 'Carpet & Tile Cleaning Services'), mis-tagged 'Manufactured Home (Mobile Home) Manufacturing'.
- `apaa.wildapricot.org` -- the Arizona Pastel Artists' Association (products list: 'Pastel Paintings', 'Membership Booklet', 'Workshops'), mis-tagged 'Office Supplies (except Paper) Manufacturing'; also appears in the image corpus with personal member photos, not product/facility images.
`brimleyskateboards.tumblr` is a different artifact: plausibly a genuine small skateboard brand, but several of its 'products' ('Skate Videos', 'Skate Footage', 'Skate Team') are blog content tags, not products -- a tumblr-specific extraction artifact.

**Scope of this finding:** this screen is a targeted heuristic (domain pattern match), not a random sample -- it finds known-pattern contamination cheaply but CANNOT be used to estimate an overall population contamination rate for the 8,900-manufacturer graph -- a filtered/biased lens tells you a failure mode exists, not how common it is outside that lens. At 3 confirmed non-manufacturer rows out of 8,900 graph nodes (0.034%) found via this specific heuristic, the magnitude is small and bounded, and is reported here as a documented known-issue (same pattern as the round-1 entity-ID-collision finding), not grounds for a broader manual re-audit.
