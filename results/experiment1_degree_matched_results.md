# Experiment 1: degree-matched negative-sampling robustness check

**SUPERSEDED NOTICE (round-22 correction): every `textsim_AUC` value below
is INVALID** (same bug as `experiment1_baseline_results.md` -- cosine
similarity between two independently-fit SVD spaces). Use
`experiment1_textsim_shared_space_results.md` instead. `SAGE_AUC`/`degree_AUC`
are unaffected. Separately: `degree_AUC`'s printed "0.5000" is a 4-decimal
rounding, not an exact 0.5 -- `experiment1_degree_baseline_precision_check.py`
finds 0.500010 at 6-decimal precision (the 2.41% non-exact fallback matches
keep it from being exactly 0.5).

Same trained SAGE checkpoint, NOT retrained. Only the negative half of the test set is reconstructed to match each positive's train-degree.

seed=1: exact=10943 (0.9719)  fallback=271  unmatched=45  total_match_rate(incl.fallback)=0.9960

seed=1 overall    [degree-matched]  n_pos=11214 n_neg=11214 SAGE_AUC=0.5451 SAGE_PRAUC=0.5316 degree_AUC=0.5000 textsim_AUC=0.5069  
seed=1 overall    [random-neg, same 11214+45-excluded positives]  n_pos=11214 n_neg=11259 SAGE_AUC=0.6012 SAGE_PRAUC=0.5805 degree_AUC=0.4154 textsim_AUC=0.5106  
seed=1 singleton  [degree-matched]  n_pos=6242 n_neg=6242 SAGE_AUC=0.5783 SAGE_PRAUC=0.5451 degree_AUC=0.5000 textsim_AUC=0.5022  
seed=1 singleton  [random-neg, same 6242+45-excluded positives]  n_pos=6242 n_neg=9799 SAGE_AUC=0.7346 SAGE_PRAUC=0.5660 degree_AUC=0.0769 textsim_AUC=0.5070  

seed=2: exact=10943 (0.9719)  fallback=271  unmatched=45  total_match_rate(incl.fallback)=0.9960

seed=2 overall    [degree-matched]  n_pos=11214 n_neg=11214 SAGE_AUC=0.5452 SAGE_PRAUC=0.5313 degree_AUC=0.5000 textsim_AUC=0.5056  
seed=2 overall    [random-neg, same 11214+45-excluded positives]  n_pos=11214 n_neg=11259 SAGE_AUC=0.6012 SAGE_PRAUC=0.5805 degree_AUC=0.4154 textsim_AUC=0.5106  
seed=2 singleton  [degree-matched]  n_pos=6242 n_neg=6242 SAGE_AUC=0.5769 SAGE_PRAUC=0.5445 degree_AUC=0.5000 textsim_AUC=0.5012  
seed=2 singleton  [random-neg, same 6242+45-excluded positives]  n_pos=6242 n_neg=9799 SAGE_AUC=0.7346 SAGE_PRAUC=0.5660 degree_AUC=0.0769 textsim_AUC=0.5070  

seed=3: exact=10943 (0.9719)  fallback=271  unmatched=45  total_match_rate(incl.fallback)=0.9960

seed=3 overall    [degree-matched]  n_pos=11214 n_neg=11214 SAGE_AUC=0.5438 SAGE_PRAUC=0.5304 degree_AUC=0.5000 textsim_AUC=0.5069  
seed=3 overall    [random-neg, same 11214+45-excluded positives]  n_pos=11214 n_neg=11259 SAGE_AUC=0.6012 SAGE_PRAUC=0.5805 degree_AUC=0.4154 textsim_AUC=0.5106  
seed=3 singleton  [degree-matched]  n_pos=6242 n_neg=6242 SAGE_AUC=0.5780 SAGE_PRAUC=0.5444 degree_AUC=0.5000 textsim_AUC=0.5045  
seed=3 singleton  [random-neg, same 6242+45-excluded positives]  n_pos=6242 n_neg=9799 SAGE_AUC=0.7346 SAGE_PRAUC=0.5660 degree_AUC=0.0769 textsim_AUC=0.5070  
