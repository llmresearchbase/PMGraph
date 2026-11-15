# Experiment 1: degree-matched negative-sampling robustness check
Same trained SAGE checkpoint, NOT retrained. Only the negative half of the test set is reconstructed to match each positive's train-degree.

seed=1: exact=10943 (0.9719)  fallback=271  unmatched=45  total_match_rate(incl.fallback)=0.9960

seed=1 overall    [degree-matched]  n_pos=11214 n_neg=11214 SAGE_AUC=0.5674 SAGE_PRAUC=0.5463 degree_AUC=0.5000 textsim_AUC=0.5069  
seed=1 overall    [random-neg, same 11214+45-excluded positives]  n_pos=11214 n_neg=11259 SAGE_AUC=0.6953 SAGE_PRAUC=0.6608 degree_AUC=0.4154 textsim_AUC=0.5106  
seed=1 singleton  [degree-matched]  n_pos=6242 n_neg=6242 SAGE_AUC=0.6106 SAGE_PRAUC=0.5656 degree_AUC=0.5000 textsim_AUC=0.5022  
seed=1 singleton  [random-neg, same 6242+45-excluded positives]  n_pos=6242 n_neg=9799 SAGE_AUC=0.8155 SAGE_PRAUC=0.6625 degree_AUC=0.0769 textsim_AUC=0.5070  

seed=2: exact=10943 (0.9719)  fallback=271  unmatched=45  total_match_rate(incl.fallback)=0.9960

seed=2 overall    [degree-matched]  n_pos=11214 n_neg=11214 SAGE_AUC=0.5679 SAGE_PRAUC=0.5466 degree_AUC=0.5000 textsim_AUC=0.5056  
seed=2 overall    [random-neg, same 11214+45-excluded positives]  n_pos=11214 n_neg=11259 SAGE_AUC=0.6953 SAGE_PRAUC=0.6608 degree_AUC=0.4154 textsim_AUC=0.5106  
seed=2 singleton  [degree-matched]  n_pos=6242 n_neg=6242 SAGE_AUC=0.6103 SAGE_PRAUC=0.5645 degree_AUC=0.5000 textsim_AUC=0.5012  
seed=2 singleton  [random-neg, same 6242+45-excluded positives]  n_pos=6242 n_neg=9799 SAGE_AUC=0.8155 SAGE_PRAUC=0.6625 degree_AUC=0.0769 textsim_AUC=0.5070  

seed=3: exact=10943 (0.9719)  fallback=271  unmatched=45  total_match_rate(incl.fallback)=0.9960

seed=3 overall    [degree-matched]  n_pos=11214 n_neg=11214 SAGE_AUC=0.5689 SAGE_PRAUC=0.5468 degree_AUC=0.5000 textsim_AUC=0.5069  
seed=3 overall    [random-neg, same 11214+45-excluded positives]  n_pos=11214 n_neg=11259 SAGE_AUC=0.6953 SAGE_PRAUC=0.6608 degree_AUC=0.4154 textsim_AUC=0.5106  
seed=3 singleton  [degree-matched]  n_pos=6242 n_neg=6242 SAGE_AUC=0.6154 SAGE_PRAUC=0.5678 degree_AUC=0.5000 textsim_AUC=0.5045  
seed=3 singleton  [random-neg, same 6242+45-excluded positives]  n_pos=6242 n_neg=9799 SAGE_AUC=0.8155 SAGE_PRAUC=0.6625 degree_AUC=0.0769 textsim_AUC=0.5070  
