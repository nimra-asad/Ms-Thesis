"""Recompute test controller/DR outputs using PatchTST verification and joint cost/peak optimization.

Uses saved forecasts and frozen validation thresholds; forecast models are not retrained.
Writes parallel *_patchtst_joint artifacts so the original run remains auditable.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from dual_evidence.controller.decision import Thresholds, decide
from dual_evidence.verification.secondary import verify_forecast
from dual_evidence.optimization.demand_response import optimize_day_ahead

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'

def main():
    log=pd.read_csv(R/'decision_log.csv')
    frames=pd.read_csv(R/'test_forecasts.csv')
    grouped=(frames.groupby(['origin','household_id_code'],sort=False)
        .agg(actual=('actual_kw',list),lightgbm=('lightgbm_kw',list))
        .reset_index().rename(columns={'household_id_code':'household_code'}))
    grouped['origin']=pd.to_datetime(grouped.origin).dt.strftime('%Y-%m-%d %H:%M:%S')
    log['origin']=pd.to_datetime(log.origin).dt.strftime('%Y-%m-%d %H:%M:%S')
    data=log.merge(grouped,on=['origin','household_code'],validate='one_to_one').reset_index(drop=True)
    with np.load(R/'forecast_cache_lightgbm.npz') as z: lightgbm=z['test'].copy()
    with np.load(R/'forecast_cache_patchtst.npz') as z: patchtst=z['test'].copy()
    params=json.loads((R/'decision_thresholds.json').read_text())
    threshold=Thresholds(**params['dual_evidence'])
    uncertainty_cut=float(params['uncertainty_only_matched_coverage_threshold'])
    dc=[]; uc=[]; ec=[]; verifiers=[]
    for i,row in data.iterrows():
        unc={'decision':'ACT' if row.normalized_interval_width<=uncertainty_cut else 'ABSTAIN',
             'reason':'validation-matched uncertainty threshold'}
        exp={'decision':'ACT' if row.explanation_stability>=threshold.explanation_act_min else 'ABSTAIN',
             'reason':'validation explanation threshold'}
        initial=decide(row.normalized_interval_width,row.explanation_stability,threshold)
        check=(verify_forecast(lightgbm[i],patchtst[i],threshold.verification_relative_tolerance)
               if initial['decision']=='VERIFY' else None)
        final=decide(row.normalized_interval_width,row.explanation_stability,threshold,
                     check['agrees'] if check is not None else None)
        dc.append({'decision':final['decision'],'initial_decision':initial['decision'],
                   'reason':final['reason'], 'verification':check})
        uc.append(unc); ec.append(exp); verifiers.append(check)
    data['initial_dual_decision']=[x['initial_decision'] for x in dc]
    data['dual_decision']=[x['decision'] for x in dc]
    data['dual_reason']=[x['reason'] for x in dc]
    data['verification_relative_difference']=[x['mean_relative_difference'] if x else None for x in verifiers]
    data['verification_agrees']=[x['agrees'] if x else None for x in verifiers]
    data.to_csv(R/'decision_log_patchtst.csv',index=False)
    decisions={'ungated':[{'decision':'ACT'} for _ in range(len(data))],
               'uncertainty_only':uc,'explanation_only':ec,
               'dual_evidence_patchtst':[x for x in dc]}
    sampled=[]
    for i,row in data.iterrows():
        day=pd.Timestamp(row.origin).date()
        if day.toordinal()%7==0:
            sampled.append(i)
    ecfg=yaml.safe_load((ROOT/'configs/experiment.yaml').read_text())
    penalty=float(ecfg.get('dr_peak_penalty_pkr_per_kw',50.0))
    cache={}; drrows=[]; summary={}
    # Solve each sampled origin at most once, then apply the same simulated outcome to eligible policies.
    for i in sampled:
        eligible=[name for name,items in decisions.items() if items[i]['decision']=='ACT']
        if not eligible: continue
        row=data.iloc[i]
        result=optimize_day_ahead(row.lightgbm,row.actual,
            flexible_share=float(ecfg.get('dr_flexible_share',.20)),
            peak_penalty_pkr_per_kw=penalty)
        cache[i]=result
        for name in eligible:
            drrows.append({'origin':row.origin,'household_code':int(row.household_code),
                           'config':name, **{k:v for k,v in result.items()
                           if k not in ('scheduled_flexible_kw','optimized_actual_kw')}})
    dr=pd.DataFrame(drrows)
    dr.to_csv(R/'demand_response_patchtst_joint.csv',index=False)
    for name,items in decisions.items():
        acts=int(sum(x['decision']=='ACT' for x in items)); vals=[cache[i] for i in cache if items[i]['decision']=='ACT']
        summary[name]={
            'automation_coverage':acts/len(data), 'act_origins':acts,
            'dr_evaluated_actions':len(vals),
            'mean_cost_saving_pkr':float(np.mean([x['cost_saving_pkr'] for x in vals])) if vals else 0.0,
            'mean_peak_reduction_kw':float(np.mean([x['peak_reduction_kw'] for x in vals])) if vals else 0.0,
            'cost_increase_rate':float(np.mean([x['cost_increase'] for x in vals])) if vals else 0.0,
            'peak_increase_rate':float(np.mean([x['peak_increase'] for x in vals])) if vals else 0.0,
        }
    payload={'secondary_verifier':'PatchTST','dr_objective':'forecast energy cost + peak_penalty * forecast peak','peak_penalty_pkr_per_kw':penalty,
             'sample_policy':'one in seven test origin dates; deterministic date ordinal','thresholds_source':'frozen validation thresholds',
             'numeric_regret':'withheld','controller':summary}
    (R/'controller_metrics_patchtst_joint.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
    lines=['# PatchTST Verification and Cost–Peak Demand-Response Results','',
           'This recomputation uses saved forecasts and frozen validation thresholds. PatchTST replaces Seasonal Naive as the secondary consistency verifier. Demand-response optimization minimizes forecast-based illustrative energy cost plus a 50 PKR/kW peak penalty. Cost and peak outcomes are reported separately; outcomes use the same deterministic one-in-seven test-date sample. No model was retrained. Numeric regret is withheld.', '',
           '| Configuration | Test ACT coverage | ACT origins | Sampled DR cases | Mean cost saving (PKR) | Mean peak reduction (kW) | Cost increase rate | Peak increase rate |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,x in summary.items():
        lines.append(f"| {name} | {x['automation_coverage']:.3f} | {x['act_origins']} | {x['dr_evaluated_actions']} | {x['mean_cost_saving_pkr']:.2f} | {x['mean_peak_reduction_kw']:.4f} | {x['cost_increase_rate']:.3f} | {x['peak_increase_rate']:.3f} |")
    lines += ['',f'PatchTST verification passed for {sum(v is True for v in data.verification_agrees)} of {sum(v is not None for v in data.verification_agrees)} VERIFY cases. These figures are simulated and depend on the stated flexible-load assumptions and peak penalty.', '']
    (ROOT/'reports'/'patchtst_joint_dr_results.md').write_text('\n'.join(lines))
    print(json.dumps(payload,indent=2))

if __name__=='__main__': main()
