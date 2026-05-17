from __future__ import annotations
from pathlib import Path
import json, math, shutil
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

RANDOM_STATE=123
BROAD='ms_ai_diffusion_q1_2026_pct'
GAP='ai_capability_conversion_gap_q1_2026_pct'
CLAUDE='log_anthropic_per_capita_index'
CLUSTER_FEATURES=['institutional_capacity_z','imf_aipi_index_2023_z','access_assets_z','reg_ethics_z','policy_intensity_z']
MAIN_X=['institutional_capacity_z','imf_aipi_index_2023_z','access_assets_z','policy_intensity_z','log_population_z']

def zscore(s):
    s=pd.to_numeric(s, errors='coerce')
    sd=s.std(ddof=0)
    if not sd or pd.isna(sd): return pd.Series(np.nan, index=s.index)
    return (s-s.mean())/sd

def ols_hc3(df, y, xs, model=''):
    d=df[[y]+xs].dropna().copy()
    if len(d)<len(xs)+5:
        return pd.DataFrame()
    res=sm.OLS(d[y], sm.add_constant(d[xs])).fit(cov_type='HC3')
    rows=[]
    for t in res.params.index:
        lo,hi=res.conf_int().loc[t]
        rows.append({'model':model,'outcome':y,'term':t,'coef':res.params[t],'se_hc3':res.bse[t],'p_hc3':res.pvalues[t],'ci_lower':lo,'ci_upper':hi,'n':int(res.nobs),'r2':res.rsquared})
    return pd.DataFrame(rows)

def eta_squared_by_group(df, y, group):
    d=df[[y,group]].dropna()
    groups=[g[y].values for _,g in d.groupby(group)]
    if len(groups)<2: return np.nan, np.nan, np.nan, len(d)
    F,p=stats.f_oneway(*groups)
    H,pk=stats.kruskal(*groups)
    overall=d[y].mean()
    ssb=sum(len(g)*(np.mean(g)-overall)**2 for g in groups)
    sst=((d[y]-overall)**2).sum()
    return float(ssb/sst) if sst>0 else np.nan, float(p), float(pk), len(d)

def access_baseline_robustness(df, out):
    df=df.copy()
    # ensure standardization for tertiary and R&D raw variables if not present
    if 'wdi_tertiary_enrollment_gross_pct_latest_z' not in df:
        df['wdi_tertiary_enrollment_gross_pct_latest_z']=zscore(df.get('wdi_tertiary_enrollment_gross_pct_latest'))
    if 'wdi_rd_expenditure_pct_gdp_latest_z' not in df:
        df['wdi_rd_expenditure_pct_gdp_latest_z']=zscore(df.get('wdi_rd_expenditure_pct_gdp_latest'))
    baselines={
        'basic_access': ['log_gdp_per_capita_ppp_z','wdi_individuals_using_internet_pct_latest_z','log_population_z'],
        'minimal_economic_scale': ['log_gdp_per_capita_ppp_z','log_population_z'],
        'education_rd_access': ['log_gdp_per_capita_ppp_z','wdi_individuals_using_internet_pct_latest_z','log_population_z','wdi_tertiary_enrollment_gross_pct_latest_z','wdi_rd_expenditure_pct_gdp_latest_z'],
        'aipi_human_innovation_access': ['log_gdp_per_capita_ppp_z','wdi_individuals_using_internet_pct_latest_z','log_population_z','human_capital_z','innovation_integration_z'],
        'readiness_enriched_access': ['log_gdp_per_capita_ppp_z','wdi_individuals_using_internet_pct_latest_z','log_population_z','human_capital_z','innovation_integration_z','digital_infra_z'],
    }
    summary=[]
    regs=[]
    for name,xs in baselines.items():
        d=df[[BROAD]+xs].dropna()
        res=sm.OLS(d[BROAD], sm.add_constant(d[xs])).fit(cov_type='HC3')
        resid_col=f'gap_{name}_q1_pct'
        df[resid_col]=np.nan
        df.loc[d.index,resid_col]=d[BROAD]-res.predict(sm.add_constant(d[xs]))
        eta3,ap3,kp3,n3=eta_squared_by_group(df, resid_col, 'k3_cluster')
        eta2,ap2,kp2,n2=eta_squared_by_group(df, resid_col, 'k2_cluster')
        # capacity PC model on residual
        pcxs=['capacity_bundle_pc1_z','capacity_bundle_pc2_z','policy_intensity_z']
        pcres=ols_hc3(df, resid_col, pcxs, model=f'access_baseline_{name}_pc_model')
        regs.append(pcres)
        pc1=pcres[pcres.term=='capacity_bundle_pc1_z'].iloc[0] if not pcres.empty and (pcres.term=='capacity_bundle_pc1_z').any() else None
        pc2=pcres[pcres.term=='capacity_bundle_pc2_z'].iloc[0] if not pcres.empty and (pcres.term=='capacity_bundle_pc2_z').any() else None
        summary.append({
            'baseline':name,
            'predictors':'; '.join(xs),
            'n':int(res.nobs),
            'access_baseline_r2':res.rsquared,
            'k2_eta2_on_residual':eta2,
            'k2_anova_p':ap2,
            'k3_eta2_on_residual':eta3,
            'k3_anova_p':ap3,
            'pc1_overall_capacity_coef':pc1.coef if pc1 is not None else np.nan,
            'pc1_overall_capacity_p':pc1.p_hc3 if pc1 is not None else np.nan,
            'pc2_access_capacity_imbalance_coef':pc2.coef if pc2 is not None else np.nan,
            'pc2_access_capacity_imbalance_p':pc2.p_hc3 if pc2 is not None else np.nan,
        })
    summary=pd.DataFrame(summary)
    summary.to_csv(out/'alternative_access_baseline_robustness_q1.csv',index=False)
    pd.concat(regs,ignore_index=True).to_csv(out/'alternative_access_baseline_pc_models_q1.csv',index=False)
    df.to_csv(out/'final_analysis_data_q1_2026_with_alternative_gaps.csv',index=False)
    return df, summary, pd.concat(regs,ignore_index=True)

def logit_robustness(df,out):
    p=df[BROAD].astype(float).clip(1e-6,100-1e-6)/100
    df=df.copy()
    df['logit_ai_diffusion_q1_2026_recalc']=np.log(p/(1-p))
    res=ols_hc3(df,'logit_ai_diffusion_q1_2026_recalc',MAIN_X,model='logit_broad_q1_channel')
    res.to_csv(out/'logit_transformed_broad_diffusion_robustness_q1.csv',index=False)
    return res

def outlier_leverage_robustness(df,out):
    rows=[]
    variants=[]
    top3=df.nlargest(3,BROAD)['iso3'].tolist()
    top5=df.nlargest(5,BROAD)['iso3'].tolist()
    anchor=['USA','CHN','ARE','SGP']
    # baseline influence from main broad model
    d=df[[BROAD]+MAIN_X+['iso3']].dropna().copy()
    res=sm.OLS(d[BROAD],sm.add_constant(d[MAIN_X])).fit()
    inf=OLSInfluence(res)
    cooks=inf.cooks_distance[0]
    leverage=inf.hat_matrix_diag
    d['cooks_d']=cooks
    d['leverage']=leverage
    cook_cut=4/len(d)
    lev_cut=2*(len(MAIN_X)+1)/len(d)
    cook_out=d.loc[d.cooks_d>cook_cut,'iso3'].tolist()
    leverage_out=d.loc[d.leverage>lev_cut,'iso3'].tolist()
    diagnostics=pd.DataFrame({'iso3':d.iso3,'cooks_d':cooks,'leverage':leverage}).sort_values(['cooks_d','leverage'],ascending=False)
    diagnostics.to_csv(out/'outlier_leverage_country_diagnostics_q1.csv',index=False)
    variants=[
        ('all_countries',[]),
        ('exclude_top3_broad_diffusion',top3),
        ('exclude_top5_broad_diffusion',top5),
        ('exclude_USA_CHN_ARE_SGP',anchor),
        ('exclude_cooks_d_gt_4_over_n',cook_out),
        ('exclude_leverage_gt_2p_over_n',leverage_out),
    ]
    # broad percentage and logit models
    for v,ex in variants:
        sub=df[~df.iso3.isin(ex)].copy()
        for y,ys in [(BROAD,'broad_pct'),('logit_ai_diffusion_q1_2026','broad_logit')]:
            if y not in sub or sub[y].isna().all():
                continue
            rr=ols_hc3(sub,y,MAIN_X,model=v)
            for term in ['institutional_capacity_z','imf_aipi_index_2023_z','access_assets_z','policy_intensity_z','log_population_z']:
                r=rr[rr.term==term]
                if len(r):
                    rows.append({'variant':v,'outcome':ys,'excluded_iso3':'; '.join(ex),'n_excluded':len(ex),'term':term,'coef':r.coef.iloc[0],'se_hc3':r.se_hc3.iloc[0],'p_hc3':r.p_hc3.iloc[0],'n':int(r.n.iloc[0]),'r2':r.r2.iloc[0]})
        # conversion PC model outcome
        pcxs=['capacity_bundle_pc1_z','capacity_bundle_pc2_z','policy_intensity_z']
        rr=ols_hc3(sub,GAP,pcxs,model=v)
        for term in pcxs:
            r=rr[rr.term==term]
            if len(r): rows.append({'variant':v,'outcome':'conversion_gap_pc','excluded_iso3':'; '.join(ex),'n_excluded':len(ex),'term':term,'coef':r.coef.iloc[0],'se_hc3':r.se_hc3.iloc[0],'p_hc3':r.p_hc3.iloc[0],'n':int(r.n.iloc[0]),'r2':r.r2.iloc[0]})
    tab=pd.DataFrame(rows)
    tab.to_csv(out/'outlier_and_leverage_robustness_q1.csv',index=False)
    # compact summary for manuscript/supp tables
    terms=['institutional_capacity_z','access_assets_z','policy_intensity_z','capacity_bundle_pc2_z']
    compact=tab[tab.term.isin(terms)].copy()
    compact.to_csv(out/'outlier_and_leverage_robustness_compact_q1.csv',index=False)
    return tab, diagnostics

def gmm_bic_cluster_robustness(df,out):
    d=df.dropna(subset=CLUSTER_FEATURES).copy()
    X=d[CLUSTER_FEATURES].astype(float).values
    # standardize
    X=(X-X.mean(axis=0))/X.std(axis=0,ddof=0)
    rows=[]
    for cov in ['full','tied','diag','spherical']:
        for k in range(1,7):
            gm=GaussianMixture(n_components=k,covariance_type=cov,n_init=1,random_state=RANDOM_STATE,reg_covar=1e-6,max_iter=100)
            gm.fit(X)
            labels=gm.predict(X)+1
            # order labels by institutional capacity if k>1
            if k>1:
                tmp=d.copy(); tmp['lab_raw']=labels
                order=tmp.groupby('lab_raw')['institutional_capacity_z'].mean().sort_values(ascending=False).index.tolist()
                mapper={raw:i+1 for i,raw in enumerate(order)}
                ordered=np.array([mapper[x] for x in labels])
            else:
                ordered=labels
            k3ari=np.nan
            if k==3 and 'k3_cluster' in d.columns:
                k3ari=adjusted_rand_score(d['k3_cluster'].astype(int), ordered)
            rows.append({
                'covariance_type':cov,'k':k,'n':len(d),'aic':gm.aic(X),'bic':gm.bic(X),'converged':gm.converged_,'n_iter':gm.n_iter_,
                'ari_with_kmeans_k3_if_k3':k3ari,
                'cluster_counts':json.dumps(pd.Series(ordered).value_counts().sort_index().to_dict()),
                'q1_broad_eta2':eta_squared_temp(d[BROAD], ordered),
                'q1_gap_eta2':eta_squared_temp(d[GAP], ordered),
                'claude_eta2':eta_squared_temp(d[CLAUDE], ordered),
            })
    tab=pd.DataFrame(rows)
    tab.to_csv(out/'gmm_bic_cluster_robustness_q1.csv',index=False)
    best=tab.loc[tab.groupby('covariance_type')['bic'].idxmin()].sort_values('bic')
    best.to_csv(out/'gmm_bic_best_by_covariance_q1.csv',index=False)
    return tab,best

def eta_squared_temp(y, labels):
    d=pd.DataFrame({'y':y.values if hasattr(y,'values') else y,'label':labels}).dropna()
    groups=[g.y.values for _,g in d.groupby('label')]
    if len(groups)<2: return np.nan
    overall=d.y.mean(); ssb=sum(len(g)*(np.mean(g)-overall)**2 for g in groups); sst=((d.y-overall)**2).sum()
    return float(ssb/sst) if sst>0 else np.nan

def wb_metadata(df, xlsx_path, out):
    meta=pd.read_excel(xlsx_path, sheet_name='Metadata - Countries')
    meta=meta.rename(columns={'Country Code':'iso3','Region':'wb_region','IncomeGroup':'wb_income_group','TableName':'wb_table_name'})
    meta=meta[['iso3','wb_region','wb_income_group','wb_table_name']]
    meta=meta[meta['wb_region'].notna() | meta['wb_income_group'].notna()].copy()
    merged=df.merge(meta,on='iso3',how='left')
    merged.to_csv(out/'final_analysis_data_q1_2026_with_wb_metadata.csv',index=False)
    match=pd.DataFrame([{
        'core_n':len(df),'matched_region_n':int(merged.wb_region.notna().sum()),'matched_income_group_n':int(merged.wb_income_group.notna().sum()),
        'unmatched_iso3':'; '.join(merged.loc[merged.wb_region.isna(),'iso3'].tolist())
    }])
    match.to_csv(out/'world_bank_region_income_metadata_match_q1.csv',index=False)
    # distributions
    reg=merged.groupby(['wb_region','k3_label']).agg(n=('iso3','count'),mean_q1_broad=(BROAD,'mean'),mean_gap=(GAP,'mean')).reset_index()
    inc=merged.groupby(['wb_income_group','k3_label']).agg(n=('iso3','count'),mean_q1_broad=(BROAD,'mean'),mean_gap=(GAP,'mean')).reset_index()
    reg.to_csv(out/'world_bank_region_by_k3_profile_q1.csv',index=False)
    inc.to_csv(out/'world_bank_income_group_by_k3_profile_q1.csv',index=False)
    return merged,match,reg,inc

def main():
    package_root=Path(__file__).resolve().parents[1]
    out=package_root/'tables'
    out.mkdir(exist_ok=True)
    df=pd.read_csv(package_root/'data/final_analysis_data_q1_2026.csv')
    # coerce k labels numeric in case
    df['k3_cluster']=pd.to_numeric(df['k3_cluster'],errors='coerce')
    df['k2_cluster']=pd.to_numeric(df['k2_cluster'],errors='coerce')
    # ensure logit exists
    p=df[BROAD].astype(float).clip(1e-6,100-1e-6)/100
    df['logit_ai_diffusion_q1_2026']=np.log(p/(1-p))
    print('running alternative access-baseline robustness')
    df_alt,ab,abregs=access_baseline_robustness(df,out)
    print('running logit robustness')
    logit=logit_robustness(df,out)
    print('running outlier and leverage robustness')
    outtab, infl=outlier_leverage_robustness(df,out)
    print('running GMM BIC robustness')
    gmm,best=gmm_bic_cluster_robustness(df,out)
    print('running World Bank metadata checks')
    wb_xlsx=package_root/'data/world_bank_population_indicator_download_with_country_metadata.xlsx'
    wbmerged,wbmatch,wbreg,wbinc=wb_metadata(df,wb_xlsx,out)
    summary={
        'alternative_access_baseline': ab.to_dict(orient='records'),
        'logit_terms': logit[logit.term.isin(['institutional_capacity_z','imf_aipi_index_2023_z','access_assets_z','policy_intensity_z','log_population_z'])].to_dict(orient='records'),
        'outlier_top5_by_cooks': infl.head(5).to_dict(orient='records'),
        'gmm_best_by_covariance': best.to_dict(orient='records'),
        'wb_match': wbmatch.to_dict(orient='records'),
    }
    (out/'extended_analysis_summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
    print('wrote',out)

if __name__=='__main__':
    main()
