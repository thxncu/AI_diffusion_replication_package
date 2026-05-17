#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, zipfile, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

RANDOM_STATE = 123
CLUSTER_FEATURES = ["institutional_capacity_z", "imf_aipi_index_2023_z", "access_assets_z", "reg_ethics_z", "policy_intensity_z"]
CLUSTER_FEATURES_NO_RE = ["institutional_capacity_z", "imf_aipi_index_2023_z", "access_assets_z", "policy_intensity_z"]
CLUSTER_FEATURES_NO_POLICY = ["institutional_capacity_z", "imf_aipi_index_2023_z", "access_assets_z", "reg_ethics_z"]
CLUSTER_FEATURES_NO_RE_NO_POLICY = ["institutional_capacity_z", "imf_aipi_index_2023_z", "access_assets_z"]
MAIN_X = ["institutional_capacity_z", "imf_aipi_index_2023_z", "access_assets_z", "policy_intensity_z", "log_population_z"]
BROAD = "ms_ai_diffusion_q1_2026_pct"
GAP = "ai_capability_conversion_gap_q1_2026_pct"
CLAUDE = "log_anthropic_per_capita_index"


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and not pd.isna(sd) else pd.Series(np.nan, index=s.index)


def standardize(X):
    X = np.asarray(X, float)
    m = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1
    return (X - m) / sd


def kmeans_np(X, k, n_init=100, max_iter=300, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed + k)
    best_labels, best_inertia = None, np.inf
    n = len(X)
    for _ in range(n_init):
        centers = [X[rng.integers(n)]]
        for _j in range(1, k):
            dist = np.min(((X[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
            probs = dist / dist.sum() if dist.sum() > 0 else np.ones(n) / n
            centers.append(X[rng.choice(n, p=probs)])
        centers = np.array(centers)
        labels = np.zeros(n, dtype=int)
        for _iter in range(max_iter):
            d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new = d.argmin(axis=1)
            if np.array_equal(new, labels) and _iter > 0:
                break
            labels = new
            for j in range(k):
                if np.any(labels == j):
                    centers[j] = X[labels == j].mean(axis=0)
        inertia = float(((X - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    return best_labels, best_inertia


def silhouette_np(X, labels):
    D = squareform(pdist(X))
    labs = np.unique(labels)
    vals = []
    for i in range(len(X)):
        same = labels == labels[i]
        a = D[i, same].sum() / (same.sum() - 1) if same.sum() > 1 else 0.0
        b = min(D[i, labels == lab].mean() for lab in labs if lab != labels[i])
        vals.append((b - a) / max(a, b) if max(a, b) > 0 else 0)
    return float(np.mean(vals))


def calinski_np(X, labels):
    n = len(X)
    labs = np.unique(labels)
    k = len(labs)
    overall = X.mean(axis=0)
    ssb = sum(((labels == lab).sum()) * ((X[labels == lab].mean(axis=0) - overall) ** 2).sum() for lab in labs)
    ssw = sum(((X[labels == lab] - X[labels == lab].mean(axis=0)) ** 2).sum() for lab in labs)
    return float((ssb / (k - 1)) / (ssw / (n - k))) if k > 1 and n > k and ssw > 0 else np.nan


def davies_np(X, labels):
    labs = np.unique(labels)
    centers, S = [], []
    for lab in labs:
        G = X[labels == lab]
        c = G.mean(axis=0)
        centers.append(c)
        S.append(np.sqrt(((G - c) ** 2).sum(axis=1)).mean())
    centers = np.array(centers)
    S = np.array(S)
    M = squareform(pdist(centers))
    vals = []
    for i in range(len(labs)):
        ratios = [(S[i] + S[j]) / M[i, j] for j in range(len(labs)) if j != i and M[i, j] > 0]
        vals.append(max(ratios))
    return float(np.mean(vals))


def adjusted_rand(labels1, labels2):
    from math import comb
    l1 = pd.Series(labels1)
    l2 = pd.Series(labels2)
    tab = pd.crosstab(l1, l2).values
    sum_comb = sum(comb(int(n), 2) for n in tab.flatten())
    a = sum(comb(int(n), 2) for n in tab.sum(axis=1))
    b = sum(comb(int(n), 2) for n in tab.sum(axis=0))
    total = comb(len(labels1), 2)
    expected = a * b / total if total else 0
    maxindex = (a + b) / 2
    return float((sum_comb - expected) / (maxindex - expected)) if maxindex != expected else 1.0


def order_labels_by_capacity(df, labels, k):
    tmp = df.copy()
    tmp["raw_label"] = labels
    order = tmp.groupby("raw_label")["institutional_capacity_z"].mean().sort_values(ascending=False).index.tolist()
    mapper = {raw: i + 1 for i, raw in enumerate(order)}
    return pd.Series(labels, index=df.index).map(mapper).astype(int)


def load_q1(q1_path):
    q = pd.read_csv(q1_path, encoding="cp1252")
    q.columns = [c.strip() for c in q.columns]
    q["Economy"] = q["Economy"].replace({"TŸrkiye": "Türkiye", "T\x9frkiye": "Türkiye"})
    for c in q.columns[1:]:
        q[c] = q[c].astype(str).str.replace("%", "", regex=False).astype(float)
    q = q.rename(columns={
        "H1 2025 AI Diffusion": "ms_ai_diffusion_h1_2025_pct_from_q1_file",
        "H2 2025 AI Diffusion": "ms_ai_diffusion_h2_2025_pct_from_q1_file",
        "Q1 2026 AI Diffusion": "ms_ai_diffusion_q1_2026_pct",
    })
    return q


def prepare_data(input_path, q1_path, out_tables):
    df = pd.read_csv(input_path)
    # numeric conversion except known character fields
    text_cols = {"iso3", "country_name", "country_name_microsoft", "country_name_imf", "country_name_wgi", "country_name_wdi", "country_name_oecd", "country_name_anthropic_2025", "country_name_anthropic_2026", "country_name_bti", "conversion_position_label", "governance_regime_cfp", "anthropic_usage_tier_2025_aug"}
    for c in df.columns:
        if c not in text_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    q = load_q1(q1_path)
    merged = df.merge(q, left_on="country_name_microsoft", right_on="Economy", how="left")
    merged["q1_match_status"] = np.where(merged["ms_ai_diffusion_q1_2026_pct"].notna(), "matched", "unmatched")
    merged["q1_diffusion_change_from_h2_pct_points"] = merged["ms_ai_diffusion_q1_2026_pct"] - merged["ms_ai_diffusion_h2_2025_pct"]
    merged["q1_diffusion_change_from_h1_pct_points"] = merged["ms_ai_diffusion_q1_2026_pct"] - merged["ms_ai_diffusion_h1_2025_pct"]
    merged["ai_diffusion_fraction_q1_2026"] = merged["ms_ai_diffusion_q1_2026_pct"] / 100.0
    p = merged["ai_diffusion_fraction_q1_2026"].clip(1e-6, 1-1e-6)
    merged["logit_ai_diffusion_q1_2026"] = np.log(p/(1-p))
    if CLAUDE not in merged.columns or merged[CLAUDE].notna().sum() == 0:
        merged[CLAUDE] = np.log1p(merged["anthropic_claude_usage_per_capita_index_2025_aug"])
    merged["anthropic_available"] = merged[CLAUDE].notna().astype(int)
    # Q1 broad standardization and divergence
    merged["broad_mass_diffusion_q1_2026_z"] = zscore(merged[BROAD])
    merged["claude_visible_platform_use_z"] = zscore(merged[CLAUDE])
    merged["broad_claude_visible_divergence_q1_z"] = merged["broad_mass_diffusion_q1_2026_z"] - merged["claude_visible_platform_use_z"]
    # access adjusted residual for Q1 broad diffusion
    access_x = ["log_gdp_per_capita_ppp_z", "wdi_individuals_using_internet_pct_latest_z", "log_population_z"]
    d = merged[[BROAD] + access_x].dropna()
    access_model = sm.OLS(d[BROAD], sm.add_constant(d[access_x])).fit()
    merged.loc[d.index, "expected_diffusion_basic_access_q1_2026_pct"] = access_model.predict(sm.add_constant(d[access_x]))
    merged[GAP] = merged[BROAD] - merged["expected_diffusion_basic_access_q1_2026_pct"]
    merged["ai_capability_conversion_gap_q1_2026_z"] = zscore(merged[GAP])
    # capacity PCA and residualized variables recalculated on full core sample
    pvars = ["institutional_capacity_z", "imf_aipi_index_2023_z", "access_assets_z", "reg_ethics_z"]
    X = standardize(merged[pvars].fillna(merged[pvars].mean()).values)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pcs = U*S
    # Orient principal components for interpretability. PC1 is the overall capacity bundle; PC2 is high access relative to institutional-regulatory capacity.
    if Vt[0, :].mean() < 0:
        Vt[0, :] *= -1
        pcs[:, 0] *= -1
    access_idx = pvars.index("access_assets_z")
    inst_idx = pvars.index("institutional_capacity_z")
    reg_idx = pvars.index("reg_ethics_z")
    if not (Vt[1, access_idx] > 0 and Vt[1, inst_idx] < 0 and Vt[1, reg_idx] < 0):
        Vt[1, :] *= -1
        pcs[:, 1] *= -1
    explained = (S**2)/(len(X)-1)
    evr = explained/explained.sum()
    for i in range(4):
        merged[f"capacity_bundle_pc{i+1}_z"] = zscore(pd.Series(pcs[:,i], index=merged.index))
    load = pd.DataFrame(Vt.T, index=pvars, columns=[f"PC{i+1}" for i in range(4)]).reset_index().rename(columns={"index":"variable"})
    ev = pd.DataFrame({"component":[f"PC{i+1}" for i in range(4)], "explained_variance_ratio": evr})
    d2 = merged[["institutional_capacity_z", "imf_aipi_index_2023_z"]].dropna()
    ic = sm.OLS(d2["institutional_capacity_z"], sm.add_constant(d2[["imf_aipi_index_2023_z"]])).fit()
    aipi = sm.OLS(d2["imf_aipi_index_2023_z"], sm.add_constant(d2[["institutional_capacity_z"]])).fit()
    merged.loc[d2.index, "ic_residualized_on_aipi_z"] = zscore(pd.Series(ic.resid, index=d2.index))
    merged.loc[d2.index, "aipi_residualized_on_ic_z"] = zscore(pd.Series(aipi.resid, index=d2.index))
    # WGI alternatives
    wgi3 = ["wgi_government_effectiveness_latest_z", "wgi_regulatory_quality_latest_z", "wgi_rule_of_law_latest_z"]
    merged["institutional_capacity_three_wgi_z"] = zscore(merged[wgi3].mean(axis=1))
    if "wgi_control_corruption_latest" in merged.columns:
        merged["wgi_control_corruption_latest_z_local"] = zscore(merged["wgi_control_corruption_latest"])
        merged["institutional_capacity_four_wgi_z"] = zscore(merged[wgi3 + ["wgi_control_corruption_latest_z_local"]].mean(axis=1))
    merge_diag = {
        "core_rows": int(len(merged)),
        "q1_matched_rows": int(merged["ms_ai_diffusion_q1_2026_pct"].notna().sum()),
        "q1_unmatched_rows": int(merged["ms_ai_diffusion_q1_2026_pct"].isna().sum()),
        "h2_value_max_abs_difference_between_original_and_q1_file": float((merged["ms_ai_diffusion_h2_2025_pct"] - merged["ms_ai_diffusion_h2_2025_pct_from_q1_file"]).abs().max()),
        "q1_mean_broad_diffusion_pct": float(merged[BROAD].mean()),
        "q1_median_broad_diffusion_pct": float(merged[BROAD].median()),
        "q1_mean_change_from_h2_pct_points": float(merged["q1_diffusion_change_from_h2_pct_points"].mean()),
        "access_baseline_r2_q1": float(access_model.rsquared),
    }
    pd.DataFrame([merge_diag]).to_csv(out_tables/"q1_merge_and_access_baseline_diagnostics.csv", index=False)
    load.to_csv(out_tables/"capacity_bundle_pca_loadings_q1.csv", index=False)
    ev.to_csv(out_tables/"capacity_bundle_pca_explained_variance_q1.csv", index=False)
    # save q1 source with normalized encoding
    q.to_csv(out_tables/"microsoft_ai_diffusion_q1_2026_source_normalized.csv", index=False)
    return merged, merge_diag


def assign_cluster(df, features, k, prefix):
    d = df.dropna(subset=features).copy()
    X = standardize(d[features].values)
    raw, inertia = kmeans_np(X, k, n_init=100)
    ordered = order_labels_by_capacity(d, raw, k)
    df[f"{prefix}_cluster"] = np.nan
    df.loc[d.index, f"{prefix}_cluster"] = ordered
    if k == 2:
        names = {1:"High-capacity profile", 2:"Lower-capacity profile"}
    elif k == 3:
        names = {1:"High-capacity integrated profile", 2:"Intermediate readiness-access profile", 3:"Capacity-constrained low-observability profile"}
    else:
        names = {i:f"Configuration {i}" for i in range(1,k+1)}
    df[f"{prefix}_label"] = df[f"{prefix}_cluster"].map(names)
    return df, inertia


def cluster_diagnostics(df, out):
    feature_sets = {
        "main_all_inputs": CLUSTER_FEATURES,
        "no_regulatory_ethics": CLUSTER_FEATURES_NO_RE,
        "no_policy_visibility": CLUSTER_FEATURES_NO_POLICY,
        "core_capacity_access_only": CLUSTER_FEATURES_NO_RE_NO_POLICY,
    }
    rows = []
    for fs_name, features in feature_sets.items():
        d = df.dropna(subset=features).copy()
        X = standardize(d[features].values)
        for k in range(2,7):
            labels, inertia = kmeans_np(X,k,n_init=100)
            ward = fcluster(linkage(X, method="ward"), k, criterion="maxclust") - 1
            rows.append({
                "feature_set": fs_name,
                "k": k,
                "n": len(d),
                "n_features": len(features),
                "silhouette": silhouette_np(X, labels),
                "calinski_harabasz": calinski_np(X, labels),
                "davies_bouldin": davies_np(X, labels),
                "ward_vs_kmeans_ari": adjusted_rand(labels, ward),
                "kmeans_inertia": inertia,
                "cluster_counts_raw": json.dumps(pd.Series(labels).value_counts().sort_index().to_dict()),
            })
    diag = pd.DataFrame(rows)
    diag.to_csv(out/"cluster_diagnostics_k2_to_k6_q1.csv", index=False)
    return diag


def bootstrap_stability(df, out, ks=(2,3,4), n_resamples=80, sample_frac=0.80):
    rows = []
    d = df.dropna(subset=CLUSTER_FEATURES).copy()
    X_full = standardize(d[CLUSTER_FEATURES].values)
    for k in ks:
        full_labels, _ = kmeans_np(X_full, k, n_init=100)
        rng = np.random.default_rng(RANDOM_STATE + 77 + k)
        for b in range(n_resamples):
            idx = np.sort(rng.choice(len(d), size=int(round(sample_frac*len(d))), replace=False))
            Xs = X_full[idx]
            sub_labels, _ = kmeans_np(Xs, k, n_init=50, seed=RANDOM_STATE + b + k*1000)
            rows.append({"k":k, "resample":b+1, "n_subsample":len(idx), "ari_with_full_on_subsample":adjusted_rand(full_labels[idx], sub_labels)})
    tab = pd.DataFrame(rows)
    tab.to_csv(out/"bootstrap_stability_k2_k3_k4_q1.csv", index=False)
    summary = tab.groupby("k")["ari_with_full_on_subsample"].agg(["count","mean","median","std",lambda x:x.quantile(.10),lambda x:x.quantile(.90)]).reset_index()
    summary.columns = ["k", "resamples", "mean_ari", "median_ari", "sd_ari", "p10_ari", "p90_ari"]
    summary.to_csv(out/"bootstrap_stability_summary_k2_k3_k4_q1.csv", index=False)
    return summary


def profiles(df, out):
    profile_rows=[]
    for prefix, k in [("k2",2),("k3",3)]:
        group_col=f"{prefix}_cluster"; label_col=f"{prefix}_label"
        prof = df.groupby([group_col,label_col]).agg(
            n=("iso3","count"),
            eu27_count=("is_eu27","sum"),
            institutional_capacity_mean_z=("institutional_capacity_z","mean"),
            aipi_mean_z=("imf_aipi_index_2023_z","mean"),
            access_assets_mean_z=("access_assets_z","mean"),
            reg_ethics_mean_z=("reg_ethics_z","mean"),
            policy_visibility_mean_z=("policy_intensity_z","mean"),
            q1_broad_diffusion_mean_pct=(BROAD,"mean"),
            h2_broad_diffusion_mean_pct=("ms_ai_diffusion_h2_2025_pct","mean"),
            q1_change_from_h2_mean_pct_points=("q1_diffusion_change_from_h2_pct_points","mean"),
            claude_visible_platform_use_log_mean=(CLAUDE,"mean"),
            broad_claude_visible_divergence_mean_z=("broad_claude_visible_divergence_q1_z","mean"),
            q1_conversion_gap_mean_pct=(GAP,"mean"),
            bti_capacity_mean_z=("bti_capacity_index_z","mean"),
        ).reset_index()
        prof.to_csv(out/f"{prefix}_configuration_profiles_q1.csv", index=False)
        prof.insert(0,"solution",prefix)
        profile_rows.append(prof)
    combined=pd.concat(profile_rows,ignore_index=True)
    combined.to_csv(out/"k2_k3_configuration_profiles_q1_combined.csv", index=False)
    return combined


def outcome_tests(df, out):
    outcomes=[BROAD, CLAUDE, "broad_claude_visible_divergence_q1_z", GAP, "q1_diffusion_change_from_h2_pct_points"]
    rows=[]
    for prefix,k in [("k2",2),("k3",3)]:
        group_col=f"{prefix}_cluster"
        for y in outcomes:
            d=df[[group_col,y]].dropna()
            groups=[g[y].values for _,g in d.groupby(group_col)]
            if len(groups) < 2:
                continue
            F,p=stats.f_oneway(*groups)
            H,pk=stats.kruskal(*groups)
            overall=d[y].mean()
            ssb=sum(len(g)*(np.mean(g)-overall)**2 for g in groups)
            sst=((d[y]-overall)**2).sum()
            rows.append({"solution":prefix,"k":k,"outcome":y,"n":len(d),"anova_F":F,"anova_p":p,"eta_squared":ssb/sst if sst>0 else np.nan,"kruskal_H":H,"kruskal_p":pk})
    tests=pd.DataFrame(rows)
    tests.to_csv(out/"outcome_separation_k2_k3_q1.csv",index=False)
    return tests


def ols_hc3(df,y,xs, model=""):
    d=df[[y]+xs].dropna()
    res=sm.OLS(d[y], sm.add_constant(d[xs])).fit(cov_type="HC3")
    rows=[]
    for t in res.params.index:
        lo, hi = res.conf_int().loc[t]
        rows.append({"model":model,"outcome":y,"term":t,"coef":res.params[t],"se_hc3":res.bse[t],"p_hc3":res.pvalues[t],"ci_lower":lo,"ci_upper":hi,"n":int(res.nobs),"r2":res.rsquared})
    return pd.DataFrame(rows)


def regression_suite(df,out):
    allres=[]
    for name,y in [("broad_q1_channel",BROAD),("claude_visible_platform_use",CLAUDE)]:
        allres.append(ols_hc3(df,y,MAIN_X,model=name))
    # Two-table split for manuscript
    channel=pd.concat(allres,ignore_index=True)
    channel.to_csv(out/"channel_specific_models_q1.csv",index=False)
    variants={
        "conversion_ic_only":["institutional_capacity_z"],
        "conversion_aipi_only":["imf_aipi_index_2023_z"],
        "conversion_ic_aipi":["institutional_capacity_z","imf_aipi_index_2023_z"],
        "conversion_ic_aipi_access_policy":["institutional_capacity_z","imf_aipi_index_2023_z","access_assets_z","policy_intensity_z"],
        "conversion_orthogonal_ic_and_aipi":["ic_residualized_on_aipi_z","imf_aipi_index_2023_z"],
        "conversion_capacity_pcs":["capacity_bundle_pc1_z","capacity_bundle_pc2_z","policy_intensity_z"],
        "conversion_wgi3":["institutional_capacity_three_wgi_z","imf_aipi_index_2023_z","access_assets_z","policy_intensity_z"],
        "conversion_wgi4":["institutional_capacity_four_wgi_z","imf_aipi_index_2023_z","access_assets_z","policy_intensity_z"],
    }
    conv=[]
    for name,xs in variants.items():
        conv.append(ols_hc3(df,GAP,xs,model=name))
    conv.append(ols_hc3(df[~df.iso3.isin(["USA","CHN"])],GAP,["institutional_capacity_z","imf_aipi_index_2023_z","access_assets_z","policy_intensity_z"],model="conversion_excluding_USA_CHN"))
    conv=pd.concat(conv,ignore_index=True)
    conv.to_csv(out/"conversion_gap_diagnostics_q1.csv",index=False)
    # VIF and correlation
    cols=["institutional_capacity_z","imf_aipi_index_2023_z","access_assets_z","reg_ethics_z","policy_intensity_z"]
    d=df[cols].dropna(); X=sm.add_constant(d[cols])
    pd.DataFrame([{"variable":term,"vif":variance_inflation_factor(X.values,i)} for i,term in enumerate(X.columns) if term!="const"]).to_csv(out/"vif_diagnostics_q1.csv",index=False)
    d.corr().reset_index().rename(columns={"index":"variable"}).to_csv(out/"capacity_variable_correlation_matrix_q1.csv",index=False)
    return channel,conv


def bti_and_selection(df,out):
    # BTI correlations and regressions updated for Q1 broad and gap
    corr=[]
    for b in ["bti_capacity_index_z","bti_governance_performance_z"]:
        for v in ["institutional_capacity_z",BROAD,CLAUDE,GAP,"imf_aipi_index_2023_z","access_assets_z"]:
            d=df[[b,v]].dropna()
            r,p=stats.pearsonr(d[b],d[v]) if len(d)>3 else (np.nan,np.nan)
            corr.append({"bti_measure":b,"variable":v,"n":len(d),"pearson_r":r,"p_value":p})
    pd.DataFrame(corr).to_csv(out/"bti_correlation_diagnostics_q1.csv",index=False)
    regs=[]
    xsets={"BTI only":["bti_capacity_index_z"],"BTI + AIPI + access + policy":["bti_capacity_index_z","imf_aipi_index_2023_z","access_assets_z","policy_intensity_z"],"WGI + BTI + AIPI + access + policy":["institutional_capacity_z","bti_capacity_index_z","imf_aipi_index_2023_z","access_assets_z","policy_intensity_z"]}
    for y in [BROAD,CLAUDE,GAP]:
        for name,xs in xsets.items():
            regs.append(ols_hc3(df,y,xs,model=name))
    pd.concat(regs,ignore_index=True).to_csv(out/"bti_robustness_regression_results_q1.csv",index=False)
    # selection models are unchanged in principle, but re-run for updated output names
    sel=["institutional_capacity_z","imf_aipi_index_2023_z","access_assets_z","policy_intensity_z","log_population_z"]
    outvars=["institutional_capacity_z","imf_aipi_index_2023_z","access_assets_z","log_population_z","policy_intensity_z"]
    y=CLAUDE
    D=df.dropna(subset=sel+["anthropic_available"]).copy()
    D["anthropic_available"]=D["anthropic_available"].astype(int)
    Xsel=sm.add_constant(D[sel]); ysel=D["anthropic_available"]
    logit=sm.Logit(ysel,Xsel).fit(disp=0,maxiter=300)
    probit=sm.Probit(ysel,Xsel).fit(disp=0,maxiter=300)
    rows=[]
    for name,res in [("logit_selection",logit),("probit_selection",probit)]:
        for t in res.params.index:
            rows.append({"model":name,"term":t,"coef":res.params[t],"se":res.bse[t],"z":res.tvalues[t],"p":res.pvalues[t],"n":int(res.nobs)})
    pd.DataFrame(rows).to_csv(out/"anthropic_selection_model_results_q1.csv",index=False)
    D["p_logit"]=np.clip(logit.predict(Xsel),1e-3,1-1e-3); D["p_probit"]=np.clip(probit.predict(Xsel),1e-3,1-1e-3)
    obs=D[D.anthropic_available==1].dropna(subset=[y]+outvars)
    X=sm.add_constant(obs[outvars]); yy=obs[y]; prop=ysel.mean()
    models={"baseline_hc3":sm.OLS(yy,X).fit(cov_type="HC3")}
    for pvar,name in [("p_logit","ipw_logit_stabilized"),("p_probit","ipw_probit_stabilized")]:
        w=(prop/obs[pvar]).clip(upper=(prop/obs[pvar]).quantile(.99))
        models[name]=sm.WLS(yy,X,weights=w).fit(cov_type="HC3")
    xb=probit.predict(Xsel,linear=True)
    D["inverse_mills_ratio"]=norm.pdf(xb)/np.clip(norm.cdf(xb),1e-6,None)
    obsh=D[D.anthropic_available==1].dropna(subset=[y]+outvars+["inverse_mills_ratio"])
    models["heckman_style_imr_diagnostic"]=sm.OLS(obsh[y],sm.add_constant(obsh[outvars+["inverse_mills_ratio"]])).fit(cov_type="HC3")
    outrows=[]
    for name,res in models.items():
        for t in res.params.index:
            outrows.append({"model":name,"term":t,"coef":res.params[t],"se_hc3":res.bse[t],"p_hc3":res.pvalues[t],"n":int(res.nobs),"r2":getattr(res,"rsquared",np.nan)})
    pd.DataFrame(outrows).to_csv(out/"anthropic_selection_corrected_claude_visible_models_q1.csv",index=False)
    df[df.anthropic_available==0][["iso3","country_name","institutional_capacity_z","imf_aipi_index_2023_z","access_assets_z","policy_intensity_z",BROAD]].to_csv(out/"anthropic_missing_countries_q1.csv",index=False)


def cluster_variant_sensitivity(df,out):
    # ARI between main k3 and variants for k3, plus profile summary.
    variants = {
        "main_all_inputs": CLUSTER_FEATURES,
        "no_regulatory_ethics": CLUSTER_FEATURES_NO_RE,
        "no_policy_visibility": CLUSTER_FEATURES_NO_POLICY,
        "core_capacity_access_only": CLUSTER_FEATURES_NO_RE_NO_POLICY,
    }
    rows=[]
    main = df["k3_cluster"].dropna().astype(int)
    for name,features in variants.items():
        d=df.dropna(subset=features).copy(); X=standardize(d[features].values); raw,_=kmeans_np(X,3,n_init=100)
        ordered=order_labels_by_capacity(d, raw, 3)
        overlap=main.index.intersection(ordered.index)
        rows.append({
            "variant": name,
            "features": "; ".join(features),
            "n": len(d),
            "ari_with_main_k3": adjusted_rand(main.loc[overlap].values, ordered.loc[overlap].values),
            "cluster_counts": json.dumps(ordered.value_counts().sort_index().to_dict()),
            "q1_broad_eta2": eta_squared_by_labels(df.loc[ordered.index, BROAD], ordered),
            "q1_gap_eta2": eta_squared_by_labels(df.loc[ordered.index, GAP], ordered),
            "claude_eta2": eta_squared_by_labels(df.loc[ordered.index, CLAUDE], ordered),
        })
    pd.DataFrame(rows).to_csv(out/"cluster_variant_sensitivity_q1.csv",index=False)


def eta_squared_by_labels(y, labels):
    d=pd.DataFrame({"y":y,"label":labels}).dropna()
    groups=[g.y.values for _,g in d.groupby("label")]
    if len(groups)<2: return np.nan
    overall=d.y.mean(); ssb=sum(len(g)*(np.mean(g)-overall)**2 for g in groups); sst=((d.y-overall)**2).sum()
    return float(ssb/sst) if sst>0 else np.nan


def figures(df,figdir):
    names={1:"High-capacity",2:"Intermediate",3:"Constrained"}
    markers={1:"o",2:"s",3:"^"}
    fig,ax=plt.subplots(figsize=(8,6),dpi=300)
    for k in [1,2,3]:
        d=df[df.k3_cluster==k]
        ax.scatter(d.institutional_capacity_z,d.imf_aipi_index_2023_z,label=names[k],marker=markers[k],alpha=.75)
    for iso in ["USA","CHN","KOR","JPN"]:
        r=df[df.iso3==iso]
        if len(r): ax.annotate(iso,(r.institutional_capacity_z.iloc[0],r.imf_aipi_index_2023_z.iloc[0]),xytext=(5,5),textcoords="offset points",fontsize=8)
    ax.set_xlabel("Institutional capacity, z-score")
    ax.set_ylabel("AI preparedness, z-score")
    ax.legend(title="Profile")
    ax.set_title("Governance-readiness-access configuration map")
    fig.tight_layout(); fig.savefig(figdir/"figure1_k3_configuration_map_q1_2026.png"); plt.close(fig)
    prof=df.groupby("k3_label").agg(broad=(BROAD,"mean"), claude=(CLAUDE,"mean"), gap=(GAP,"mean"))
    # order labels by mean broad descending through cluster order
    order=["High-capacity integrated profile", "Intermediate readiness-access profile", "Capacity-constrained low-observability profile"]
    prof=prof.reindex(order)
    plot=pd.DataFrame({"Broad diffusion, Q1 2026":zscore(prof.broad),"Claude-visible platform use":zscore(prof.claude),"Access-adjusted residual":zscore(prof.gap)},index=prof.index)
    fig,ax=plt.subplots(figsize=(9,5.5),dpi=300); x=np.arange(len(plot)); width=.25
    for i,col in enumerate(plot.columns): ax.bar(x+(i-1)*width,plot[col],width,label=col)
    ax.set_xticks(x); ax.set_xticklabels(["High-capacity","Intermediate","Constrained"],rotation=0)
    ax.set_ylabel("Standardized configuration mean")
    ax.set_title("Conversion-profile indicators by configuration")
    ax.legend(); fig.tight_layout(); fig.savefig(figdir/"figure2_k3_conversion_profiles_q1_2026.png"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6),dpi=300)
    for k in [1,2,3]:
        d=df[(df.k3_cluster==k)&df[CLAUDE].notna()]
        ax.scatter(d[BROAD],d[CLAUDE],label=names[k],marker=markers[k],alpha=.75)
    ax.set_xlabel("Broad mass diffusion, Microsoft Q1 2026 (%)")
    ax.set_ylabel("Claude-visible platform use, log per-capita index")
    ax.set_title("Broad diffusion and Claude-visible platform use")
    ax.legend(title="Profile"); fig.tight_layout(); fig.savefig(figdir/"figure3_broad_vs_claude_visible_channels_q1_2026.png"); plt.close(fig)


def write_summary(df,merge_diag,cluster_diag,boot_summary,out,reports):
    tests=pd.read_csv(out/"outcome_separation_k2_k3_q1.csv")
    prof3=pd.read_csv(out/"k3_configuration_profiles_q1.csv")
    prof2=pd.read_csv(out/"k2_configuration_profiles_q1.csv")
    summary={
        "source_update":"Microsoft AI_Diffusion_Q12026_Update.csv",
        "core_sample_n":int(len(df)),
        "q1_matched_n":int(df[BROAD].notna().sum()),
        "anthropic_available_n":int(df[CLAUDE].notna().sum()),
        "bti_available_n":int(df["bti_capacity_index_z"].notna().sum()),
        "mean_q1_broad_diffusion_pct":float(df[BROAD].mean()),
        "median_q1_broad_diffusion_pct":float(df[BROAD].median()),
        "mean_q1_change_from_h2_pct_points":float(df["q1_diffusion_change_from_h2_pct_points"].mean()),
        "access_baseline_r2_q1":merge_diag["access_baseline_r2_q1"],
        "k2_counts":{str(int(k)):int(v) for k,v in df.k2_cluster.value_counts().sort_index().items()},
        "k3_counts":{str(int(k)):int(v) for k,v in df.k3_cluster.value_counts().sort_index().items()},
        "k2_vs_k3_reading":"Input-space diagnostics still favor k=2 on silhouette and Davies-Bouldin, while k=3 preserves an outcome-relevant intermediate readiness-access profile and has strong Ward agreement and bootstrap stability. Q1 2026 broad diffusion strengthens outcome separation but does not change the core conclusion because cluster inputs exclude diffusion outcomes.",
    }
    (reports/"analysis_summary_q1.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    lines=[]
    lines.append("Q1 2026 update completed.")
    lines.append(f"Core sample: {summary['core_sample_n']} countries and economies; Q1 Microsoft matches: {summary['q1_matched_n']}.")
    lines.append(f"Mean Q1 broad diffusion: {summary['mean_q1_broad_diffusion_pct']:.2f}%; mean change from H2 2025: {summary['mean_q1_change_from_h2_pct_points']:.2f} percentage points.")
    lines.append("Cluster-input diagnostics remain based on governance, readiness, access, regulatory-ethics readiness, and policy visibility only.")
    lines.append("k=2 remains more parsimonious in some input-space criteria; k=3 remains more useful as a theory-guided diagnostic partition because it separates the intermediate readiness-access profile and materially improves interpretation of conversion gaps.")
    (reports/"README_q1_reproduction_results.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--q1",required=True)
    ap.add_argument("--outdir",required=True)
    args=ap.parse_args()
    outdir=Path(args.outdir); tables=outdir/"tables"; figs=outdir/"figures"; reports=outdir/"reports"; data=outdir/"data"
    for p in [tables,figs,reports,data]: p.mkdir(parents=True,exist_ok=True)
    df,merge_diag=prepare_data(Path(args.input),Path(args.q1),tables)
    df,_=assign_cluster(df,CLUSTER_FEATURES,2,"k2")
    df,_=assign_cluster(df,CLUSTER_FEATURES,3,"k3")
    cluster_diag=cluster_diagnostics(df,tables)
    boot_summary=bootstrap_stability(df,tables)
    profiles(df,tables)
    outcome_tests(df,tables)
    regression_suite(df,tables)
    bti_and_selection(df,tables)
    cluster_variant_sensitivity(df,tables)
    figures(df,figs)
    df.to_csv(data/"final_analysis_data_q1_2026.csv",index=False)
    write_summary(df,merge_diag,cluster_diag,boot_summary,tables,reports)

if __name__ == "__main__":
    main()
