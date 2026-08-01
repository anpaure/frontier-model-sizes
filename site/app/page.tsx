"use client";

import { useMemo, useState } from "react";
import rawData from "../public/data/forecast-model.json";
import rawUncertainty from "../public/data/predictive-uncertainty.json";

type FactorId = "aa" | "eci" | "price" | "horizon" | "compute" | "ikp" | "crowd";
type Weights = Record<FactorId, number>;
type Factor = { id: FactorId; label: string; shortLabel: string; description: string };
type Model = {
  id: string;
  name: string;
  shortName: string;
  provider: string;
  releaseDate: string;
  aaScore: number | null;
  eciScore: number | null;
  eciCi90: [number, number] | null;
  aaConfiguration: string | null;
  aaFallbackModel: string | null;
  lockedAnchor: boolean;
  disclosedT: number | null;
  currentEvidenceT: number;
  currentFinalT: number;
  factors: Record<FactorId, number | null>;
  crowd: { n: number; pooled: boolean; contributors: string[]; forecasts: string[] };
  methodNote: string;
};
type Dataset = {
  snapshotDate: string;
  unit: string;
  defaultWeights: Weights;
  factors: Factor[];
  presets: { id: string; label: string; weights: Weights }[];
  models: Model[];
  humanForecasts: { activeRecords: number; contributors: number };
  method: { combination: string; anchors: string; currentMix: string; ikp: string; noCotDates: string; noCotArchitecture: string; primaryEvidence: string; metrPrimary: string; operational: string; operationalTemporal: string; eciReproduction: string; eciFit: string; activePrice: string; historicalPrice: string; components: string; posttraining: string; aaExpansion: string; aaInference: string; activeTransport: string; computeDependency: string };
  noCotDateSignals: {
    models: number;
    exactDates: number;
    remainingMonthOnly: number;
    dateOnlyOverrides: number;
    parameterIdentitiesAdded: number;
    paperTimeDays: number;
    adjustedTimeDays: number;
    paperTokenDays: number;
    adjustedTokenDays: number;
    weightChanged: boolean;
  };
  noCotArchitectureSignals: {
    models: number;
    families: number;
    denseModels: number;
    moeModels: number;
    moeParetoModels: number;
    moeFactor: number;
    directDelta: number;
    directCi90: [number, number];
    paretoDelta: number;
    paretoCi90: [number, number];
    replacePooled: boolean;
  };
  frontierPrimarySignals: {
    directSolMinutes: number;
    gpt55ComparatorMinutes: number;
    currentProjectedSolMinutes: number;
    currentProjectedSolPriorT: number;
    modelLevelSolT: number;
    pooledElasticitySolT: number;
    moeElasticitySolT: number;
    rebasedPooledSolT: number;
    methodSpread: number;
    heldoutPredictions: number;
    heldoutDevelopers: number;
    horizonBetterProbability: number;
    incrementalCi90: [number, number];
    incrementalWeight: number;
    fableMythosSharedWeights: boolean;
    opusFallbackIsSharedBase: boolean;
  };
  metrPrimarySignals: {
    officialRows: number;
    uniqueSourceIds: number;
    fullScaffoldEntries: number;
    legacyExactRows: number;
    legacyRows: number;
    mismatchCount: number;
    from2023DoublingDays: number;
    from2023CiLowDays: number;
    from2023CiHighDays: number;
  };
  ikpSignals: {
    calibrationConfigurations: number;
    calibrationWeightBases: number;
    servingVariantsCollapsed: number;
    strictPredictionRows: number;
    overlapModels: number;
    overlapFamilies: number;
    existingMedianX: number;
    ikpMedianX: number;
    blendMedianX: number;
    fullBootstrapCi90: [number, number];
    chronologicalSubsetModels: number;
    chronologicalSubsetFamilies: number;
    chronologicalBootstrapCi90: [number, number];
    familiesImproved: number;
    signedErrorCorrelation: number;
    fableStrictT: number;
    fablePublishedT: number;
    fableStrictFormRangeT: [number, number];
    fableSourcePi90T: [number, number];
    fableRefusalRate: number;
    evidenceWeight: number;
    finalWeight: number;
    solObserved: boolean;
    conditionalGpqaModels: number;
    conditionalGpqaVendors: number;
    conditionalGpqaPassingSpecifications: number;
    conditionalMmluModels: number;
    conditionalMmluPassingSpecifications: number;
    conditionalMmluProPassingSpecifications: number;
    conditionalSignalCorroborated: boolean;
    conditionalWeightChanged: boolean;
    staleUpstreamNarrativeClaims: number;
  };
  operationalSignals: {
    snapshotDate: string;
    epochCalibrationCheckpoints: number;
    developerFamilies: number;
    priceMedianHeldoutErrorX: number;
    tokSIncrementalWeight: number;
  };
  openRouterTemporalSignals: {
    immutableSnapshots: number;
    currentDailyRows: number;
    historyDailyRows: number;
    serviceTierRows: { default: number; priority: number; flex: number };
    multiTierEndpointModels: number;
    unmatchedEndpointModels: number;
    familyPriceMae: number;
    familyPriceTokSMae: number;
    chronologicalPriceMae: number;
    chronologicalPriceTokSMae: number;
    medianModelWithinWeekMaxOverMin: number;
    p90ModelWithinWeekMaxOverMin: number;
    focalModels: {
      model: string;
      openrouter_model_id: string;
      refresh_max_over_min: number;
      refresh_median_tps: number;
      refreshes: number;
      within_week_dates: number;
      within_week_max_over_min: number;
      within_week_median_tps: number;
    }[];
    tokSIncrementalWeight: number;
  };
  openRouterOfficialSignals: {
    modelRequests: number;
    successfulModelRequests: number;
    officialEndpointRows: number;
    frontendEndpointTierRows: number;
    officialPriceExactShare: number;
    comparisonGroupCounts: { exact: number; frontend_only: number; official_only: number; signature_mismatch: number };
    endpointTierRows: number;
    endpointTierServiceTierCounts: { default: number; flex: number; priority: number };
    highContextPriceRows: number;
    focalModels: Record<string, { official_rows: number; frontend_rows: number; non_exact_groups: string[] }>;
    incrementalForecastWeight: number;
  };
  openRouterRequestWeightedSignals: {
    completeCheckpoints: number;
    completeFamilies: number;
    minimumRequests: number;
    supportedCandidates: string[];
    familyPriceMedianX: number;
    familyLatencyMedianX: number;
    familyLatencyCi90: [number, number];
    incrementalWeight: number;
  };
  eciReproductionSignals: {
    officialCommit: string;
    inputRows: number;
    reproducedModels: number;
    publishedScoreMatches: number;
    maximumPublishedScoreDifference: number;
    publishedReleaseDatesUsed: number;
    canonicalInputDateFallbacks: number;
    preservedDateDisagreements: number;
  };
  eciFitSignals: {
    historicalSnapshots: number;
    historicalScoreRows: number;
    firstObservedTargets: number;
    selectionTargets: number;
    prospectiveTargets: number;
    selectedCandidate: string;
    baselineSelectionMedianX: number;
    candidateSelectionMedianX: number;
    selectionCi90: [number, number];
    baselineProspectiveMae: number;
    candidateProspectiveMae: number;
    fableExtrapolationPoints: number;
    solExtrapolationPoints: number;
    fableSensitivityRatio: number;
    solSensitivityRatio: number;
    changeLiveForm: boolean;
  };
  componentSignals: {
    expandedParameterCheckpoints: number;
    exactEpochAdditions: number;
    activeParameterCheckpoints: number;
    eligibleBenchmarks: number;
    familywiseSupported: number;
    incrementalWeight: number;
    bestUncorrectedBenchmark: string;
    bestAdjustedP: number;
    multivariateOuterPredictions: number;
    multivariateFamilies: number;
    multivariateBaselineMedianX: number;
    multivariateCandidateMedianX: number;
    multivariateCi90: [number, number];
    narrowCiOnlyPredictions: number;
    narrowCiOnlyFamilies: number;
    narrowCiOnlyBaselineMedianX: number;
    narrowCiOnlyCandidateMedianX: number;
    narrowCiOnlyCi90: [number, number];
    multivariateIncrementalWeight: number;
    targetSensitivities: {
      model: string;
      observed_selected_count: number;
      narrow_ci_training_observed_count: number;
      component_adjustment_factor: number;
      narrow_ci_training_adjustment_factor: number;
      full_vs_narrow_ci_direction_agrees: boolean;
    }[];
  };
  posttrainingSignals: {
    epochBaseLinks: number;
    sameParameterBothOpenLinks: number;
    candidateOpenLanguageLinks: number;
    measuredEdges: number;
    measuredBases: number;
    measuredDevelopers: number;
    eciPredictionEdges: number;
    aaPredictionEdges: number;
    noCotEdges: number;
    metrEdges: number;
    sameWeightReasoningPairs: number;
    sameWeightReasoningCreators: number;
    sameWeightReasoningMedianUplift: number;
    eciMedianImpliedRatio: number;
    eciCollapseCi90: [number, number];
    aaCollapseCi90: [number, number];
    knowledgeMedianUplift: number;
    otherMedianUplift: number;
    incrementalWeight: number;
    publiclyVerifiedProprietarySharedBase: boolean;
  };
  aaExpansionSignals: {
    currentCheckpoints: number;
    exactEpochCheckpoints: number;
    reconciledOverlaps: number;
    expandedCheckpoints: number;
    eligiblePredictions: number;
    frontierLikePredictions: number;
    incrementalWeight: number;
  };
  aaInferenceSignals: {
    rawModels: number;
    openWeightConfigurations: number;
    uniqueCheckpoints: number;
    tokenCoveredCheckpoints: number;
    allReasoningPairs: number;
    exactWeightPairs: number;
    pairCreators: number;
    equalCreatorMedianUplift: number;
    exactEpochCrosschecks: number;
    metadataDisagreements: number;
    portableFrontierBaselineMedianX: number;
    portableFrontierCandidateMedianX: number;
    incrementalTokenWeight: number;
    incrementalReasoningWeight: number;
  };
  aaOperationalSignals: {
    priceCheckpoints: number;
    speedCheckpoints: number;
    providerMedianPriceCheckpoints: number;
    firstPartyPriceCheckpoints: number;
    exactOpenRouterOverlap: number;
    priceSpearman: number;
    speedSpearman: number;
    providerMedianFrontierBaselineMedianX: number;
    providerMedianFrontierCandidateMedianX: number;
    firstPartyFrontierBaselineMedianX: number;
    firstPartyFrontierCandidateMedianX: number;
    incrementalPriceWeight: number;
    incrementalSpeedWeight: number;
    incrementalLatencyWeight: number;
  };
  activeParameterSignals: {
    totalParameterCheckpoints: number;
    activeParameterCheckpoints: number;
    developers: number;
    chronologicalPredictions: number;
    frontierLikePredictions: number;
    activeMedianErrorX: number;
    samePanelTotalMedianErrorX: number;
    activeComparisonCi90: [number, number];
    highSparsityPredictions: number;
    transportCandidateMedianErrorX: number;
    transportBaselineMedianErrorX: number;
    transportCi90: [number, number];
    predictedK3ActiveB: number;
    k3ImpliedRatio: number;
    targetSensitivities: {
      model: string;
      release_date: string;
      aa_score: number;
      predicted_active_b: number;
      k3_anchored_total_t: number;
      k2_k3_structural_midpoint_total_t: number;
      status: string;
    }[];
    incrementalWeight: number;
    estimatedTargetComputeModels: number;
    disclosedTargetComputeModels: number;
    computeIndependent: boolean;
  };
  activePriceSignals: {
    exactActiveMatches: number;
    disclosedActiveMatches: number;
    denseConfigControls: number;
    developers: number;
    chronologicalPredictions: number;
    predictionDevelopers: number;
    highSparsityPredictions: number;
    highSparsityDevelopers: number;
    candidateMedianErrorX: number;
    baselineMedianErrorX: number;
    transportCi90: [number, number];
    performanceGatePassed: boolean;
    coverageGatePassed: boolean;
    incrementalWeight: number;
    prospectivePriceBacktest: boolean;
    targetSensitivities: {
      model: string;
      release_date: string;
      aa_score: number;
      blended_price_usd_per_mtoken: number;
      price_over_training_max: number;
      predicted_active_date_price_b: number;
      predicted_active_score_date_price_b: number;
      k3_anchored_total_date_price_t: number;
      k3_anchored_total_score_date_price_t: number;
      status: string;
    }[];
  };
  historicalPriceSignals: {
    ledgerModels: number;
    changePoints: number;
    calibrationCheckpoints: number;
    missingCalibrationAliases: number;
    oneDayTotalRows: number;
    oneDayActiveRows: number;
    oneDayTotalPredictions: number;
    oneDayTotalMedianErrorX: number;
    oneDayDateOnlyMedianErrorX: number;
    totalRobustAcrossWindows: boolean;
    activeIncrementalRobust: boolean;
    incrementalWeight: number;
    targetSensitivities: {
      model: string;
      release_date: string;
      openrouter_first_seen: string;
      aa_score: number;
      first_day_prompt_price_usd_per_mtoken: number;
      first_day_completion_price_usd_per_mtoken: number;
      first_day_blended_price_usd_per_mtoken: number;
      predicted_active_date_historical_price_b: number;
      k3_anchored_total_date_historical_price_t: number;
      predicted_active_score_date_historical_price_b: number;
      k3_anchored_total_score_date_historical_price_t: number;
      status: string;
    }[];
  };
  sources: Record<string, { name: string; sha256: string }>;
};

type PredictiveUncertainty = {
  method: {
    primary_cohort: string;
    family_policy: string;
    crowd_policy: string;
  };
  cohorts: Record<string, {
    rows: number;
    families: number;
    chronological_coverage: Record<string, {
      eligible_tests: number;
      test_families: number;
      observed_coverage: number | null;
    }>;
  }>;
  targets: {
    model_id: string;
    center_t: number;
    displayed_final_center_t: number;
    calibration_rows: number;
    calibration_families: number;
    calibration_developers: number;
    intervals: Record<string, {
      multiplicative_factor: number;
      low_t: number;
      high_t: number;
    }>;
  }[];
  decision: {
    use_frontier_moe_reasoning_cohort: boolean;
    change_central_forecasts: boolean;
  };
  source_files: Record<string, string>;
};

const data = rawData as Dataset;
const uncertainty = rawUncertainty as PredictiveUncertainty;
const factorClass: Record<FactorId, string> = {
  aa: "factor-aa",
  eci: "factor-eci",
  price: "factor-price",
  horizon: "factor-horizon",
  compute: "factor-compute",
  ikp: "factor-ikp",
  crowd: "factor-crowd",
};

const formatT = (value: number) => `${value.toFixed(1)}T`;
const formatPct = (value: number) => `${Math.min(100, Math.max(0, value)).toFixed(4)}%`;
const BASE_AXIS_MAX_T = 5;
const axisTicks = (maximum: number) => {
  if (maximum === BASE_AXIS_MAX_T) return [0, 1.25, 2.5, 3.75, 5];
  return [...new Set([0, maximum / 2, BASE_AXIS_MAX_T, maximum])].sort((a, b) => a - b);
};

function axisMaximum(peak: number) {
  if (peak <= BASE_AXIS_MAX_T) return BASE_AXIS_MAX_T;
  return peak * 1.06;
}
const formatDate = (value: string) =>
  new Intl.DateTimeFormat("en", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" }).format(
    new Date(`${value}T00:00:00Z`),
  );

function forecast(model: Model, weights: Weights) {
  if (model.lockedAnchor) return model.disclosedT ?? model.currentFinalT;
  const evidence = data.factors.filter((factor) => factor.id !== "crowd");
  const availableEvidence = evidence.filter((factor) => model.factors[factor.id] != null && weights[factor.id] > 0);
  const requestedEvidenceWeight = evidence.reduce((sum, factor) => sum + Math.max(0, weights[factor.id]), 0);
  const availableEvidenceWeight = availableEvidence.reduce((sum, factor) => sum + weights[factor.id], 0);
  const effective = availableEvidence.map((factor) => ({
    factor,
    weight: availableEvidenceWeight ? requestedEvidenceWeight * weights[factor.id] / availableEvidenceWeight : 0,
  }));
  const crowd = data.factors.find((factor) => factor.id === "crowd")!;
  if (model.factors.crowd != null && weights.crowd > 0) effective.push({ factor: crowd, weight: weights.crowd });
  const total = effective.reduce((sum, item) => sum + item.weight, 0);
  if (!total) return model.currentFinalT;
  return Math.exp(
    effective.reduce((sum, item) => sum + (item.weight / total) * Math.log(model.factors[item.factor.id] as number), 0),
  );
}

function normalizedWeights(model: Model, weights: Weights) {
  const evidence = data.factors.filter((factor) => factor.id !== "crowd");
  const availableEvidence = evidence.filter((factor) => model.factors[factor.id] != null && weights[factor.id] > 0);
  const requestedEvidenceWeight = evidence.reduce((sum, factor) => sum + Math.max(0, weights[factor.id]), 0);
  const availableEvidenceWeight = availableEvidence.reduce((sum, factor) => sum + weights[factor.id], 0);
  const raw = Object.fromEntries(data.factors.map((factor) => [factor.id, 0])) as Record<FactorId, number>;
  for (const factor of availableEvidence) {
    raw[factor.id] = availableEvidenceWeight ? requestedEvidenceWeight * weights[factor.id] / availableEvidenceWeight : 0;
  }
  if (model.factors.crowd != null && weights.crowd > 0) raw.crowd = weights.crowd;
  const total = Object.values(raw).reduce((sum, value) => sum + value, 0);
  return Object.fromEntries(data.factors.map((factor) => [factor.id, total ? raw[factor.id] / total : 0])) as Record<FactorId, number>;
}

export default function Home() {
  const [weights, setWeights] = useState<Weights>({ ...data.defaultWeights });
  const [selectedId, setSelectedId] = useState(data.models[0].id);
  const [showSpread, setShowSpread] = useState(true);

  const results = useMemo(
    () =>
      data.models
        .map((model) => ({ model, value: forecast(model, weights), delta: forecast(model, weights) / model.currentFinalT - 1 }))
        .sort((a, b) => b.value - a.value),
    [weights],
  );
  const selected = data.models.find((model) => model.id === selectedId) ?? data.models[0];
  const selectedResult = results.find((result) => result.model.id === selected.id)!;
  const selectedUncertainty = uncertainty.targets.find((row) => row.model_id === selected.id);
  const activeSensitivity = data.activeParameterSignals.targetSensitivities.find((row) => row.model === selected.name);
  const activePriceSensitivity = data.activePriceSignals.targetSensitivities.find((row) => row.model === selected.name);
  const historicalPriceSensitivity = data.historicalPriceSignals.targetSensitivities.find((row) => row.model === selected.name);
  const selectedTemporalName = selected.name === "Claude Opus 4.7 / 4.8 shared base" ? "Claude Opus 4.8" : selected.name;
  const throughputStability = data.openRouterTemporalSignals.focalModels.find((row) => row.model === selectedTemporalName);
  const effective = normalizedWeights(selected, weights);
  const peak = Math.max(
    ...results.map((result) => result.value),
    ...data.models.map((model) => model.currentFinalT),
  );
  const maximum = axisMaximum(peak);
  const pastBaseline = maximum > BASE_AXIS_MAX_T;
  const globalTotal = Object.values(weights).reduce((sum, value) => sum + value, 0);
  const activePreset = data.presets.find((preset) =>
    data.factors.every((factor) => Math.abs(preset.weights[factor.id] - weights[factor.id]) < 0.0001),
  )?.id;

  const setWeight = (factor: FactorId, value: number) => setWeights((current) => ({ ...current, [factor]: value }));

  return (
    <main>
      <header className="masthead">
        <div className="navline">
          <a className="brand" href="#top" aria-label="Frontier Parameter Lab home">
            <span className="brand-mark">FP</span>
            <span>Frontier Parameter Lab</span>
          </a>
          <div className="sync-status"><span className="sync-dot" />Pipeline synced · {formatDate(data.snapshotDate)}</div>
        </div>
        <div className="hero" id="top">
          <div>
            <p className="eyebrow">Interactive posterior explorer</p>
            <h1>Change the evidence.<br />Watch the frontier move.</h1>
          </div>
          <div className="hero-copy">
            <p>Reweight seven audited, partly correlated views of model scale and see the implied total parameter counts update instantly.</p>
            <div className="hero-metrics">
              <span><strong>{data.models.length}</strong> frontier bases</span>
              <span><strong>{data.humanForecasts.activeRecords}</strong> human forecasts</span>
              <span><strong>{data.humanForecasts.contributors}</strong> contributors</span>
              <span><strong>{data.operationalSignals.epochCalibrationCheckpoints}</strong> price checks</span>
              <span><strong>{data.openRouterTemporalSignals.immutableSnapshots}</strong> OR snapshots</span>
              <span><strong>{(100 * data.openRouterOfficialSignals.officialPriceExactShare).toFixed(1)}%</strong> OR price exact</span>
              <span><strong>{data.eciReproductionSignals.reproducedModels}</strong> reproduced ECI</span>
              <span><strong>{data.eciFitSignals.historicalSnapshots}</strong> archived ECI vintages</span>
              <span><strong>{data.eciFitSignals.firstObservedTargets}</strong> vintage fit tests</span>
              <span><strong>{data.componentSignals.activeParameterCheckpoints}</strong> component checks</span>
              <span><strong>{data.componentSignals.multivariateOuterPredictions}</strong> nested ECI folds</span>
              <span><strong>{data.posttrainingSignals.measuredBases}</strong> lineage bases</span>
              <span><strong>{data.aaExpansionSignals.expandedCheckpoints}</strong> AA checks</span>
              <span><strong>{data.aaInferenceSignals.rawModels}</strong> AA records</span>
              <span><strong>{data.aaOperationalSignals.exactOpenRouterOverlap}</strong> cross-source checks</span>
              <span><strong>{data.activeParameterSignals.activeParameterCheckpoints}</strong> active-param checks</span>
              <span><strong>{data.activePriceSignals.exactActiveMatches}</strong> active-price labels</span>
              <span><strong>{data.historicalPriceSignals.changePoints.toLocaleString("en-US")}</strong> dated prices</span>
              <span><strong>{data.noCotDateSignals.exactDates}/{data.noCotDateSignals.models}</strong> exact no-CoT dates</span>
              <span><strong>{data.noCotArchitectureSignals.models}</strong> architecture checks</span>
              <span><strong>{data.frontierPrimarySignals.directSolMinutes.toFixed(1)} min</strong> direct Sol no-CoT</span>
              <span><strong>{data.ikpSignals.overlapModels}</strong> IKP overlap tests</span>
              <span><strong>{data.ikpSignals.conditionalGpqaModels}</strong> conditional IKP tests</span>
              <span><strong>{formatT(data.ikpSignals.fableStrictT)}</strong> strict IKP Fable</span>
            </div>
          </div>
        </div>
      </header>

      <section className="lab-shell" aria-label="Weighting laboratory">
        <aside className="control-panel">
          <div className="panel-heading">
            <div>
              <p className="section-kicker">01 · Evidence mix</p>
              <h2>Weight controls</h2>
            </div>
            <button className="text-button" onClick={() => setWeights({ ...data.defaultWeights })}>Reset</button>
          </div>

          <div className="presets" aria-label="Weight presets">
            {data.presets.map((preset) => (
              <button
                key={preset.id}
                className={activePreset === preset.id ? "preset active" : "preset"}
                onClick={() => setWeights({ ...preset.weights })}
              >
                {preset.label}
              </button>
            ))}
          </div>

          <div className="sliders">
            {data.factors.map((factor) => {
              const normalized = globalTotal ? (weights[factor.id] / globalTotal) * 100 : 0;
              return (
                <label className="slider-row" key={factor.id}>
                  <span className="slider-label">
                    <span><i className={`factor-dot ${factorClass[factor.id]}`} />{factor.label}</span>
                    <strong>{normalized.toFixed(1)}%</strong>
                  </span>
                  <span className="slider-input-row">
                    <input
                      aria-label={`${factor.label} relative weight`}
                      type="range"
                      min="0"
                      max="100"
                      step="0.5"
                      value={weights[factor.id]}
                      onChange={(event) => setWeight(factor.id, Number(event.target.value))}
                    />
                    <input
                      className="weight-number"
                      aria-label={`${factor.label} weight value`}
                      type="number"
                      min="0"
                      max="100"
                      step="0.5"
                      value={weights[factor.id]}
                      onChange={(event) => setWeight(factor.id, Math.max(0, Number(event.target.value)))}
                    />
                  </span>
                  <small>{factor.description}</small>
                </label>
              );
            })}
          </div>
          {!globalTotal && <p className="warning">Set at least one factor above zero. Current workbook values are shown meanwhile.</p>}
          <p className="control-note">Unavailable factors are omitted model by model; the remaining weights renormalize automatically.</p>
        </aside>

        <div className="chart-panel">
          <div className="panel-heading chart-heading">
            <div>
              <p className="section-kicker">02 · Live output</p>
              <h2>Implied total parameters</h2>
            </div>
            <div className="chart-tools">
              <label className="toggle">
                <input type="checkbox" checked={showSpread} onChange={(event) => setShowSpread(event.target.checked)} />
                <span />Factor spread
              </label>
            </div>
          </div>
          <div className="axis" data-axis-max={maximum} data-past-baseline={pastBaseline}>
            {axisTicks(maximum).map((tick, index, ticks) => (
              <span
                key={tick}
                className={`${index === 0 ? "axis-first " : ""}${index === ticks.length - 1 ? "axis-last " : ""}${pastBaseline && tick === BASE_AXIS_MAX_T ? "baseline-tick" : ""}`.trim()}
                style={{ left: formatPct((tick / maximum) * 100) }}
              >
                {formatT(tick)}
                {pastBaseline && tick === BASE_AXIS_MAX_T && <small>baseline</small>}
              </span>
            ))}
          </div>
          <div className="bars">
            {results.map(({ model, value, delta }) => {
              const availableValues = Object.values(model.factors).filter((item): item is number => item != null);
              const low = Math.min(...availableValues);
              const high = Math.max(...availableValues);
              const clippedLow = Math.min(low, maximum);
              const clippedHigh = Math.min(high, maximum);
              const displayedDelta = Math.abs(delta) < 0.0005 ? 0 : delta;
              return (
                <button
                  className={selected.id === model.id ? "model-bar selected" : "model-bar"}
                  key={model.id}
                  onClick={() => setSelectedId(model.id)}
                  aria-label={`Inspect ${model.name}, ${formatT(value)}`}
                >
                  <span className="bar-label"><strong>{model.shortName}</strong><small>{model.provider}</small></span>
                  <span className="bar-stage">
                    {pastBaseline && <span className="normal-ceiling" style={{ left: formatPct((BASE_AXIS_MAX_T / maximum) * 100) }} />}
                    {showSpread && !model.lockedAnchor && (
                      <span
                        className={high > maximum ? "factor-spread overflowing" : "factor-spread"}
                        style={{ left: formatPct((clippedLow / maximum) * 100), width: formatPct(((clippedHigh - clippedLow) / maximum) * 100) }}
                        title={high > maximum ? `Factor range extends to ${formatT(high)}` : undefined}
                      />
                    )}
                    <span className="current-marker" style={{ left: formatPct((model.currentFinalT / maximum) * 100) }} />
                    <span className="bar-fill" style={{ width: formatPct((value / maximum) * 100) }} />
                    {value > BASE_AXIS_MAX_T && (
                      <span
                        className="bar-overflow"
                        style={{
                          left: formatPct((BASE_AXIS_MAX_T / maximum) * 100),
                          width: formatPct(((value - BASE_AXIS_MAX_T) / maximum) * 100),
                        }}
                      />
                    )}
                  </span>
                  <span className="bar-value">
                    <strong>{formatT(value)}</strong>
                    {model.lockedAnchor ? <em className="anchor-tag">disclosed</em> : <em className={displayedDelta === 0 ? "flat" : displayedDelta > 0 ? "up" : "down"}>{displayedDelta > 0 ? "+" : ""}{(displayedDelta * 100).toFixed(1)}%</em>}
                  </span>
                </button>
              );
            })}
          </div>
          <div className="chart-legend">
            <span><i className="legend-bar" />Scenario</span>
            <span><i className="legend-mark" />Published mix</span>
            <span><i className="legend-range" />Factor range</span>
          </div>
        </div>
      </section>

      <section className="model-inspector" aria-live="polite">
        <div className="inspector-intro">
          <p className="section-kicker">03 · Selected model</p>
          <div className="selected-title">
            <div>
              <h2>{selected.name}</h2>
              <p>{selected.provider} · released {formatDate(selected.releaseDate)}</p>
            </div>
            <div className="selected-forecast">
              <span>Scenario forecast</span>
              <strong>{formatT(selectedResult.value)}</strong>
              <small>Published mix {formatT(selected.currentFinalT)}</small>
            </div>
          </div>
          <p className="model-note">{selected.methodNote}</p>
          {selectedUncertainty && (
            <div className="active-transport-callout">
              <span>Held-out predictive uncertainty · evidence center</span>
              <strong>80%: {formatT(selectedUncertainty.intervals["80"].low_t)}–{formatT(selectedUncertainty.intervals["80"].high_t)}</strong>
              <small>Descriptive 50% band: {formatT(selectedUncertainty.intervals["50"].low_t)}–{formatT(selectedUncertainty.intervals["50"].high_t)} · it undercovered in sequential backtests · {selectedUncertainty.calibration_developers} independent frontier developers · crowd does not narrow the band</small>
            </div>
          )}
          {selected.name === "GPT-5.6 Sol" && (
            <div className="active-transport-callout">
              <span>Official no-CoT measurement · 0% incremental weight</span>
              <strong>{data.frontierPrimarySignals.directSolMinutes.toFixed(1)} min</strong>
              <small>{data.frontierPrimarySignals.heldoutPredictions} chronological developer-holdouts · parameter mappings span {formatT(data.frontierPrimarySignals.modelLevelSolT)}–{formatT(data.frontierPrimarySignals.rebasedPooledSolT)}</small>
            </div>
          )}
          {selected.name === "Claude Fable 5" && data.frontierPrimarySignals.fableMythosSharedWeights && (
            <div className="active-transport-callout">
              <span>Official parameter identity</span>
              <strong>Fable = Mythos weights</strong>
              <small>Opus 4.8 fallback is serving behavior, not another base model.</small>
            </div>
          )}
          {selected.name === "Claude Opus 5" && selected.eciCi90 && (
            <div className="active-transport-callout">
              <span>New-base regression inputs</span>
              <strong>AA {selected.aaScore?.toFixed(1)} · ECI {selected.eciScore?.toFixed(1)}</strong>
              <small>ECI 90% CI {selected.eciCi90[0].toFixed(1)}–{selected.eciCi90[1].toFixed(1)} · AA {selected.aaConfiguration?.replace(/_/g, " ")} with {selected.aaFallbackModel} fallback · parameter count undisclosed · no crowd, IKP, No-CoT, or METR model-level measurement</small>
            </div>
          )}
          {activeSensitivity && (
            <div className="active-transport-callout">
              <span>Architecture sensitivity · 0% live weight</span>
              <strong>{formatT(activeSensitivity.k3_anchored_total_t)}</strong>
              <small>K3-anchored active-parameter transport · predicted active {activeSensitivity.predicted_active_b.toFixed(1)}B</small>
            </div>
          )}
          {activePriceSensitivity && (
            <div className="active-transport-callout">
              <span>API-price active-capacity sensitivity · 0% live weight</span>
              <strong>{formatT(activePriceSensitivity.k3_anchored_total_score_date_price_t)}</strong>
              <small>Current-price diagnostic · target price is {activePriceSensitivity.price_over_training_max.toFixed(1)}× the common training maximum</small>
            </div>
          )}
          {historicalPriceSensitivity && (
            <div className="active-transport-callout">
              <span>Launch-price sensitivity · 0% incremental weight</span>
              <strong>{formatT(historicalPriceSensitivity.k3_anchored_total_score_date_historical_price_t)}</strong>
              <small>First observed day · ${historicalPriceSensitivity.first_day_prompt_price_usd_per_mtoken.toFixed(1)} input / ${historicalPriceSensitivity.first_day_completion_price_usd_per_mtoken.toFixed(1)} output per million tokens</small>
            </div>
          )}
          {throughputStability && (
            <div className="active-transport-callout">
              <span>Default-tier serving stability · 0% tok/s weight</span>
              <strong>{throughputStability.within_week_median_tps.toFixed(1)} tok/s</strong>
              <small>{throughputStability.within_week_dates} daily points · {throughputStability.within_week_max_over_min.toFixed(1)}× within-week max/min · {throughputStability.refreshes} immutable refreshes</small>
            </div>
          )}
          {selected.lockedAnchor && <div className="anchor-callout"><strong>Fixed disclosed anchor.</strong> Weight changes do not override a public total parameter count.</div>}
        </div>
        <div className="factor-grid">
          {data.factors.map((factor) => {
            const value = selected.factors[factor.id];
            return (
              <article className="factor-card" key={factor.id}>
                <div><i className={`factor-dot ${factorClass[factor.id]}`} /><span>{factor.shortLabel}</span></div>
                <strong>{value == null ? "—" : formatT(value)}</strong>
                <small>{value == null ? (factor.id === "crowd" && selected.crowd.n > 0 ? `${selected.crowd.n} forecast${selected.crowd.n === 1 ? "" : "s"} · not pooled` : "No model-level observation") : `${(effective[factor.id] * 100).toFixed(1)}% effective weight`}</small>
              </article>
            );
          })}
        </div>
        {selected.crowd.n > 0 && (
          <div className="crowd-strip">
            <div><span>Human pool</span><strong>n={selected.crowd.n}</strong></div>
            <div className="forecast-chips">
              {selected.crowd.contributors.map((contributor, index) => <span key={`${contributor}-${index}`}>{contributor} <b>{selected.crowd.forecasts[index]}</b></span>)}
            </div>
          </div>
        )}
      </section>

      <section className="method-section">
        <div>
          <p className="section-kicker">04 · What the controls mean</p>
          <h2>A geometric ensemble, not a parameter census.</h2>
        </div>
        <div className="method-copy">
          <p>{data.method.combination}</p>
          <p>{data.method.anchors}</p>
          <p>{data.method.currentMix}</p>
          <p>{data.method.ikp}</p>
          <p>{data.method.noCotDates}</p>
          <p>{data.method.noCotArchitecture}</p>
          <p>{data.method.primaryEvidence}</p>
          <p>{data.method.metrPrimary}</p>
          <p>{data.method.operational}</p>
          <p>{data.method.operationalTemporal}</p>
          <p>{data.method.eciReproduction}</p>
          <p>{data.method.eciFit}</p>
          <p>{data.method.activePrice}</p>
          <p>{data.method.historicalPrice}</p>
          <p>{data.method.components}</p>
          <p>{data.method.posttraining}</p>
          <p>{data.method.aaExpansion}</p>
          <p>{data.method.aaInference}</p>
          <p>{data.method.activeTransport}</p>
          <p>{data.method.computeDependency}</p>
        </div>
      </section>

      <footer>
        <div><span className="brand-mark">FP</span><strong>Frontier Parameter Lab</strong></div>
        <p>Generated directly from the audited forecast pipeline. Values are system-equivalent estimates unless marked disclosed.</p>
        <p className="hash">Ledger {data.sources.forecastLedger.sha256.slice(0, 10)} · Workbook {data.sources.finalWorkbook.sha256.slice(0, 10)} · Opus 5 evidence {data.sources.claudeOpus5Evidence.sha256.slice(0, 10)} · IKP {data.sources.ikpAudit.sha256.slice(0, 10)} · IKP conditional {data.sources.ikpConditionalAudit.sha256.slice(0, 10)} · Frontier primary {data.sources.frontierPrimaryAudit.sha256.slice(0, 10)} · METR primary {data.sources.metrPrimaryAudit.sha256.slice(0, 10)} · No-CoT dates {data.sources.noCotExactDateAudit.sha256.slice(0, 10)} · No-CoT architecture {data.sources.noCotArchitectureAudit.sha256.slice(0, 10)} · OpenRouter {data.sources.openRouterAudit.sha256.slice(0, 10)} · OR official prices {data.sources.openRouterOfficialAudit.sha256.slice(0, 10)} · OR temporal {data.sources.openRouterTemporalAudit.sha256.slice(0, 10)} · OR history {data.sources.openRouterHistoryManifest.sha256.slice(0, 10)} · OR dated prices {data.sources.openRouterHistoricalPriceAudit.sha256.slice(0, 10)} · HF configs {data.sources.hfArchitectureAudit.sha256.slice(0, 10)} · ECI reproduction {data.sources.eciReproductionAudit.sha256.slice(0, 10)} · ECI vintages {data.sources.eciFitTournament.sha256.slice(0, 10)} · Active price {data.sources.openRouterActivePriceAudit.sha256.slice(0, 10)} · ECI components {data.sources.eciComponentAudit.sha256.slice(0, 10)} · ECI multivariate {data.sources.eciMultivariateAudit.sha256.slice(0, 10)} · Lineage {data.sources.posttrainingLineageAudit.sha256.slice(0, 10)} · AA detailed {data.sources.aaInferenceAudit.sha256.slice(0, 10)} · AA operations {data.sources.aaOperationalAudit.sha256.slice(0, 10)} · Active transport {data.sources.activeTransportAudit.sha256.slice(0, 10)}</p>
      </footer>
    </main>
  );
}
