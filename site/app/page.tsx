"use client";

import { type CSSProperties, useEffect, useMemo, useState } from "react";
import rawData from "../public/data/parameter-scatter.json";

export const dynamic = "force-static";

type FactorId = "aa" | "eci" | "price" | "horizon" | "compute" | "ikp" | "crowd";
type Weights = Record<FactorId, number>;
type Factor = { id: FactorId; label: string; shortLabel: string; description: string };
type ScatterModel = {
  id: string;
  forecastModelId: string;
  name: string;
  shortName: string;
  organization: string;
  organizationGroup: string;
  releaseDate: string;
  parameterT: number;
  parameterKind: "forecast" | "disclosed";
  parameterSource: string;
  lockedAnchor: boolean;
  currentFinalT: number;
  publicEstimateT: number | null;
  factors: Record<FactorId, number | null>;
  factorWeights: Weights;
  signals: {
    aaScore: number | null;
    eciScore: number | null;
    noCotMinutes: number | null;
    metrP50Minutes: number | null;
    trainingComputeFlop: number | null;
    activeParametersB: number | null;
  };
  architecture: { moe: boolean; reasoning: boolean };
};
type Dataset = {
  snapshotDate: string;
  title: string;
  parameterPolicy: string;
  counts: { models: number; frontier: number; calibration: number; disclosedAnchors: number };
  organizationGroups: { id: string; color: string }[];
  factors: Factor[];
  defaultWeights: Weights;
  presets: { id: string; label: string; weights: Weights }[];
  models: ScatterModel[];
  sources: Record<string, { name: string; sha256: string }>;
};
const data = rawData as Dataset;
const ONE_X = Object.fromEntries(data.factors.map((factor) => [factor.id, 1])) as Weights;
const BASE_MIN_T = 1;
const BASE_MAX_T = 7;
const FACTOR_COLORS: Record<FactorId, string> = {
  aa: "#00A5A6",
  eci: "#6A3ECB",
  price: "#C48A17",
  horizon: "#E45C45",
  compute: "#377FC4",
  ikp: "#C84B8F",
  crowd: "#5E8C42",
};

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const formatDate = (date: string) =>
  new Intl.DateTimeFormat("en", { month: "long", day: "numeric", year: "numeric", timeZone: "UTC" }).format(
    new Date(`${date}T00:00:00Z`),
  );
const formatParameter = (valueT: number) => valueT >= 1 ? `${valueT.toFixed(1)}T` : `${(valueT * 1000).toFixed(0)}B`;
const formatFactorRange = (range: readonly [number, number]) => `${range[0].toFixed(1)}–${range[1].toFixed(1)}T`;
const formatMultiplier = (value: number) => `${Number(value.toPrecision(2))}x`;
const formatChange = (valueT: number, baselineT: number) => {
  const delta = (valueT / baselineT - 1) * 100;
  if (Math.abs(delta) < .05) return null;
  return `${delta > 0 ? "+" : ""}${delta.toFixed(1)}%`;
};
const changeTone = (valueT: number, baselineT: number) => {
  const delta = (valueT / baselineT - 1) * 100;
  if (delta > .05) return "positive";
  if (delta < -.05) return "negative";
  return "neutral";
};

function adaptiveMaximum(peak: number) {
  if (peak <= BASE_MAX_T) return BASE_MAX_T;
  return Math.max(BASE_MAX_T + .5, peak * 1.08);
}

function axisTicks(maximum: number) {
  if (maximum === BASE_MAX_T) return [1, 2, 3, 4, 5, 6, 7];
  const span = maximum - BASE_MIN_T;
  const ticks = [BASE_MIN_T, BASE_MIN_T + span * .25, BASE_MIN_T + span * .5, BASE_MIN_T + span * .75, maximum];
  if (!ticks.some((tick) => Math.abs(tick - BASE_MAX_T) < .01)) ticks.push(BASE_MAX_T);
  return [...new Set(ticks)].sort((a, b) => a - b);
}

function forecast(model: ScatterModel, weights: Weights) {
  if (model.lockedAnchor) return model.parameterT;
  const evidenceIds = data.factors.filter((factor) => factor.id !== "crowd").map((factor) => factor.id);
  const availableEvidence = evidenceIds.filter((id) => model.factors[id] != null && weights[id] > 0);
  const requestedEvidenceWeight = evidenceIds.reduce((sum, id) => sum + Math.max(0, weights[id]), 0);
  const availableEvidenceWeight = availableEvidence.reduce((sum, id) => sum + weights[id], 0);
  const effective = availableEvidence.map((id) => ({
    id,
    weight: availableEvidenceWeight ? requestedEvidenceWeight * weights[id] / availableEvidenceWeight : 0,
  }));
  if (model.factors.crowd != null && weights.crowd > 0) effective.push({ id: "crowd" as FactorId, weight: weights.crowd });
  const total = effective.reduce((sum, item) => sum + item.weight, 0);
  if (!total) return model.currentFinalT;
  return Math.exp(
    effective.reduce((sum, item) => sum + (item.weight / total) * Math.log(model.factors[item.id] as number), 0),
  );
}

function effectiveWeights(model: ScatterModel, weights: Weights): Weights {
  const result = Object.fromEntries(data.factors.map((factor) => [factor.id, 0])) as Weights;
  const evidenceIds = data.factors.filter((factor) => factor.id !== "crowd").map((factor) => factor.id);
  const availableEvidence = evidenceIds.filter((id) => model.factors[id] != null && weights[id] > 0);
  const requestedEvidenceWeight = evidenceIds.reduce((sum, id) => sum + Math.max(0, weights[id]), 0);
  const availableEvidenceWeight = availableEvidence.reduce((sum, id) => sum + weights[id], 0);
  for (const id of availableEvidence) {
    result[id] = availableEvidenceWeight ? requestedEvidenceWeight * weights[id] / availableEvidenceWeight : 0;
  }
  if (model.factors.crowd != null && weights.crowd > 0) result.crowd = weights.crowd;
  const total = Object.values(result).reduce((sum, value) => sum + value, 0);
  for (const factor of data.factors) result[factor.id] = total ? result[factor.id] / total : 0;
  return result;
}

function weightedQuantile(points: { value: number; weight: number }[], quantile: number) {
  const sorted = [...points].sort((a, b) => a.value - b.value);
  const total = sorted.reduce((sum, point) => sum + point.weight, 0);
  if (!sorted.length || !total) return null;
  const positioned = sorted.map((point, index) => {
    const cumulative = sorted.slice(0, index + 1).reduce((sum, item) => sum + item.weight, 0);
    return { ...point, position: (cumulative - point.weight / 2) / total };
  });
  if (quantile <= positioned[0].position) return positioned[0].value;
  if (quantile >= positioned[positioned.length - 1].position) return positioned[positioned.length - 1].value;
  const upperIndex = positioned.findIndex((point) => point.position >= quantile);
  const lower = positioned[upperIndex - 1];
  const upper = positioned[upperIndex];
  const interpolation = (quantile - lower.position) / (upper.position - lower.position);
  return Math.exp(Math.log(lower.value) + interpolation * (Math.log(upper.value) - Math.log(lower.value)));
}

function factorRange(model: ScatterModel, weights: Weights) {
  const effective = effectiveWeights(model, weights);
  const points = data.factors
    .filter((factor) => effective[factor.id] > 0 && model.factors[factor.id] != null)
    .map((factor) => ({ value: model.factors[factor.id] as number, weight: effective[factor.id] }));
  const low = weightedQuantile(points, .1);
  const high = weightedQuantile(points, .9);
  return low != null && high != null ? [low, high] as const : [model.currentFinalT, model.currentFinalT] as const;
}

function ContributionList({ model, weights, compact = false }: { model: ScatterModel; weights: Weights; compact?: boolean }) {
  const effective = effectiveWeights(model, weights);
  const rows = data.factors
    .map((factor) => ({ factor, value: model.factors[factor.id], weight: effective[factor.id] }))
    .filter((row) => row.value != null && row.weight > 0);
  return (
    <div className={compact ? "contribution-list compact" : "contribution-list"}>
      <div className="contribution-stack" aria-label="Effective evidence contribution weights">
        {rows.map(({ factor, weight }) => (
          <span key={factor.id} title={`${factor.label}: ${(weight * 100).toFixed(1)}%`} style={{ width: `${weight * 100}%`, background: FACTOR_COLORS[factor.id] }} />
        ))}
      </div>
      <div className="contribution-rows">
        {rows.map(({ factor, weight }) => (
          <div className="contribution-row" key={factor.id}>
            <span><i style={{ background: FACTOR_COLORS[factor.id] }} />{factor.shortLabel}</span>
            <b>{(weight * 100).toFixed(1)}%</b>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Home() {
  const [multipliers, setMultipliers] = useState<Weights>({ ...ONE_X });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [activeOrganization, setActiveOrganization] = useState<string | null>(null);
  const [openHelp, setOpenHelp] = useState<FactorId | null>(null);
  const [railOpen, setRailOpen] = useState(false);
  const [themeMode, setThemeMode] = useState<"system" | "light" | "dark">("system");
  const [systemDark, setSystemDark] = useState(false);

  useEffect(() => {
    const close = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setSelectedId(null);
      setRailOpen(false);
      setOpenHelp(null);
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, []);

  useEffect(() => {
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setSystemDark(query.matches);
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    if (themeMode === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = themeMode;
  }, [themeMode]);

  const weights = useMemo(
    () => Object.fromEntries(data.factors.map((factor) => [factor.id, data.defaultWeights[factor.id] * multipliers[factor.id]])) as Weights,
    [multipliers],
  );

  const results = useMemo(
    () => data.models.map((model) => ({ model, value: forecast(model, weights) })),
    [weights],
  );
  const stableRows = [...results].sort((a, b) => b.model.currentFinalT - a.model.currentFinalT);
  const peak = Math.max(...results.flatMap(({ model, value }) => [value, model.publicEstimateT ?? value]));
  const maximum = adaptiveMaximum(peak);
  const ticks = axisTicks(maximum);
  const pastBaseline = maximum > BASE_MAX_T;
  const selected = selectedId ? data.models.find((model) => model.id === selectedId) ?? null : null;
  const selectedFactorRange = selected && !selected.lockedAnchor ? factorRange(selected, weights) : null;
  const focused = data.models.find((model) => model.id === (hoveredId ?? selectedId)) ?? null;
  const focusedValue = focused ? forecast(focused, weights) : null;
  const colors = Object.fromEntries(data.organizationGroups.map((organization) => [organization.id, organization.color]));
  const axisSpan = maximum - BASE_MIN_T;
  const position = (value: number) => `${clamp((value - BASE_MIN_T) / axisSpan, 0, 1) * 100}%`;
  const width = (low: number, high: number) => `${Math.max(0, clamp(high, BASE_MIN_T, maximum) - clamp(low, BASE_MIN_T, maximum)) / axisSpan * 100}%`;
  const focusId = hoveredId ?? selectedId;
  const darkMode = themeMode === "dark" || (themeMode === "system" && systemDark);

  const setMultiplier = (factor: FactorId, value: number) => setMultipliers((current) => ({ ...current, [factor]: value }));

  return (
    <main className="site-shell">
      <section className={railOpen ? "chart-layout" : "chart-layout rail-collapsed"} id="top">
        <div className="plot-panel">
          <div className="plot-header">
            <div>
              <h1>{data.title}</h1>
              <p className="eyebrow"><a href="https://twitter.com/anpaure" target="_blank" rel="noreferrer">made by @anpaure</a></p>
            </div>
            <div className="header-actions">
              <button className="rail-toggle theme-toggle" onClick={() => setThemeMode(darkMode ? "light" : "dark")} aria-label={darkMode ? "Use light mode" : "Use dark mode"} title={darkMode ? "Use light mode" : "Use dark mode"} aria-pressed={darkMode}><span aria-hidden="true">{darkMode ? "☀" : "☾"}</span></button>
              <button className={`rail-toggle floating${railOpen ? " active" : ""}`} onClick={() => setRailOpen((open) => !open)} aria-label={railOpen ? "Hide controls" : "Show controls"} title={railOpen ? "Hide controls" : "Show controls"} aria-expanded={railOpen} aria-controls="evidence-controls"><i /></button>
            </div>
          </div>

          <div className="forest-chart" data-model-count={data.counts.models} data-axis-min={BASE_MIN_T} data-axis-max={maximum} data-past-baseline={pastBaseline}>
            <div className="forest-axis">
              <span />
              <div>
                <strong>Implied total parameters</strong>
                {ticks.map((tick, index) => {
                  const internal = index > 0 && index < ticks.length - 1;
                  return <i className={`${pastBaseline && tick === BASE_MAX_T ? "baseline" : ""}${internal ? " mobile-internal" : ""}${internal && index % 2 === 1 ? " mobile-optional" : ""}${index === Math.floor((ticks.length - 1) / 2) ? " mobile-midpoint" : ""}`} key={tick} style={{ left: position(tick) }}>{formatParameter(tick)}</i>;
                })}
              </div>
              <span />
            </div>

            <div className="forest-rows">
              {stableRows.map(({ model, value }, index) => {
                const [low, high] = factorRange(model, weights);
                const change = formatChange(value, model.currentFinalT);
                const color = colors[model.organizationGroup] ?? "#A75237";
                const highlighted = focusId === model.id;
                const dimmed = (focusId != null && !highlighted) || (activeOrganization != null && model.organizationGroup !== activeOrganization);
                return (
                  <button
                    className={`forest-row${index % 2 ? " striped" : ""}${highlighted ? " highlighted" : ""}${dimmed ? " dimmed" : ""}`}
                    key={model.id}
                    onFocus={() => setHoveredId(model.id)}
                    onBlur={() => setHoveredId(null)}
                    onClick={() => setSelectedId((current) => current === model.id ? null : model.id)}
                    aria-pressed={selectedId === model.id}
                    aria-label={`Inspect ${model.name}, model estimate ${formatParameter(value)}${model.publicEstimateT != null ? `, public estimate ${formatParameter(model.publicEstimateT)}` : ""}`}
                  >
                    <span className="forest-label"><strong>{model.shortName}</strong></span>
                    <span className="forest-track">
                      {ticks.map((tick) => <i className={`row-grid${pastBaseline && tick === BASE_MAX_T ? " baseline" : ""}`} key={tick} style={{ left: position(tick) }} />)}
                      {!model.lockedAnchor && <span className="factor-whisker" style={{ left: position(low), width: width(low, high) }} title={`Weighted factor range ${formatFactorRange([low, high])}`}><i /><i /></span>}
                      {model.publicEstimateT != null && <span className="public-marker" style={{ left: position(model.publicEstimateT) }} title={`Public estimate ${formatParameter(model.publicEstimateT)}`} onMouseEnter={() => setHoveredId(model.id)} onMouseLeave={() => setHoveredId(null)} />}
                      <span className="scenario-marker" style={{ left: position(value), background: color }} title={`Model estimate ${formatParameter(value)}`} onMouseEnter={() => setHoveredId(model.id)} onMouseLeave={() => setHoveredId(null)} />
                    </span>
                    <span className="forest-value"><strong>{formatParameter(value)}</strong>{model.lockedAnchor ? <small>disclosed</small> : change && <small className={changeTone(value, model.currentFinalT)}>{change}</small>}</span>
                  </button>
                );
              })}
            </div>

            <div className="forest-legend">
              <span><i className="legend-circle" />Public estimate</span>
              <span><i className="legend-diamond" />Model estimate</span>
              <span><i className="legend-line" />Weighted factor range</span>
            </div>

            {selected && (
              <article className="evidence-card" role="dialog" aria-modal="false" aria-label={`${selected.name} evidence contribution`}>
                <div className="card-header"><div><p>{selected.organization}</p><h2>{selected.name}</h2></div><button onClick={() => setSelectedId(null)} aria-label="Close model details">×</button></div>
                <dl className="summary-grid">
                  <div><dt>Model estimate</dt><dd>{formatParameter(forecast(selected, weights))} <small>{selected.parameterKind}</small></dd></div>
                  {selected.publicEstimateT != null && <div><dt>Public estimate</dt><dd>{formatParameter(selected.publicEstimateT)}</dd></div>}
                  {selectedFactorRange && <div><dt>Weighted factor range</dt><dd>{formatFactorRange(selectedFactorRange)}</dd></div>}
                  <div><dt>Release date</dt><dd>{formatDate(selected.releaseDate)}</dd></div>
                  <div><dt>Architecture</dt><dd>MoE</dd></div>
                  {selected.signals.activeParametersB != null && <div><dt>Active parameters</dt><dd>{formatParameter(selected.signals.activeParametersB / 1000)}</dd></div>}
                </dl>
                {!selected.lockedAnchor && <div className="card-section"><div className="section-title"><h3>Evidence weights</h3></div><ContributionList model={selected} weights={weights} /></div>}
              </article>
            )}
          </div>
        </div>

        {railOpen && (
          <aside className="control-rail" id="evidence-controls" aria-label="Chart controls">
            <div className="rail-head"><button className="rail-close" onClick={() => setRailOpen(false)} aria-label="Close controls">×</button></div>
            <section className="rail-section">
              <h2>Developer</h2>
              <div className="organization-legend">
                {data.organizationGroups.map((organization) => {
                  const count = data.models.filter((model) => model.organizationGroup === organization.id).length;
                  return (
                    <button className={activeOrganization && activeOrganization !== organization.id ? "inactive" : ""} key={organization.id} onClick={() => setActiveOrganization((current) => current === organization.id ? null : organization.id)}>
                      <i style={{ background: organization.color }} /><span>{organization.id}</span><small>{count}</small>
                    </button>
                  );
                })}
              </div>
            </section>
            <section className="rail-section weights-section">
              <div className="weights-heading"><h2>Evidence weights</h2><button onClick={() => setMultipliers({ ...ONE_X })}>Reset</button></div>
              <div className="weight-sliders">
                {data.factors.filter((factor) => data.defaultWeights[factor.id] > 0).map((factor) => {
                  const exponent = Math.log10(multipliers[factor.id]);
                  const progress = (exponent + 1) * 50;
                  return (
                    <div className="weight-control" key={factor.id}>
                      <div className="weight-label"><b>{factor.shortLabel}</b><button type="button" aria-label={`Explain ${factor.label}`} aria-expanded={openHelp === factor.id} aria-controls={`factor-help-${factor.id}`} onClick={() => setOpenHelp((current) => current === factor.id ? null : factor.id)}>?</button><em>{formatMultiplier(multipliers[factor.id])}</em></div>
                      {openHelp === factor.id && <p className="factor-explanation" id={`factor-help-${factor.id}`}>{factor.description}</p>}
                      <input aria-label={`${factor.label} weight multiplier`} aria-valuetext={formatMultiplier(multipliers[factor.id])} type="range" min="-1" max="1" step="0.01" value={exponent} style={{ "--factor-color": FACTOR_COLORS[factor.id], "--factor-progress": `${progress}%` } as CSSProperties} onChange={(event) => setMultiplier(factor.id, Math.pow(10, Number(event.target.value)))} />
                    </div>
                  );
                })}
              </div>
            </section>
            {focused && focusedValue != null && <section className="rail-section rail-preview" aria-live="polite"><p>Highlighted model</p><h2>{focused.shortName}</h2><strong>{formatParameter(focusedValue)}</strong>{!focused.lockedAnchor && <ContributionList model={focused} weights={weights} compact />}</section>}
          </aside>
        )}
      </section>
      <section className="methodology-note" aria-labelledby="methodology-title">
        <div className="methodology-meta">
          <h2 id="methodology-title">Methodology</h2>
          <a className="repo-link" href="https://github.com/anpaure/frontier-model-sizes" target="_blank" rel="noreferrer">
            <svg className="repo-mark" viewBox="0 0 32 32" aria-hidden="true"><path fill="currentColor" fillRule="evenodd" clipRule="evenodd" d="M16 0C7.16 0 0 7.16 0 16c0 7.08 4.58 13.06 10.94 15.18.8.14 1.1-.34 1.1-.76 0-.38-.02-1.64-.02-2.98-4.02.74-5.06-.98-5.38-1.88-.18-.46-.96-1.88-1.64-2.26-.56-.3-1.36-1.04-.02-1.06 1.26-.02 2.16 1.16 2.46 1.64 1.44 2.42 3.74 1.74 4.66 1.32.14-1.04.56-1.74 1.02-2.14-3.56-.4-7.28-1.78-7.28-7.9 0-1.74.62-3.18 1.64-4.3-.16-.4-.72-2.04.16-4.24 0 0 1.34-.42 4.4 1.64 1.28-.36 2.64-.54 4-.54s2.72.18 4 .54c3.06-2.08 4.4-1.64 4.4-1.64.88 2.2.32 3.84.16 4.24 1.02 1.12 1.64 2.54 1.64 4.3 0 6.14-3.74 7.5-7.3 7.9.58.5 1.08 1.46 1.08 2.96 0 2.14-.02 3.86-.02 4.4 0 .42.3.92 1.1.76C27.42 29.06 32 23.06 32 16 32 7.16 24.84 0 16 0Z" /></svg>
            Check the repo
          </a>
        </div>
        <p>The Artificial Analysis Intelligence Index and Epoch Capabilities Index provide audited capability signals, while the No-CoT Time Horizon supplies the primary parameter-scaling branch; API price and compute are lower-weight checks. The goal is to make otherwise opaque frontier-model parameter counts inspectable, adjustable, and reproducible instead of presenting one unexplained guess.</p>
      </section>
    </main>
  );
}
