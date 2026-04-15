"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import useSWR from "swr";
import toast from "react-hot-toast";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  BarChart,
  Bar,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
} from "recharts";
import { useAuth } from "../context/AuthContext";

const RiskMap = dynamic(() => import("./risk-map"), { ssr: false });

const tabs = ["Risk Map", "Remedies & Precautions", "Forecast", "AI Explainability", "Resource Tracker"];

const severityColor = {
  critical: "#ff536f",
  high: "#ffb24c",
  medium: "#46a2ff",
  low: "#2ed39a",
};

const chartGrid = "rgba(146,165,199,0.14)";
const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const fetcher = async (url) => {
  const response = await fetch(`${API_BASE_URL}${url}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Request failed for ${url}`);
  }
  return response.json();
};

function getRiskCommandSubtitle(detail) {
  if (!detail) return "District command view";
  if (detail.riskLevel === "critical") {
    return `${detail.district} requires immediate field action`;
  }
  if (detail.riskLevel === "high") {
    return `${detail.district} is flagged for high-priority intervention`;
  }
  if (detail.riskLevel === "medium") {
    return `${detail.district} is under enhanced surveillance review`;
  }
  return `${detail.district} is currently in monitoring mode`;
}

function buildFallbackExplainability(detail) {
  if (!detail) {
    return {
      primaryDriver: "No local driver available",
      explanationText: "No district explainability context is available.",
      features: [],
    };
  }

  const features = [
    { feature: "rainfall anomaly", value: Math.min(1, Math.abs((detail.rainfallAnomalyPct ?? 0) / 100)) },
    { feature: "sanitation stress", value: Math.min(1, Math.max(0, (100 - (detail.sanitationCoveragePct ?? 0)) / 100)) },
    { feature: "recent case burden", value: Math.min(1, (detail.caseCount ?? 0) / 15) },
    { feature: "wash fragility", value: Math.min(1, Math.max(0, 1 - (detail.washIndex ?? 0))) },
    { feature: "district risk score", value: Math.min(1, detail.riskScore ?? 0) },
  ]
    .sort((a, b) => b.value - a.value)
    .map((item) => ({ ...item, value: Number(item.value.toFixed(2)) }));

  const primary = features[0]?.feature ?? "district risk score";
  return {
    primaryDriver: primary,
    explanationText: `${detail.district} is showing ${detail.riskLevel.toUpperCase()} ${detail.topDisease} risk. The signal is currently driven most by ${primary}, alongside rainfall anomaly, sanitation conditions, and local case pressure.`,
    features,
  };
}

function cls(...parts) {
  return parts.filter(Boolean).join(" ");
}

function severityBadge(level) {
  return (
    <span
      className="rounded-full border px-3 py-1 text-[11px] font-extrabold uppercase tracking-[0.22em]"
      style={{
        borderColor: `${severityColor[level]}55`,
        color: severityColor[level],
        backgroundColor: `${severityColor[level]}18`,
      }}
    >
      {level}
    </span>
  );
}

function Panel({ title, subtitle, children, className = "" }) {
  return (
    <div className={cls("glass-panel rounded-[1.3rem] p-5", className)}>
      {(title || subtitle) && (
        <div className="mb-4">
          {title && <div className="text-[11px] font-extrabold uppercase tracking-[0.22em] text-slate-400">{title}</div>}
          {subtitle && <div className="mt-2 text-xl font-extrabold text-white">{subtitle}</div>}
        </div>
      )}
      {children}
    </div>
  );
}

function MetricCard({ label, value, note, glowClass }) {
  return (
    <motion.div whileHover={{ y: -6 }} className={cls("glass-panel rounded-[1.2rem] p-5", glowClass)}>
      <div className="text-[11px] font-extrabold uppercase tracking-[0.22em] text-slate-400">{label}</div>
      <div className="mt-3 text-4xl font-black text-white">{value}</div>
      <div className="mt-3 inline-flex rounded-full bg-white/5 px-3 py-1 text-sm font-semibold text-slate-300">{note}</div>
    </motion.div>
  );
}

function ChartFrame({ title, children }) {
  return (
    <Panel className="h-[24rem] p-4">
      <div className="mb-4 text-lg font-extrabold text-white">{title}</div>
      <div className="h-[19rem]">{children}</div>
    </Panel>
  );
}

function modelLabel(model) {
  if (model.includes("HydroCast")) return "HydroCast";
  if (model === "Random Forest") return "RF";
  if (model === "Logistic Regression") return "LogReg";
  return model;
}

export default function Dashboard({ initialTab = "Risk Map" }) {
  const [activeTab, setActiveTab] = useState("Risk Map");
  const [selectedDistrict, setSelectedDistrict] = useState("Raigad");
  const [selectedDisease, setSelectedDisease] = useState("Cholera");
  const [mapMode, setMapMode] = useState("both");
  const [clock, setClock] = useState("");
  const { user, logout } = useAuth();

  const summaryQuery = useSWR("/api/dashboard/summary", fetcher, { revalidateOnFocus: false });
  const forecastQuery = useSWR("/api/dashboard/forecast", fetcher, { revalidateOnFocus: false });
  const riskMapQuery = useSWR("/api/dashboard/risk-map", fetcher, { revalidateOnFocus: false });
  const shapQuery = useSWR("/api/dashboard/shap", fetcher, { revalidateOnFocus: false });
  const resourcesQuery = useSWR("/api/dashboard/resources", fetcher, { revalidateOnFocus: false });

  useEffect(() => {
    setActiveTab(initialTab);
  }, [initialTab]);

  useEffect(() => {
    const defaultDistrict = summaryQuery.data?.districtRankings?.[0]?.district;
    if (defaultDistrict && !selectedDistrict) {
      setSelectedDistrict(defaultDistrict);
    }
  }, [selectedDistrict, summaryQuery.data?.districtRankings]);

  useEffect(() => {
    const errors = [
      summaryQuery.error,
      forecastQuery.error,
      riskMapQuery.error,
      shapQuery.error,
      resourcesQuery.error,
    ].filter(Boolean);
    if (errors.length) {
      toast.error("Failed to load live dashboard data. Please retry.");
    }
  }, [forecastQuery.error, resourcesQuery.error, riskMapQuery.error, shapQuery.error, summaryQuery.error]);

  useEffect(() => {
    const tick = () =>
      setClock(
        new Intl.DateTimeFormat("en-IN", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
        }).format(new Date()),
      );
    tick();
    const timer = setInterval(tick, 1000);
    return () => clearInterval(timer);
  }, []);

  const overview = summaryQuery.data?.overview;
  const rankedDistricts = summaryQuery.data?.districtRankings ?? [];
  const districtDetails = summaryQuery.data?.districtDetails ?? {};
  const alerts = summaryQuery.data?.alerts ?? [];
  const baselines = summaryQuery.data?.baselines ?? [];
  const geojson = riskMapQuery.data?.geojson;
  const selectedDetail = districtDetails?.[selectedDistrict];
  const selectedForecasts = forecastQuery.data?.forecasts?.[selectedDistrict];
  const selectedRemedies = resourcesQuery.data?.remedies?.[selectedDistrict] ?? {};
  const selectedRemedy = selectedRemedies?.[selectedDisease];
  const selectedResources = resourcesQuery.data?.resources?.[selectedDistrict];
  const selectedDiseaseForecast = selectedForecasts?.[selectedDisease];

  useEffect(() => {
    if (selectedDetail?.topDisease) {
      setSelectedDisease(selectedDetail.topDisease);
    }
  }, [selectedDistrict, selectedDetail?.topDisease]);

  const mapDistricts = useMemo(
    () =>
      rankedDistricts.map((item) => ({
        district: item.district,
        topDisease: item.top_disease,
        riskLevel: item.risk_level,
        riskScore: item.risk_score,
        rainfallAnomalyPct: item.rainfall_anomaly_pct,
        sanitationCoveragePct: item.sanitation_coverage_pct,
        caseCount: item.case_count,
        latitude: districtDetails?.[item.district]?.latitude ?? 0,
        longitude: districtDetails?.[item.district]?.longitude ?? 0,
      })),
    [districtDetails, rankedDistricts],
  );

  const topDiseaseForecast = useMemo(() => {
    if (!selectedDisease || !selectedForecasts) return [];
    const disease = selectedDisease;
    const history = (selectedForecasts[disease]?.history ?? []).map((row) => ({
      label: row.week,
      observed: row[`${disease.toLowerCase()}_cases`] ?? Object.values(row).at(-1),
    }));
    const projected = (selectedForecasts[disease]?.projectedCases ?? []).map((value, index) => ({
      label: `W+${index + 1}`,
      forecast: value,
    }));
    return [...history, ...projected];
  }, [selectedDisease, selectedForecasts]);

  const districtComparison = useMemo(
    () => rankedDistricts.slice(0, 10).map((item) => ({ district: item.district, risk: Math.round(item.risk_score * 100) })),
    [rankedDistricts],
  );

  const shapLocal = useMemo(() => {
    if (!selectedDistrict || !selectedDisease || !shapQuery.data?.shapValues?.[selectedDistrict]) return [];
    const disease = selectedDisease;
    return (shapQuery.data?.shapValues?.[selectedDistrict]?.[disease]?.top_features ?? []).map(([feature, value]) => ({
      feature: feature.replaceAll("_", " "),
      value,
    }));
  }, [selectedDisease, selectedDistrict, shapQuery.data?.shapValues]);

  const selectedShapEntry = useMemo(() => {
    if (!selectedDistrict || !selectedDisease) return null;
    return shapQuery.data?.shapValues?.[selectedDistrict]?.[selectedDisease] ?? null;
  }, [selectedDisease, selectedDistrict, shapQuery.data?.shapValues]);

  const fallbackExplainability = useMemo(() => buildFallbackExplainability(selectedDetail), [selectedDetail]);
  const miniBarFeatures = shapLocal.length ? shapLocal : fallbackExplainability.features;
  const resourceRadar = useMemo(
    () => Object.entries(selectedResources ?? {}).map(([resource, readiness]) => ({ resource, readiness })),
    [selectedResources],
  );

  const isLoading =
    !overview ||
    !selectedDetail ||
    !selectedForecasts ||
    !selectedDiseaseForecast ||
    !selectedRemedy ||
    !selectedResources ||
    !geojson;

  if (isLoading) {
    return (
      <main className="min-h-screen bg-command p-8 text-white">
        <div className="mx-auto max-w-[1600px] space-y-4">
          <div className="glass-panel rounded-[1.5rem] p-6">
            <div className="h-8 w-44 animate-pulse rounded-full bg-white/10" />
            <div className="mt-5 h-16 w-96 max-w-full animate-pulse rounded-2xl bg-white/10" />
            <div className="mt-6 grid gap-4 lg:grid-cols-3">
              {[1, 2, 3].map((item) => (
                <div key={item} className="h-40 animate-pulse rounded-[1.4rem] bg-white/5" />
              ))}
            </div>
          </div>
          <div className="grid gap-4 lg:grid-cols-5">{[1, 2, 3, 4, 5].map((item) => <div key={item} className="glass-panel h-48 animate-pulse rounded-[1.4rem] bg-white/[0.03]" />)}</div>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-command px-4 py-5 text-white lg:px-6">
      <div className="mx-auto max-w-[1600px]">
        <div className="space-y-4">
          <div className="glass-panel sticky top-4 z-30 rounded-[1.5rem] p-4">
            <div className="grid gap-4 xl:grid-cols-[1.45fr_1fr_0.82fr]">
              <Panel className="overflow-hidden">
                <div className="text-[11px] font-extrabold uppercase tracking-[0.26em] text-slate-400">Maharashtra Waterborne Surveillance Grid</div>
                <div className="orbitron mt-3 text-4xl font-bold">HydroCast</div>
                <div className="mt-2 text-lg font-bold text-slate-100">AI Early Warning System</div>
                <div className="mt-3 max-w-xl text-sm leading-7 text-slate-300">
                  Real-time decision dashboard for outbreak prediction, remedy prioritization, explainable AI oversight, and district-level
                  resource command.
                </div>
                <div className="mt-5 inline-flex items-center gap-3 rounded-full border border-critical/30 bg-critical/10 px-4 py-2 font-bold text-critical">
                  <span className="h-3 w-3 animate-pulse rounded-full bg-critical" />
                  Live monitoring active
                </div>
                <div className="mt-5 flex items-center justify-between gap-3 rounded-[1rem] border border-white/10 bg-white/[0.04] px-4 py-3">
                  <div>
                    <div className="text-[11px] font-extrabold uppercase tracking-[0.22em] text-slate-500">Active user</div>
                    <div className="mt-1 font-bold text-white">{user?.name ?? "Operator"}</div>
                    <div className="text-sm text-slate-400">{user?.email ?? "session@hydrocast.ai"}</div>
                  </div>
                  <button
                    onClick={logout}
                    className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm font-bold text-white transition hover:border-cyan-400/40 hover:text-cyan-300"
                  >
                    Logout
                  </button>
                </div>
              </Panel>

              <div className="grid gap-4 sm:grid-cols-2">
                {[
                  ["Active outbreaks", overview.activeOutbreaks, "districts under active watch"],
                  ["High-risk districts", overview.highRiskDistricts, "critical + high severity queue"],
                  ["Model F1 score", overview.modelF1.toFixed(3), "best validation benchmark"],
                  ["Actions issued", overview.actionsIssued, "automated response triggers"],
                ].map(([label, value, note]) => (
                  <Panel key={label}>
                    <div className="text-[11px] font-extrabold uppercase tracking-[0.22em] text-slate-400">{label}</div>
                    <div className="mt-3 text-3xl font-black text-white">{value}</div>
                    <div className="mt-2 text-sm text-slate-300">{note}</div>
                  </Panel>
                ))}
              </div>

              <Panel className="flex h-full flex-col justify-center">
                <div className="text-[11px] font-extrabold uppercase tracking-[0.22em] text-slate-400">Situation clock</div>
                <div className="orbitron mt-4 text-4xl font-bold">{clock}</div>
                <div className="mt-3 text-sm text-slate-300">
                  {new Intl.DateTimeFormat("en-IN", { day: "2-digit", month: "short", year: "numeric" }).format(new Date())} | Maharashtra
                  command view
                </div>
              </Panel>
            </div>

            <div className="mt-4 flex flex-wrap gap-3">
              {tabs.map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={cls(
                    "rounded-full border px-4 py-2 text-sm font-bold transition-all",
                    activeTab === tab
                      ? "border-info/60 bg-info/15 text-white shadow-[0_0_24px_rgba(70,162,255,0.18)]"
                      : "border-white/10 bg-white/5 text-slate-300 hover:border-info/40 hover:text-white",
                  )}
                >
                  {tab}
                  {tab === "Remedies & Precautions" && (
                    <span className="ml-3 rounded-full bg-critical px-2 py-0.5 text-[11px] font-extrabold text-white">4 urgent</span>
                  )}
                </button>
              ))}
            </div>
          </div>

          <div className="grid gap-4 lg:grid-cols-5">
            <MetricCard label="Critical Signals" value={overview.criticalDistricts} note="escalate immediate response" glowClass="metric-glow-critical" />
            <MetricCard label="Districts On Watch" value={overview.highRiskDistricts} note="field verification needed" glowClass="metric-glow-high" />
            <MetricCard label="Average Risk" value={`${Math.round(overview.averageRisk * 100)}%`} note="statewide surveillance burden" glowClass="metric-glow-info" />
            <MetricCard label="Observed Cases" value={overview.observedCases} note="latest capture from monitoring feed" glowClass="metric-glow-safe" />
            <MetricCard label="System Readiness" value="94%" note="models, alerts, and resources aligned" glowClass="metric-glow-ai" />
          </div>
        </div>

        <div className="mt-5 grid gap-4 xl:grid-cols-[1.02fr_2.65fr_1.18fr]">
          <aside className="space-y-4">
            <Panel title="District Priority Queue" subtitle="Risk-ranked Maharashtra districts">
              <div className="scroll-skin max-h-[72rem] space-y-3 overflow-auto pr-1">
                {rankedDistricts.slice(0, 12).map((district) => (
                  <motion.button
                    key={district.district}
                    whileHover={{ y: -4 }}
                    onClick={() => {
                      setSelectedDistrict(district.district);
                      setActiveTab("Risk Map");
                    }}
                    className={cls(
                      "w-full rounded-[1.1rem] border p-4 text-left transition-all",
                      selectedDistrict === district.district
                        ? "border-info/60 bg-info/10 shadow-glow"
                        : "border-white/8 bg-white/[0.03] hover:border-info/40",
                    )}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="text-lg font-extrabold text-white">{district.district}</div>
                        <div className="mt-1 text-sm text-slate-300">{district.top_disease} lead signal</div>
                      </div>
                      {severityBadge(district.risk_level)}
                    </div>
                    <div className="mt-4 flex items-end justify-between">
                      <div>
                        <div className="text-[11px] font-extrabold uppercase tracking-[0.22em] text-slate-500">Risk score</div>
                        <div className="mt-1 text-2xl font-black text-white">{Math.round(district.risk_score * 100)}%</div>
                      </div>
                      <div className="text-right">
                        <div className="text-[11px] font-extrabold uppercase tracking-[0.22em] text-slate-500">Cases</div>
                        <div className="mt-1 text-xl font-black text-white">{district.case_count}</div>
                      </div>
                    </div>
                  </motion.button>
                ))}
              </div>
            </Panel>
          </aside>

          <section className="space-y-4">
            <AnimatePresence mode="wait">
              {activeTab === "Risk Map" && (
                <motion.div key="risk-map" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="space-y-4">
                  <Panel title="Risk Map Command View" subtitle={getRiskCommandSubtitle(selectedDetail)}>
                    <div className="mb-4 flex flex-wrap items-center justify-between gap-4">
                      <div className="max-w-2xl text-sm leading-7 text-slate-300">
                        Dark-mode command map of Maharashtra. Hover for district surveillance context and click any district to move directly into the
                        remedy and operational workflow.
                      </div>
                      <div className="flex gap-2">
                        {[
                          ["choropleth", "Choropleth"],
                          ["bubble", "Bubble"],
                          ["both", "Both"],
                        ].map(([value, label]) => (
                          <button
                            key={value}
                            onClick={() => setMapMode(value)}
                            className={cls(
                              "rounded-full border px-4 py-2 text-sm font-bold",
                              mapMode === value ? "border-info/60 bg-info/15 text-white" : "border-white/10 bg-white/5 text-slate-300",
                            )}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="grid gap-4 xl:grid-cols-[1.8fr_0.9fr]">
                      <div className="map-shell rounded-[1.25rem] border border-white/10 bg-[#08111f] p-2">
                        <RiskMap
                          geojson={geojson}
                          districts={mapDistricts}
                          selectedDistrict={selectedDistrict}
                          mapMode={mapMode}
                          onDistrictSelect={(district) => {
                            setSelectedDistrict(district);
                            setActiveTab("Remedies & Precautions");
                          }}
                        />
                      </div>

                      <Panel title="Legend" subtitle="Always-visible severity guidance" className="h-full">
                        <div className="space-y-4">
                          {Object.entries(severityColor).map(([key, color]) => (
                            <div key={key} className="flex items-center justify-between rounded-2xl border border-white/8 bg-white/[0.03] px-4 py-3">
                              <div className="flex items-center gap-3">
                                <span className="h-3.5 w-3.5 rounded-full" style={{ backgroundColor: color }} />
                                <span className="font-semibold capitalize text-white">{key}</span>
                              </div>
                              <span className="text-sm text-slate-400">
                                {key === "critical" && ">= 80%"}
                                {key === "high" && ">= 60%"}
                                {key === "medium" && ">= 40%"}
                                {key === "low" && "< 40%"}
                              </span>
                            </div>
                          ))}
                          <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4 text-sm leading-7 text-slate-300">
                            Bubble size tracks recent case burden. Polygon fill represents projected outbreak severity from the HydroCast model.
                          </div>
                        </div>
                      </Panel>
                    </div>
                  </Panel>

                  <div className="grid gap-4 lg:grid-cols-2">
                    <ChartFrame title={`${selectedDisease} trend and forecast | ${selectedDistrict}`}>
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={topDiseaseForecast}>
                          <defs>
                            <linearGradient id="forecastFill" x1="0" x2="0" y1="0" y2="1">
                              <stop offset="0%" stopColor="#46a2ff" stopOpacity={0.34} />
                              <stop offset="100%" stopColor="#46a2ff" stopOpacity={0.02} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid stroke={chartGrid} vertical={false} />
                          <XAxis dataKey="label" stroke="#8ea2c2" />
                          <YAxis stroke="#8ea2c2" />
                          <Tooltip contentStyle={{ background: "#08111f", border: "1px solid rgba(146,165,199,0.16)", borderRadius: 16 }} />
                          <Area type="monotone" dataKey="observed" stroke="#46a2ff" fill="url(#forecastFill)" strokeWidth={3} />
                          <Line type="monotone" dataKey="forecast" stroke={severityColor[selectedDiseaseForecast.riskLevel]} strokeWidth={3} dot={{ fill: severityColor[selectedDiseaseForecast.riskLevel] }} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </ChartFrame>

                    <ChartFrame title="Model comparison benchmark">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={baselines} margin={{ top: 8, right: 8, left: -18, bottom: 48 }}>
                          <CartesianGrid stroke={chartGrid} horizontal vertical={false} />
                          <XAxis
                            dataKey="Model"
                            interval={0}
                            angle={-20}
                            height={64}
                            textAnchor="end"
                            stroke="#8ea2c2"
                            tickFormatter={modelLabel}
                            tick={{ fontSize: 11 }}
                          />
                          <YAxis stroke="#8ea2c2" />
                          <Tooltip contentStyle={{ background: "#08111f", border: "1px solid rgba(146,165,199,0.16)", borderRadius: 16 }} />
                          <Bar dataKey="F1_Macro" radius={[12, 12, 0, 0]} minPointSize={6}>
                            {baselines.map((entry) => (
                              <Cell key={entry.Model} fill={entry.Model.includes("HydroCast") ? "#2ed39a" : "#365a86"} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </ChartFrame>
                  </div>
                </motion.div>
              )}

              {activeTab === "Remedies & Precautions" && (
                <motion.div key="remedies" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="space-y-4">
                  <Panel title="AI recommendation card" subtitle={`${selectedDistrict} | ${selectedRemedy.disease}`}>
                    <div className="flex flex-wrap items-start justify-between gap-4">
                      <div className="max-w-3xl">
                        <div className="mb-3">{severityBadge(selectedRemedy.risk_level)}</div>
                        <div className="text-2xl font-black text-white">Risk score {Math.round(selectedRemedy.risk_score * 100)}%</div>
                        <div className="mt-4 text-base leading-8 text-slate-200">{selectedRemedy.ai_recommendation}</div>
                      </div>
                      <div className="flex flex-col gap-3 sm:min-w-[240px]">
                        <select
                          className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 font-semibold text-white outline-none"
                          value={selectedDistrict}
                          onChange={(event) => setSelectedDistrict(event.target.value)}
                        >
                          {rankedDistricts.map((item) => (
                            <option key={item.district} value={item.district} className="bg-slate-900">
                              {item.district}
                            </option>
                          ))}
                        </select>
                        <select
                          className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 font-semibold text-white outline-none"
                          value={selectedDisease}
                          onChange={(event) => setSelectedDisease(event.target.value)}
                        >
                          {Object.keys(selectedForecasts ?? {}).map((disease) => (
                            <option key={disease} value={disease} className="bg-slate-900">
                              {disease}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </Panel>

                  <div className="grid gap-4 xl:grid-cols-4">
                    {selectedRemedy.timeline.map((step, index) => (
                      <Panel key={step.phase} className={cls(index === 0 && "border-ai/50 shadow-[0_0_30px_rgba(171,140,255,0.16)]")}>
                        <div className="text-[11px] font-extrabold uppercase tracking-[0.22em] text-slate-400">{step.phase}</div>
                        <div className="mt-3 text-lg font-extrabold text-white">{step.title}</div>
                        <div className="mt-3 text-sm leading-7 text-slate-300">{step.actions}</div>
                      </Panel>
                    ))}
                  </div>

                  <div className="grid gap-4 lg:grid-cols-2 2xl:grid-cols-3">
                    {[
                      ["Emergency", selectedRemedy.emergency_actions, "text", "priority"],
                      ["Medical", selectedRemedy.medical_protocols, "notes", "drug"],
                      ["WASH", selectedRemedy.wash_actions, "action", "priority"],
                      ["Personal", selectedRemedy.precautions, "advisory", "priority"],
                      ["Community", selectedRemedy.community_actions, "action", "priority"],
                      ["Government", selectedRemedy.government_protocol, "action", "authority"],
                    ].map(([title, items, textKey, badgeKey]) => (
                      <Panel key={title} title={title} subtitle={`${title} protocol`}>
                        <div className="space-y-3">
                          {items.slice(0, 4).map((item, index) => {
                            const badgeValue = item[badgeKey] ?? "INFO";
                            const level = String(badgeValue).toUpperCase().includes("CRITICAL")
                              ? "critical"
                              : String(badgeValue).toUpperCase().includes("HIGH")
                                ? "high"
                                : "medium";
                            return (
                              <label key={`${title}-${index}`} className="flex gap-3 rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                                <input type="checkbox" className="mt-1 h-4 w-4 rounded border-white/20 bg-transparent accent-[#46a2ff]" />
                                <div className="space-y-3">
                                  <div>{severityBadge(level)}</div>
                                  <div className="text-sm leading-7 text-slate-200">{item[textKey]}</div>
                                </div>
                              </label>
                            );
                          })}
                        </div>
                      </Panel>
                    ))}
                  </div>

                  <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                    {Object.entries(selectedRemedy.emergency_contacts).slice(0, 4).map(([label, value]) => (
                      <Panel key={label} title={label} subtitle={value} className="text-center" />
                    ))}
                  </div>
                </motion.div>
              )}

              {activeTab === "Forecast" && (
                <motion.div key="forecast" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="grid gap-4 lg:grid-cols-2">
                  {Object.entries(selectedForecasts)
                    .slice(0, 3)
                    .map(([disease, details]) => {
                      const chartData = [
                        ...details.history.map((row) => ({ label: row.week, observed: row[Object.keys(row).find((key) => key !== "week")] })),
                        ...details.projectedCases.map((value, index) => ({ label: `W+${index + 1}`, forecast: value })),
                      ];
                      return (
                        <ChartFrame key={disease} title={`${disease} outlook`}>
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={chartData}>
                              <CartesianGrid stroke={chartGrid} vertical={false} />
                              <XAxis dataKey="label" stroke="#8ea2c2" />
                              <YAxis stroke="#8ea2c2" />
                              <Tooltip contentStyle={{ background: "#08111f", border: "1px solid rgba(146,165,199,0.16)", borderRadius: 16 }} />
                              <Line type="monotone" dataKey="observed" stroke="#46a2ff" strokeWidth={3} />
                              <Line type="monotone" dataKey="forecast" stroke={severityColor[details.riskLevel]} strokeWidth={3} />
                            </LineChart>
                          </ResponsiveContainer>
                        </ChartFrame>
                      );
                    })}

                  <ChartFrame title="Top districts by forecasted outbreak probability">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={districtComparison}>
                        <CartesianGrid stroke={chartGrid} vertical={false} />
                        <XAxis dataKey="district" angle={-25} height={70} textAnchor="end" stroke="#8ea2c2" />
                        <YAxis stroke="#8ea2c2" />
                        <Tooltip contentStyle={{ background: "#08111f", border: "1px solid rgba(146,165,199,0.16)", borderRadius: 16 }} />
                        <Bar dataKey="risk" fill="#ab8cff" radius={[12, 12, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </ChartFrame>
                </motion.div>
              )}

              {activeTab === "AI Explainability" && (
                <motion.div key="explainability" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="space-y-4">
                  <div className="grid gap-4 lg:grid-cols-2">
                    <ChartFrame title={`Local SHAP drivers | ${selectedDistrict}`}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={shapLocal.length ? shapLocal : fallbackExplainability.features} layout="vertical">
                          <CartesianGrid stroke={chartGrid} horizontal vertical={false} />
                          <XAxis type="number" stroke="#8ea2c2" />
                          <YAxis dataKey="feature" type="category" width={150} stroke="#8ea2c2" />
                          <Tooltip contentStyle={{ background: "#08111f", border: "1px solid rgba(146,165,199,0.16)", borderRadius: 16 }} />
                          <Bar dataKey="value" radius={[0, 12, 12, 0]}>
                            {(shapLocal.length ? shapLocal : fallbackExplainability.features).map((entry, index) => (
                              <Cell key={`${entry.feature}-${index}`} fill={index % 2 === 0 ? "#ff536f" : "#46a2ff"} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </ChartFrame>

                    <ChartFrame title="Global feature importance">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={shapQuery.data?.globalShap ?? []} layout="vertical">
                          <CartesianGrid stroke={chartGrid} horizontal vertical={false} />
                          <XAxis type="number" stroke="#8ea2c2" />
                          <YAxis dataKey="feature" type="category" width={160} stroke="#8ea2c2" />
                          <Tooltip contentStyle={{ background: "#08111f", border: "1px solid rgba(146,165,199,0.16)", borderRadius: 16 }} />
                          <Bar dataKey="value" fill="#ab8cff" radius={[0, 12, 12, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </ChartFrame>
                  </div>

                  <div className="grid gap-4 lg:grid-cols-3">
                    {[
                      ["Primary driver", shapLocal[0]?.feature ?? fallbackExplainability.primaryDriver, "The strongest local feature influencing the current district risk score."],
                      ["Operational takeaway", selectedShapEntry?.explanation_text ?? fallbackExplainability.explanationText, "Short, briefing-friendly AI interpretation of the current signal."],
                      ["System pattern", "Rainfall anomaly, lagged cases, and sanitation repeatedly dominate the statewide risk surface.", "Consistent global pattern across HydroCast explainability outputs."],
                    ].map(([title, value, note]) => (
                      <Panel key={title} title={title} subtitle={value}>
                        <div className="text-sm leading-7 text-slate-300">{note}</div>
                      </Panel>
                    ))}
                  </div>
                </motion.div>
              )}

              {activeTab === "Resource Tracker" && (
                <motion.div key="resources" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="space-y-4">
                  <div className="grid gap-4 lg:grid-cols-[1fr_1.1fr]">
                    <Panel title="Resource availability" subtitle={`District readiness | ${selectedDistrict}`}>
                      <div className="space-y-4">
                        {Object.entries(selectedResources).map(([label, value]) => {
                          const color = value < 45 ? "#ff536f" : value < 70 ? "#ffb24c" : "#2ed39a";
                          return (
                            <div key={label} className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                              <div className="flex items-center justify-between gap-3">
                                <div className="font-bold text-white">{label}</div>
                                <div className="text-lg font-black" style={{ color }}>
                                  {value}%
                                </div>
                              </div>
                              <div className="mt-3 h-3 rounded-full bg-white/8">
                                <div className="h-3 rounded-full" style={{ width: `${value}%`, backgroundColor: color }} />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </Panel>

                    <ChartFrame title="Resource deployment radar">
                      <ResponsiveContainer width="100%" height="100%">
                        <RadarChart data={resourceRadar}>
                          <PolarGrid stroke={chartGrid} />
                          <PolarAngleAxis dataKey="resource" tick={{ fill: "#dbe7ff", fontSize: 12 }} />
                          <PolarRadiusAxis angle={90} stroke="#8ea2c2" tick={{ fill: "#8ea2c2" }} />
                          <Radar dataKey="readiness" stroke="#46a2ff" fill="#46a2ff" fillOpacity={0.25} />
                        </RadarChart>
                      </ResponsiveContainer>
                    </ChartFrame>
                  </div>

                  <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                    {[
                      ["Central PHC", "Operational", "12 beds", "97%"],
                      ["Monsoon Response Unit", "Standby", "8 beds", "88%"],
                      ["Mobile Surveillance Van", "Deployed", "4 teams", "76%"],
                      ["Water Lab Node", "Processing", "22 samples", "81%"],
                      ["Field Ops Cell", "Escalated", "6 blocks", "69%"],
                      ["Medicine Cold Chain", "Stable", "11 units", "92%"],
                    ].map(([title, status, volume, readiness]) => (
                      <Panel key={title} title={status} subtitle={title}>
                        <div className="text-sm text-slate-300">{volume}</div>
                        <div className="mt-4 text-4xl font-black text-white">{readiness}</div>
                      </Panel>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </section>

          <aside className="space-y-4">
            <Panel title="Operational Right Rail" subtitle="Alerts, explainability, and model health">
              <div className="space-y-3">
                {alerts.slice(0, 4).map((alert) => (
                  <div
                    key={`${alert.district}-${alert.title}`}
                    className="rounded-[1.1rem] border bg-white/[0.03] p-4"
                    style={{ borderColor: `${severityColor[alert.severity]}50` }}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="font-bold text-white">{alert.district}</div>
                      {severityBadge(alert.severity)}
                    </div>
                    <div className="mt-2 font-semibold text-slate-100">{alert.title}</div>
                    <div className="mt-2 text-sm leading-7 text-slate-300">{alert.message}</div>
                  </div>
                ))}
              </div>
            </Panel>

            <Panel title="SHAP mini bars" subtitle={`${selectedDistrict} feature pressure`}>
              <div className="space-y-4">
                {miniBarFeatures.slice(0, 5).map((item) => (
                  <div key={item.feature}>
                    <div className="flex items-center justify-between gap-3 text-sm">
                      <span className="text-slate-200">{item.feature}</span>
                      <span className="font-black text-white">{item.value.toFixed(2)}</span>
                    </div>
                    <div className="mt-2 h-2 rounded-full bg-white/8">
                      <div className="h-2 rounded-full bg-info" style={{ width: `${Math.min(100, item.value * 100)}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </Panel>

            <Panel title="Quick remedy suggestions" subtitle={`${selectedRemedy.disease} action starters`}>
              <div className="space-y-3">
                {selectedRemedy.emergency_actions.slice(0, 3).map((item, index) => (
                  <div key={`${item.text}-${index}`} className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                    <div className="mb-2">{severityBadge(String(item.priority).toLowerCase().includes("critical") ? "critical" : "high")}</div>
                    <div className="text-sm leading-7 text-slate-200">{item.text}</div>
                  </div>
                ))}
              </div>
            </Panel>

            <Panel title="Model metrics" subtitle="HydroCast operating profile">
              {(() => {
                const hydrocast = baselines.find((item) => item.Model.includes("HydroCast")) ?? baselines[0];
                return (
                  <div className="grid grid-cols-2 gap-4">
                    {[
                      ["F1", hydrocast.F1_Macro.toFixed(3)],
                      ["Precision", hydrocast.Precision.toFixed(3)],
                      ["Recall", hydrocast.Recall.toFixed(3)],
                      ["Top model", "HydroCast"],
                    ].map(([label, value]) => (
                      <div key={label} className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                        <div className="text-[11px] font-extrabold uppercase tracking-[0.22em] text-slate-500">{label}</div>
                        <div className="mt-3 text-2xl font-black text-white">{value}</div>
                      </div>
                    ))}
                  </div>
                );
              })()}
            </Panel>
          </aside>
        </div>
      </div>
    </main>
  );
}


