import { promises as fs } from "fs";
import path from "path";
import { execFile } from "child_process";
import { promisify } from "util";

const execFileAsync = promisify(execFile);

const frontendRoot = process.cwd();
const hydrocastRoot = path.resolve(frontendRoot, "..");
const dataDir = path.join(frontendRoot, "public", "data");
const bundlePath = path.join(dataDir, "dashboard-data.json");
const frontendGeoJsonPath = path.join(dataDir, "maharashtra_districts.geojson");
const sourceGeoJsonPath = path.join(hydrocastRoot, "src", "data", "geojson", "maharashtra_districts.geojson");
const exporterPath = path.join(frontendRoot, "export_dashboard_data.py");
const pythonPath = path.join(hydrocastRoot, ".venv", "Scripts", "python.exe");

const sourceFiles = [
  path.join(hydrocastRoot, "results", "baseline_comparison.csv"),
  path.join(hydrocastRoot, "results", "classification_metrics.csv"),
  path.join(hydrocastRoot, "results", "lead_time.json"),
  path.join(hydrocastRoot, "results", "shap_values.json"),
  path.join(hydrocastRoot, "src", "data", "processed", "epiclim_maharashtra_merged.csv"),
  sourceGeoJsonPath,
];

let refreshPromise = null;

async function readJson(filePath) {
  const content = await fs.readFile(filePath, "utf-8");
  return JSON.parse(content);
}

async function getMTime(filePath) {
  try {
    const stats = await fs.stat(filePath);
    return stats.mtimeMs;
  } catch {
    return 0;
  }
}

async function syncGeoJsonIfNeeded() {
  const [sourceTime, frontendTime] = await Promise.all([getMTime(sourceGeoJsonPath), getMTime(frontendGeoJsonPath)]);
  if (sourceTime > frontendTime) {
    await fs.mkdir(dataDir, { recursive: true });
    await fs.copyFile(sourceGeoJsonPath, frontendGeoJsonPath);
  }
}

async function rebuildBundle() {
  await fs.mkdir(dataDir, { recursive: true });
  await execFileAsync(pythonPath, [exporterPath], {
    cwd: frontendRoot,
    windowsHide: true,
  });
  await syncGeoJsonIfNeeded();
}

async function ensureFreshDashboardBundle() {
  const bundleTime = await getMTime(bundlePath);
  const sourceTimes = await Promise.all(sourceFiles.map((filePath) => getMTime(filePath)));
  const newestSourceTime = Math.max(...sourceTimes);

  if (bundleTime >= newestSourceTime && bundleTime > 0) {
    await syncGeoJsonIfNeeded();
    return;
  }

  if (!refreshPromise) {
    refreshPromise = rebuildBundle().finally(() => {
      refreshPromise = null;
    });
  }

  await refreshPromise;
}

export async function getDashboardBundle() {
  await ensureFreshDashboardBundle();
  return readJson(bundlePath);
}

export async function getMaharashtraGeoJson() {
  await ensureFreshDashboardBundle();
  return readJson(frontendGeoJsonPath);
}

export async function getSummaryPayload() {
  const bundle = await getDashboardBundle();
  return {
    overview: bundle.overview,
    districtRankings: bundle.districtRankings,
    districtDetails: bundle.districtDetails,
    baselines: bundle.baselines,
    alerts: bundle.alerts,
    metadata: bundle.metadata,
  };
}

export async function getForecastPayload() {
  const bundle = await getDashboardBundle();
  return {
    forecasts: bundle.forecasts,
    districtRankings: bundle.districtRankings,
    districtDetails: bundle.districtDetails,
  };
}

export async function getRiskMapPayload() {
  const [bundle, geojson] = await Promise.all([getDashboardBundle(), getMaharashtraGeoJson()]);
  return {
    districts: bundle.districtRankings.map((item) => ({
      district: item.district,
      topDisease: item.top_disease,
      riskLevel: item.risk_level,
      riskScore: item.risk_score,
      rainfallAnomalyPct: item.rainfall_anomaly_pct,
      sanitationCoveragePct: item.sanitation_coverage_pct,
      caseCount: item.case_count,
      latitude: bundle.districtDetails?.[item.district]?.latitude ?? 0,
      longitude: bundle.districtDetails?.[item.district]?.longitude ?? 0,
    })),
    geojson,
  };
}

export async function getShapPayload() {
  const bundle = await getDashboardBundle();
  return {
    shapValues: bundle.shapValues,
    globalShap: bundle.globalShap,
  };
}

export async function getResourcesPayload() {
  const bundle = await getDashboardBundle();
  return {
    resources: bundle.resources,
    remedies: bundle.remedies,
  };
}
