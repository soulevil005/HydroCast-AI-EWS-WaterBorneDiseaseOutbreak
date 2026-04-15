"use client";

import { useEffect, useMemo, useRef } from "react";
import L from "leaflet";

const severityColor = {
  critical: "#ff536f",
  high: "#ffb24c",
  medium: "#46a2ff",
  low: "#2ed39a",
};

function styleFeature(feature, districtMap) {
  const districtName = feature.properties?.DISTRICT || feature.properties?.district;
  const district = districtMap[districtName] || {};
  const color = severityColor[district.riskLevel] || "#2ed39a";
  return {
    color,
    weight: district.isSelected ? 3 : 1.6,
    opacity: 1,
    fillColor: color,
    fillOpacity: district.mode === "bubble" ? 0.08 : district.mode === "both" ? 0.28 : 0.4,
  };
}

export default function RiskMap({ geojson, districts, selectedDistrict, mapMode, onDistrictSelect }) {
  const mapRef = useRef(null);
  const containerRef = useRef(null);

  const districtMap = useMemo(() => {
    return Object.fromEntries(
      districts.map((district) => [
        district.district,
        {
          ...district,
          isSelected: district.district === selectedDistrict,
          mode: mapMode,
        },
      ]),
    );
  }, [districts, selectedDistrict, mapMode]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return undefined;

    const map = L.map(containerRef.current, {
      zoomControl: false,
      attributionControl: false,
    }).setView([19.3, 75.4], 6);

    L.control.zoom({ position: "bottomright" }).addTo(map);

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      subdomains: "abcd",
      maxZoom: 19,
    }).addTo(map);

    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!mapRef.current || !geojson) return undefined;
    const map = mapRef.current;

    const polygonLayer = L.geoJSON(geojson, {
      style: (feature) => styleFeature(feature, districtMap),
      onEachFeature: (feature, layer) => {
        const districtName = feature.properties?.DISTRICT || feature.properties?.district;
        const district = districtMap[districtName];
        if (!district) return;

        layer.bindTooltip(
          `
            <div style="min-width:200px">
              <div style="font-size:12px; letter-spacing:.12em; text-transform:uppercase; color:${severityColor[district.riskLevel]}; font-weight:700;">
                ${district.district}
              </div>
              <div style="font-size:20px; font-weight:800; margin-top:6px;">${Math.round(district.riskScore * 100)}% risk</div>
              <div style="margin-top:8px; color:#d7e6ff;">${district.topDisease} lead signal · ${district.caseCount} recent cases</div>
              <div style="margin-top:8px; color:#9ab4dd;">Rainfall anomaly ${district.rainfallAnomalyPct}% · sanitation ${district.sanitationCoveragePct}%</div>
            </div>
          `,
          {
            sticky: true,
            direction: "top",
            className: "map-tooltip",
          },
        );

        layer.on({
          mouseover: () => layer.setStyle({ weight: 2.8, fillOpacity: 0.52 }),
          mouseout: () => polygonLayer.resetStyle(layer),
          click: () => onDistrictSelect?.(districtName),
        });
      },
    });

    polygonLayer.addTo(map);

    const bubbles = L.layerGroup();
    if (mapMode !== "choropleth") {
      districts.forEach((district) => {
        const color = severityColor[district.riskLevel] || "#2ed39a";
        const radius = Math.max(8, Math.min(26, 8 + district.caseCount * 0.7));
        const marker = L.circleMarker([district.latitude, district.longitude], {
          radius,
          color,
          weight: district.district === selectedDistrict ? 3 : 1.6,
          fillColor: color,
          fillOpacity: 0.65,
        });

        marker.bindTooltip(
          `<div style="min-width:180px"><strong>${district.district}</strong><br/>${district.topDisease} · ${Math.round(
            district.riskScore * 100,
          )}% risk</div>`,
          { sticky: true, className: "map-tooltip" },
        );
        marker.on("click", () => onDistrictSelect?.(district.district));
        marker.addTo(bubbles);
      });
    }

    bubbles.addTo(map);

    return () => {
      polygonLayer.remove();
      bubbles.remove();
    };
  }, [districtMap, districts, geojson, mapMode, onDistrictSelect, selectedDistrict]);

  return <div ref={containerRef} className="h-[34rem] w-full rounded-[1.25rem]" />;
}
