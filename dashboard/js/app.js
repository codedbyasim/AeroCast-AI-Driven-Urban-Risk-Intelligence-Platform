/**
 * AeroCast — Lahore Urban Environmental Risk Intelligence Platform
 * Module M6: Web GIS Situational Command Center Client
 */

// Global Dashboard State
const state = {
  activeLayer: 'forecast', // 'forecast' | 'pm25_live' | 'heat_island' | 'flash_flood' | 'composite' | 'firms_fires'
  activeTab: 'zone',       // 'zone' | 'alerts'
  alertLang: 'en',         // 'en' | 'ur' | 'roman_ur'
  zoneAiLang: 'en',        // 'en' | 'ur' | 'roman_ur'
  selectedZoneId: null,
  selectedZoneData: null,
  geoJsonData: null,
  geoJsonLayer: null,
  fireMarkersLayer: null,
  fireHotspotsData: null,
  zonesIndex: {},
  summaryData: null,
  activeAlerts: [],
  forecastChart: null,
  map: null,
};

// API Base URL (Relative for seamless local/proxy deployment)
const API_BASE = '';

// Color Scales & Thresholds
const COLOR_RAMPS = {
  pm25: [
    { limit: 12.0, color: '#10b981', label: 'Good (0 - 12)' },
    { limit: 35.4, color: '#f59e0b', label: 'Moderate (12 - 35)' },
    { limit: 55.4, color: '#f97316', label: 'Sensitive (35 - 55)' },
    { limit: 150.4, color: '#ef4444', label: 'Unhealthy (55 - 150)' },
    { limit: 250.4, color: '#a855f7', label: 'Very Unhealthy (150 - 250)' },
    { limit: Infinity, color: '#e11d48', label: 'Hazardous (> 250)' },
  ],
  heat_island: [
    { limit: 0.25, color: '#10b981', label: 'Low (0.0 - 0.25)' },
    { limit: 0.50, color: '#f59e0b', label: 'Moderate (0.25 - 0.50)' },
    { limit: 0.75, color: '#f97316', label: 'High (0.50 - 0.75)' },
    { limit: Infinity, color: '#ef4444', label: 'Severe (> 0.75)' },
  ],
  flash_flood: [
    { limit: 0.25, color: '#38bdf8', label: 'Low (0.0 - 0.25)' },
    { limit: 0.50, color: '#0284c7', label: 'Moderate (0.25 - 0.50)' },
    { limit: 0.75, color: '#f97316', label: 'High (0.50 - 0.75)' },
    { limit: Infinity, color: '#ef4444', label: 'Severe Alert (> 0.75)' },
  ],
  composite: [
    { limit: 0.25, color: '#10b981', label: 'Low (0.0 - 0.25)' },
    { limit: 0.40, color: '#f59e0b', label: 'Moderate (0.25 - 0.40)' },
    { limit: 0.60, color: '#f97316', label: 'Elevated (0.40 - 0.60)' },
    { limit: Infinity, color: '#ef4444', label: 'Critical Threat (> 0.60)' },
  ]
};

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  initClock();
  initMap();
  initEventListeners();
  loadAllData();
});

/**
 * 1. Initialize Real-time Clock (Pakistan Standard Time UTC+5)
 */
function initClock() {
  const clockEl = document.getElementById('current-pkt-time');
  function update() {
    const now = new Date();
    // Format in PKT (UTC+5)
    const options = { timeZone: 'Asia/Karachi', hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' };
    if (clockEl) {
      clockEl.textContent = new Intl.DateTimeFormat('en-GB', options).format(now);
    }
  }
  update();
  setInterval(update, 1000);
}

/**
 * 2. Initialize Leaflet Map Centered on Lahore
 */
function initMap() {
  // Lahore District Centroid [31.5204, 74.3587]
  state.map = L.map('gis-map', {
    zoomControl: false,
    attributionControl: true,
  }).setView([31.5204, 74.3587], 11);

  // Add Zoom Control to Top Right
  L.control.zoom({ position: 'topright' }).addTo(state.map);

  // High-Contrast Solid Dark Basemap (Esri World Dark Gray Canvas - Clean, no watermark, free)
  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
    attribution: '&copy; Esri, HERE, Garmin, OpenStreetMap contributors | AeroCast',
    maxZoom: 16,
  }).addTo(state.map);

  // Reference overlay for clear district labels
  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}', {
    attribution: '',
    maxZoom: 16,
  }).addTo(state.map);
}

/**
 * 3. Setup UI Event Listeners
 */
function initEventListeners() {
  // Layer Switcher Buttons
  const layerBtns = document.querySelectorAll('.layer-btn');
  layerBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      const target = e.currentTarget;
      const layerKey = target.getAttribute('data-layer');
      if (layerKey && layerKey !== state.activeLayer) {
        layerBtns.forEach(b => b.classList.remove('active'));
        target.classList.add('active');
        state.activeLayer = layerKey;
        if (layerKey === 'firms_fires') {
          showFirmsFireLayer();
        } else {
          hideFirmsFireLayer();
          updateMapLayerStyle();
        }
        updateLegend();
      }
    });
  });

  // Panel Tab Switcher (Zone vs Alerts)
  const tabBtns = document.querySelectorAll('.panel-tab-btn');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      const target = e.currentTarget;
      const tabKey = target.getAttribute('data-tab');
      if (tabKey) {
        switchPanelTab(tabKey);
      }
    });
  });

  // Header Alert Toggle & Broadcast Banner Action
  const alertToggleBtn = document.getElementById('btn-toggle-alerts');
  if (alertToggleBtn) {
    alertToggleBtn.addEventListener('click', () => switchPanelTab('alerts'));
  }

  const broadcastOpenBtn = document.getElementById('btn-broadcast-open');
  if (broadcastOpenBtn) {
    broadcastOpenBtn.addEventListener('click', () => switchPanelTab('alerts'));
  }

  // AI Copilot Modal Buttons
  const openCopilotBtn = document.getElementById('btn-open-copilot');
  const closeCopilotBtn = document.getElementById('btn-close-copilot');
  const copilotBackdrop = document.getElementById('modal-copilot-backdrop');
  if (openCopilotBtn) openCopilotBtn.addEventListener('click', openCopilotModal);
  if (closeCopilotBtn) closeCopilotBtn.addEventListener('click', closeCopilotModal);
  if (copilotBackdrop) {
    copilotBackdrop.addEventListener('click', (e) => {
      if (e.target === copilotBackdrop) closeCopilotModal();
    });
  }

  // Copilot Send Button & Enter Key
  const sendCopilotBtn = document.getElementById('btn-send-copilot');
  const copilotInput = document.getElementById('copilot-user-input');
  if (sendCopilotBtn) {
    sendCopilotBtn.addEventListener('click', () => {
      const q = copilotInput ? copilotInput.value.trim() : '';
      if (q) sendCopilotMessage(q);
    });
  }
  if (copilotInput) {
    copilotInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const q = copilotInput.value.trim();
        if (q) sendCopilotMessage(q);
      }
    });
  }

  // Copilot Quick Prompt Chips
  const quickChips = document.querySelectorAll('.quick-chip');
  quickChips.forEach(chip => {
    chip.addEventListener('click', (e) => {
      const prompt = e.currentTarget.getAttribute('data-prompt');
      if (prompt) {
        if (copilotInput) copilotInput.value = prompt;
        sendCopilotMessage(prompt);
      }
    });
  });

  // Situation Report Modal Buttons
  const openSitrepBtn = document.getElementById('btn-open-sitrep');
  const closeSitrepBtn = document.getElementById('btn-close-sitrep');
  const sitrepBackdrop = document.getElementById('modal-sitrep-backdrop');
  const copySitrepBtn = document.getElementById('btn-copy-sitrep');
  const printSitrepBtn = document.getElementById('btn-print-sitrep');
  if (openSitrepBtn) openSitrepBtn.addEventListener('click', openSitrepModal);
  if (closeSitrepBtn) closeSitrepBtn.addEventListener('click', closeSitrepModal);
  if (copySitrepBtn) copySitrepBtn.addEventListener('click', copySitrepText);
  if (printSitrepBtn) printSitrepBtn.addEventListener('click', printSitrep);
  if (sitrepBackdrop) {
    sitrepBackdrop.addEventListener('click', (e) => {
      if (e.target === sitrepBackdrop) closeSitrepModal();
    });
  }

  // Zone Detail AI Mitigation Plan Button & Mini Language Toggles
  const genAiPlanBtn = document.getElementById('btn-generate-ai-plan');
  if (genAiPlanBtn) {
    genAiPlanBtn.addEventListener('click', generateZoneMitigationPlan);
  }

  const zoneAiLangBtns = document.querySelectorAll('.ai-lang-btn');
  zoneAiLangBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      const lang = e.currentTarget.getAttribute('data-lang');
      if (lang && lang !== state.zoneAiLang) {
        zoneAiLangBtns.forEach(b => b.classList.remove('active'));
        e.currentTarget.classList.add('active');
        state.zoneAiLang = lang;
        if (state.selectedZoneId && document.getElementById('ai-plan-content').style.display !== 'none') {
          generateZoneMitigationPlan();
        }
      }
    });
  });

  // Trigger Dispatch Scan Button
  const dispatchBtn = document.getElementById('btn-trigger-dispatch');
  if (dispatchBtn) {
    dispatchBtn.addEventListener('click', async () => {
      dispatchBtn.disabled = true;
      dispatchBtn.innerHTML = `
        <svg width="14" height="14" class="pulse-red" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/></svg>
        <span>Scanning 241 Zones...</span>
      `;
      try {
        const resp = await fetch(`${API_BASE}/api/v1/alerts/dispatch?force_reevaluate=true`, { method: 'POST' });
        if (resp.ok) {
          await loadAlerts();
        }
      } catch (err) {
        console.error("Alert dispatch error:", err);
      } finally {
        dispatchBtn.disabled = false;
        dispatchBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/></svg>
          <span>Scan & Dispatch</span>
        `;
      }
    });
  }

  // Multi-Lingual Language Switcher Chips
  const langChips = document.querySelectorAll('.lang-chip');
  langChips.forEach(chip => {
    chip.addEventListener('click', (e) => {
      const lang = e.currentTarget.getAttribute('data-lang');
      if (lang && lang !== state.alertLang) {
        langChips.forEach(c => c.classList.remove('active'));
        e.currentTarget.classList.add('active');
        state.alertLang = lang;
        renderAlertsList();
      }
    });
  });

  // Refresh Button
  const refreshBtn = document.getElementById('btn-refresh-data');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => {
      refreshBtn.disabled = true;
      refreshBtn.classList.add('loading');
      loadAllData().finally(() => {
        refreshBtn.disabled = false;
        refreshBtn.classList.remove('loading');
      });
    });
  }

  // Reset Map View Button
  const resetBtn = document.getElementById('btn-reset-view');
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      state.map.setView([31.5204, 74.3587], 11);
    });
  }

  // Search Input
  const searchInput = document.getElementById('zone-search-input');
  const clearBtn = document.getElementById('btn-clear-search');
  const suggestionsBox = document.getElementById('search-suggestions');

  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      const query = e.target.value.trim().toLowerCase();
      if (query.length > 0) {
        clearBtn.style.display = 'block';
        showSearchSuggestions(query);
      } else {
        clearBtn.style.display = 'none';
        suggestionsBox.style.display = 'none';
      }
    });

    searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const query = searchInput.value.trim().toUpperCase();
        if (state.zonesIndex[query]) {
          selectZone(query);
          suggestionsBox.style.display = 'none';
        }
      }
    });
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      searchInput.value = '';
      clearBtn.style.display = 'none';
      suggestionsBox.style.display = 'none';
    });
  }

  // Close Panel Button
  const closePanelBtn = document.getElementById('btn-close-panel');
  if (closePanelBtn) {
    closePanelBtn.addEventListener('click', () => {
      deselectZone();
    });
  }
}

/**
 * Toast Notification Utility
 */
function showToast(message, type = 'info') {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container';
    document.body.appendChild(container);
  }

  const icons = {
    info: 'ℹ️',
    warning: '⚠️',
    error: '❌',
    success: '✅',
  };

  const toast = document.createElement('div');
  toast.className = `toast-item ${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || 'ℹ️'}</span>
    <span class="toast-text">${message}</span>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(15px) scale(0.95)';
    setTimeout(() => {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 350);
  }, 4500);
}

/**
 * 4. Load Complete System Data (GeoJSON + Unified Priority Summary + Alerts)
 */
async function loadAllData() {
  try {
    const [geoResp, summaryResp, alertsResp] = await Promise.all([
      fetch(`${API_BASE}/api/v1/spatial/geojson`),
      fetch(`${API_BASE}/api/v1/hazards/unified-risk-summary?top_n=10`),
      fetch(`${API_BASE}/api/v1/alerts/active`),
    ]);

    if (!geoResp.ok) throw new Error(`Failed to load GeoJSON: HTTP ${geoResp.status}`);
    state.geoJsonData = await geoResp.json();

    if (summaryResp.ok) {
      state.summaryData = await summaryResp.json();
    }

    if (alertsResp.ok) {
      const alertPayload = await alertsResp.json();
      state.activeAlerts = alertPayload.alerts || [];
    }

    // Build Fast Index
    state.zonesIndex = {};
    if (state.geoJsonData && state.geoJsonData.features) {
      state.geoJsonData.features.forEach(f => {
        const zid = f.properties.zone_id;
        state.zonesIndex[zid] = f;
      });
    }

    renderGeoJsonLayer();
    updateKPIs();
    updatePriorityFeed();
    updateLegend();
    updateAlertUI();

  } catch (err) {
    console.error("Data loading error:", err);
    showToast("AeroCast Server: Unable to fetch live telemetry. Check backend connection.", "error");
  }
}

/**
 * 5. Render Leaflet GeoJSON Polygon Layer
 */
function renderGeoJsonLayer() {
  if (state.geoJsonLayer) {
    state.map.removeLayer(state.geoJsonLayer);
  }

  state.geoJsonLayer = L.geoJSON(state.geoJsonData, {
    style: styleFeature,
    onEachFeature: (feature, layer) => {
      const props = feature.properties || {};
      const zid = props.zone_id;

      // Tooltip on Hover
      layer.bindTooltip(() => getTooltipContent(props), {
        className: 'custom-map-tooltip',
        sticky: true,
      });

      // Click to Select Zone
      layer.on('click', () => {
        selectZone(zid);
      });

      // Hover — only brighten fill, keep existing border as-is
      layer.on('mouseover', (e) => {
        const l = e.target;
        if (state.selectedZoneId !== zid) {
          l.setStyle({ fillOpacity: 0.92 });
        }
      });

      layer.on('mouseout', (e) => {
        if (state.selectedZoneId !== zid) {
          l.setStyle(styleFeature(l.feature));
        }
      });
    }
  }).addTo(state.map);
}

/**
 * 6. Dynamic Feature Styling per Active Layer
 */
function styleFeature(feature) {
  const props = feature.properties || {};
  const zid = props.zone_id;
  const isSelected = (zid === state.selectedZoneId);
  const isDirect = props.is_direct_sensor === true;

  if (state.activeLayer === 'firms_fires') {
    return {
      fillColor: '#0b0f17',
      fillOpacity: isSelected ? 0.60 : 0.15,
      weight: isSelected ? 2.5 : 0.8,
      color: isSelected ? '#38bdf8' : '#232d42',
      dashArray: (!isDirect && !isSelected) ? '2, 2' : null,
    };
  }

  const color = getFeatureColor(props);

  return {
    fillColor: color,
    fillOpacity: isSelected ? 0.90 : 0.70,
    weight: isSelected ? 3.5 : (isDirect ? 2.0 : 1.0),
    color: isSelected ? '#ffffff' : (isDirect ? '#38bdf8' : '#232d42'),
    dashArray: (!isDirect && !isSelected) ? '3, 3' : null,
  };
}

function getFeatureColor(props) {
  switch (state.activeLayer) {
    case 'forecast': {
      const val = parseFloat(props.forecast_pm25_24h) || 65.0;
      return getColorFromRamp(val, COLOR_RAMPS.pm25);
    }
    case 'pm25_live': {
      const val = parseFloat(props.pm25_current) || 65.0;
      return getColorFromRamp(val, COLOR_RAMPS.pm25);
    }
    case 'heat_island': {
      const val = parseFloat(props.heat_island_score ?? props.uhi_risk_score) || 0.4;
      return getColorFromRamp(val, COLOR_RAMPS.heat_island);
    }
    case 'flash_flood': {
      const val = parseFloat(props.flood_risk_score) || 0.2;
      return getColorFromRamp(val, COLOR_RAMPS.flash_flood);
    }
    case 'composite': {
      const p = (parseFloat(props.forecast_pm25_24h) || 65.0) / 250.0;
      const h = parseFloat(props.heat_island_score ?? props.uhi_risk_score) || 0.4;
      const f = parseFloat(props.flood_risk_score) || 0.2;
      const comp = 0.40 * Math.min(1.0, p) + 0.30 * h + 0.30 * f;
      return getColorFromRamp(comp, COLOR_RAMPS.composite);
    }
    case 'firms_fires':
      return '#121824';
    default:
      return '#3b82f6';
  }
}

function getColorFromRamp(val, ramp) {
  for (const step of ramp) {
    if (val <= step.limit) {
      return step.color;
    }
  }
  return ramp[ramp.length - 1].color;
}

function updateMapLayerStyle() {
  if (state.geoJsonLayer) {
    state.geoJsonLayer.setStyle(styleFeature);
  }
}

/**
 * 7. Hover Tooltip Content Formatter
 */
function getTooltipContent(props) {
  const zid = props.zone_id || 'ZONE';
  const name = props.zone_name || zid;
  let metricLabel = '';
  let metricVal = '';

  switch (state.activeLayer) {
    case 'forecast':
      metricLabel = '24h Forecast';
      metricVal = `${Math.round(props.forecast_pm25_24h || 65)} µg/m³ (${props.hazard_category_24h || 'Moderate'})`;
      break;
    case 'pm25_live':
      metricLabel = 'Current PM2.5';
      metricVal = `${Math.round(props.pm25_current || 65)} µg/m³`;
      break;
    case 'heat_island':
      metricLabel = 'UHI Anomaly';
      metricVal = `${(props.heat_island_score || 0.4).toFixed(2)} (${props.heat_risk_category || 'Moderate'})`;
      break;
    case 'flash_flood':
      metricLabel = 'Flood Risk';
      metricVal = `${(props.flood_risk_score || 0.2).toFixed(2)} (${props.flood_risk_category || 'Low'})`;
      break;
    case 'composite':
      metricLabel = 'Composite Risk';
      metricVal = 'Multi-Hazard Threat';
      break;
    case 'firms_fires':
      metricLabel = 'Zone Baseline PM2.5';
      metricVal = `${Math.round(props.pm25_current || 65)} µg/m³`;
      break;
  }

  const sensorTag = props.is_direct_sensor
    ? '<span style="color: #38bdf8; font-weight: bold;">● Direct Station</span>'
    : '<span style="color: #94a3b8;">○ Spatial Kriging</span>';

  return `
    <div style="font-family: 'Plus Jakarta Sans', sans-serif;">
      <div style="font-weight: 700; font-size: 13px; color: #ffffff;">${zid}</div>
      <div style="font-size: 11px; color: #94a3b8; margin-bottom: 4px;">${name}</div>
      <div style="font-size: 12px; font-weight: 600; color: #38bdf8;">${metricLabel}: ${metricVal}</div>
      <div style="font-size: 10px; margin-top: 4px;">${sensorTag}</div>
    </div>
  `;
}

/**
 * 8. Legend Renderer
 */
function updateLegend() {
  const legendTitle = document.getElementById('legend-title');
  const scaleEl = document.getElementById('legend-scale');
  if (!legendTitle || !scaleEl) return;

  let ramp = COLOR_RAMPS.pm25;
  let title = '24h PM2.5 Forecast (µg/m³)';

  switch (state.activeLayer) {
    case 'forecast':
      title = '24h PM2.5 Forecast (µg/m³)';
      ramp = COLOR_RAMPS.pm25;
      break;
    case 'pm25_live':
      title = 'Live Spatial PM2.5 (µg/m³)';
      ramp = COLOR_RAMPS.pm25;
      break;
    case 'heat_island':
      title = 'Urban Heat Island Index (0 - 1)';
      ramp = COLOR_RAMPS.heat_island;
      break;
    case 'flash_flood':
      title = 'Flash Flood Runoff Risk (0 - 1)';
      ramp = COLOR_RAMPS.flash_flood;
      break;
    case 'composite':
      title = 'Multi-Hazard Risk Index (0 - 1)';
      ramp = COLOR_RAMPS.composite;
      break;
    case 'firms_fires':
      legendTitle.textContent = 'NASA FIRMS Satellite Fire Telemetry';
      scaleEl.innerHTML = `
        <div class="legend-row">
          <span class="legend-color-chip" style="background-color: #ef4444; box-shadow: 0 0 8px #ef4444;"></span>
          <span>High Thermal Intensity (&ge; 25 MW)</span>
        </div>
        <div class="legend-row">
          <span class="legend-color-chip" style="background-color: #f97316;"></span>
          <span>Moderate Fire Hotspot (15 - 25 MW)</span>
        </div>
        <div class="legend-row">
          <span class="legend-color-chip" style="background-color: #f59e0b;"></span>
          <span>Crop Stubble Anomaly (&lt; 15 MW)</span>
        </div>
      `;
      return;
  }

  legendTitle.textContent = title;
  scaleEl.innerHTML = ramp.map(step => `
    <div class="legend-row">
      <span class="legend-color-chip" style="background-color: ${step.color};"></span>
      <span>${step.label}</span>
    </div>
  `).join('');
}

/**
 * 9. Top KPI Cards Calculation & Updates
 */
function updateKPIs() {
  if (!state.geoJsonData || !state.geoJsonData.features) return;
  const features = state.geoJsonData.features;

  let sumPm25 = 0;
  let maxForecast = -1;
  let maxForecastZone = '';
  let maxUhi = -1;
  let maxUhiZone = '';
  let maxFlood = -1;
  let maxFloodZone = '';

  features.forEach(f => {
    const p = f.properties || {};
    const pm25 = parseFloat(p.pm25_current) || 0;
    const fc = parseFloat(p.forecast_pm25_24h) || 0;
    const uhi = parseFloat(p.heat_island_score ?? p.uhi_risk_score ?? p.heat_island_risk_score) || 0;
    const fl = parseFloat(p.flood_risk_score) || 0;

    sumPm25 += pm25;
    if (fc > maxForecast) { maxForecast = fc; maxForecastZone = p.zone_id; }
    if (uhi > maxUhi) { maxUhi = uhi; maxUhiZone = p.zone_id; }
    if (fl > maxFlood) { maxFlood = fl; maxFloodZone = p.zone_id; }
  });

  const meanPm25 = (sumPm25 / features.length).toFixed(1);

  // Update DOM
  const meanPm25El = document.getElementById('kpi-mean-pm25');
  if (meanPm25El) meanPm25El.textContent = meanPm25;

  const peakFcEl = document.getElementById('kpi-peak-forecast-val');
  if (peakFcEl) peakFcEl.textContent = maxForecast > 0 ? maxForecast.toFixed(1) : '--';
  const peakFcZoneEl = document.getElementById('kpi-peak-forecast-zone');
  if (peakFcZoneEl) peakFcZoneEl.textContent = maxForecastZone ? `Zone ${maxForecastZone}` : 'Model v4.0 Hurdle';

  const peakUhiEl = document.getElementById('kpi-peak-uhi-val');
  if (peakUhiEl) peakUhiEl.textContent = maxUhi >= 0 ? maxUhi.toFixed(2) : '--';
  const peakUhiZoneEl = document.getElementById('kpi-peak-uhi-zone');
  if (peakUhiZoneEl) peakUhiZoneEl.textContent = maxUhiZone ? `Zone ${maxUhiZone}` : 'Copernicus NDVI & Impervious';

  const peakFloodEl = document.getElementById('kpi-peak-flood-val');
  if (peakFloodEl) peakFloodEl.textContent = maxFlood >= 0 ? maxFlood.toFixed(2) : '--';
  const peakFloodZoneEl = document.getElementById('kpi-peak-flood-zone');
  if (peakFloodZoneEl) peakFloodZoneEl.textContent = maxFloodZone ? `Zone ${maxFloodZone}` : 'Precipitation & Slope';
}

/**
 * 10. Select Zone & Drill-Down Detail Drawer
 */
async function selectZone(zoneId) {
  state.selectedZoneId = zoneId;
  updateMapLayerStyle();

  const feat = state.zonesIndex[zoneId];
  if (!feat) return;

  // Zoom & Pan to Polygon
  const layer = findLayerByZoneId(zoneId);
  if (layer) {
    state.map.fitBounds(layer.getBounds(), { maxZoom: 13, padding: [40, 40] });
  }

  // Toggle UI containers
  const emptyEl = document.getElementById('panel-empty-state');
  const contentEl = document.getElementById('panel-content');
  const closeBtn = document.getElementById('btn-close-panel');
  if (emptyEl) emptyEl.style.display = 'none';
  if (contentEl) contentEl.style.display = 'flex';
  if (closeBtn) closeBtn.style.display = 'block';

  // Fetch Full Zone Snapshot from REST API
  try {
    const resp = await fetch(`${API_BASE}/api/v1/zones/${zoneId}`);
    if (!resp.ok) throw new Error('Zone snapshot fetch failed');
    const data = await resp.json();
    populateZoneDetails(data);
  } catch (err) {
    console.error('Failed to load zone snapshot:', err);
  }
}

function deselectZone() {
  state.selectedZoneId = null;
  updateMapLayerStyle();

  const emptyEl = document.getElementById('panel-empty-state');
  const contentEl = document.getElementById('panel-content');
  const closeBtn = document.getElementById('btn-close-panel');
  if (emptyEl) emptyEl.style.display = 'flex';
  if (contentEl) contentEl.style.display = 'none';
  if (closeBtn) closeBtn.style.display = 'none';

  document.getElementById('panel-zone-name').textContent = 'Select a Zone';
  document.getElementById('panel-zone-id').textContent = 'Click any grid polygon on map';
}

function findLayerByZoneId(zoneId) {
  let found = null;
  if (state.geoJsonLayer) {
    state.geoJsonLayer.eachLayer(l => {
      if (l.feature && l.feature.properties && l.feature.properties.zone_id === zoneId) {
        found = l;
      }
    });
  }
  return found;
}

/**
 * 11. Populate Zone Intelligence Panel
 */
function populateZoneDetails(data) {
  const zid = data.zone_id;
  const name = data.zone_name || zid;

  document.getElementById('panel-zone-name').textContent = name;
  document.getElementById('panel-zone-id').textContent = `${zid} • (Row ${data.grid_row}, Col ${data.grid_col})`;

  // Current Conditions
  const curr = data.current_conditions || {};
  document.getElementById('val-current-pm25').textContent = `${Math.round(curr.pm25_current_ug_m3 || 65)} µg/m³`;
  document.getElementById('val-kriging-conf').textContent = `Confidence: ${(curr.spatial_kriging_confidence * 100).toFixed(0)}%`;

  // 24h Forecast
  const fc = data.forecast_24h_aqi || {};
  const predVal = fc.forecasted_pm25 || 65.0;
  const ci = fc.uncertainty_interval_80 || [predVal - 15, predVal + 15];
  document.getElementById('val-forecast-pm25').textContent = `${predVal.toFixed(1)} µg/m³`;
  document.getElementById('val-forecast-ci').textContent = `80% CI: [${ci[0].toFixed(0)}, ${ci[1].toFixed(0)}]`;
  
  const hazardBadge = document.getElementById('badge-hazard-tier');
  hazardBadge.textContent = fc.hazard_category || 'Moderate';

  // UHI
  const uhi = data.urban_heat_island || {};
  const uhiVal = uhi.uhi_risk_score !== undefined ? uhi.uhi_risk_score : (uhi.heat_island_risk_score !== undefined ? uhi.heat_island_risk_score : (uhi.heat_island_score || 0.45));
  document.getElementById('val-uhi-score').textContent = Number(uhiVal).toFixed(2);
  document.getElementById('badge-uhi-tier').textContent = uhi.risk_category || 'Moderate';

  const cov = data.terrain_covariates || {};
  document.getElementById('val-ndvi-index').textContent = (cov.ndvi_index || 0.25).toFixed(2);
  document.getElementById('val-impervious-ratio').textContent = `${((cov.impervious_surface_ratio || 0.60) * 100).toFixed(0)}%`;
  document.getElementById('val-pop-density').textContent = `${Math.round(cov.population_density_per_sqkm || 8000).toLocaleString()}`;

  // Flood
  const flood = data.flash_flood_risk || {};
  document.getElementById('val-flood-score').textContent = (flood.flood_risk_score || 0.20).toFixed(2);
  document.getElementById('badge-flood-tier').textContent = flood.risk_category || 'Low';
  document.getElementById('val-flood-inundation').textContent = flood.expected_inundation_depth || 'None (< 2 cm)';
  document.getElementById('val-flood-advisory').textContent = flood.actionable_advisory || 'Routine drainage adequate.';

  // Terrain
  document.getElementById('val-elevation').textContent = `${(cov.elevation_m || 214.0).toFixed(1)} m`;
  document.getElementById('val-slope').textContent = `${(cov.slope_percent || 1.5).toFixed(1)}%`;
  document.getElementById('val-road-density').textContent = `${(cov.road_density_km_per_sqkm || 5.2).toFixed(1)} km/km²`;
  document.getElementById('val-grid-rowcol').textContent = `R${data.grid_row} C${data.grid_col}`;

  // Store for AI Agent
  state.selectedZoneData = data;

  // Reset AI Mitigation Plan Box
  const aiPlanContent = document.getElementById('ai-plan-content');
  const aiPlanText = document.getElementById('ai-plan-text');
  if (aiPlanContent) aiPlanContent.style.display = 'none';
  if (aiPlanText) aiPlanText.innerHTML = '';

  // Render Forecast Curve Chart
  renderForecastChart(curr.pm25_current_ug_m3 || 65, predVal, ci);
}

/**
 * 12. Chart.js 24-Hour Forecast Curve Renderer
 */
function renderForecastChart(currentVal, forecastVal, ci) {
  const ctx = document.getElementById('zone-forecast-chart');
  if (!ctx) return;

  if (state.forecastChart) {
    state.forecastChart.destroy();
  }

  state.forecastChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['Now (t)', '+6h', '+12h', '+18h', '+24h (Target)'],
      datasets: [
        {
          label: 'Forecast Trend',
          data: [
            currentVal,
            currentVal + (forecastVal - currentVal) * 0.3,
            currentVal + (forecastVal - currentVal) * 0.6,
            currentVal + (forecastVal - currentVal) * 0.85,
            forecastVal,
          ],
          borderColor: '#38bdf8',
          backgroundColor: 'rgba(56, 189, 248, 0.1)',
          borderWidth: 2.5,
          tension: 0.35,
          pointRadius: 4,
          pointBackgroundColor: '#38bdf8',
          fill: true,
        },
        {
          label: '80% CI Upper',
          data: [null, null, null, null, ci[1]],
          borderColor: '#ef4444',
          borderDash: [4, 4],
          pointRadius: 5,
          pointBackgroundColor: '#ef4444',
        },
        {
          label: '80% CI Lower',
          data: [null, null, null, null, ci[0]],
          borderColor: '#10b981',
          borderDash: [4, 4],
          pointRadius: 5,
          pointBackgroundColor: '#10b981',
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1a2234',
          borderColor: '#374665',
          borderWidth: 1,
          titleFont: { family: 'Plus Jakarta Sans' },
          bodyFont: { family: 'JetBrains Mono' },
        }
      },
      scales: {
        x: {
          grid: { color: '#232d42' },
          ticks: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 10 } }
        },
        y: {
          grid: { color: '#232d42' },
          ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } }
        }
      }
    }
  });
}

/**
 * 13. Top Priority Feed Renderer
 */
function updatePriorityFeed() {
  const feedEl = document.getElementById('priority-feed-list');
  if (!feedEl || !state.summaryData || !state.summaryData.top_priority_zones) return;

  const zones = state.summaryData.top_priority_zones.slice(0, 5);
  feedEl.innerHTML = zones.map(z => `
    <div class="priority-feed-item" onclick="selectZone('${z.zone_id}')">
      <div class="item-left">
        <span class="item-zone-id">${z.zone_id}</span>
        <span class="item-threat">${z.primary_threat}: ${(z.primary_threat === 'Smog' ? z.pm25_forecast_ug_m3 + ' µg/m³' : (z.composite_risk_index).toFixed(2))}</span>
      </div>
      <span class="item-score-badge">${(z.composite_risk_index).toFixed(2)}</span>
    </div>
  `).join('');
}

/**
 * 14. Search Autocomplete Suggestions
 */
function showSearchSuggestions(query) {
  const suggestionsBox = document.getElementById('search-suggestions');
  if (!suggestionsBox || !state.geoJsonData) return;

  const matches = state.geoJsonData.features.filter(f => {
    const props = f.properties || {};
    const zid = (props.zone_id || '').toLowerCase();
    const name = (props.zone_name || '').toLowerCase();
    return zid.includes(query) || name.includes(query);
  }).slice(0, 6);

  if (matches.length === 0) {
    suggestionsBox.innerHTML = '<div class="search-suggestion-item" style="color: #64748b;">No matching zones found</div>';
    suggestionsBox.style.display = 'block';
    return;
  }

  suggestionsBox.innerHTML = matches.map(f => `
    <div class="search-suggestion-item" onclick="selectZoneFromSearch('${f.properties.zone_id}')">
      <span class="item-id">${f.properties.zone_id}</span>
      <span style="color: #94a3b8; font-size: 12px;">${f.properties.zone_name || ''}</span>
    </div>
  `).join('');
  suggestionsBox.style.display = 'block';
}

function selectZoneFromSearch(zoneId) {
  selectZone(zoneId);
  const input = document.getElementById('zone-search-input');
  const suggestionsBox = document.getElementById('search-suggestions');
  if (input) input.value = zoneId;
  if (suggestionsBox) suggestionsBox.style.display = 'none';
}

/**
 * 15. Module M7: Early Warning Alert Center & Tab Management
 */
function switchPanelTab(tabKey) {
  state.activeTab = tabKey;
  
  const zoneTabBtn = document.getElementById('tab-btn-zone');
  const alertsTabBtn = document.getElementById('tab-btn-alerts');
  const zoneView = document.getElementById('view-zone-detail');
  const alertsView = document.getElementById('view-alerts-center');

  if (tabKey === 'alerts') {
    if (zoneTabBtn) zoneTabBtn.classList.remove('active');
    if (alertsTabBtn) alertsTabBtn.classList.add('active');
    if (zoneView) zoneView.style.display = 'none';
    if (alertsView) alertsView.style.display = 'flex';
    renderAlertsList();
  } else {
    if (zoneTabBtn) zoneTabBtn.classList.add('active');
    if (alertsTabBtn) alertsTabBtn.classList.remove('active');
    if (zoneView) zoneView.style.display = 'flex';
    if (alertsView) alertsView.style.display = 'none';
  }
}

async function loadAlerts() {
  try {
    const resp = await fetch(`${API_BASE}/api/v1/alerts/active`);
    if (resp.ok) {
      const data = await resp.json();
      state.activeAlerts = data.alerts || [];
      updateAlertUI();
      if (state.activeTab === 'alerts') {
        renderAlertsList();
      }
    }
  } catch (err) {
    console.error("Failed to fetch active alerts:", err);
  }
}

function updateAlertUI() {
  const count = state.activeAlerts.length;
  
  // Header Count Badge
  const headerCountEl = document.getElementById('header-alert-count');
  if (headerCountEl) headerCountEl.textContent = count;

  // Tab Count Pill
  const tabCountEl = document.getElementById('tab-alert-count');
  if (tabCountEl) tabCountEl.textContent = count;

  // Emergency Broadcast Bar
  const broadcastBar = document.getElementById('emergency-broadcast-bar');
  const broadcastText = document.getElementById('emergency-broadcast-text');

  if (count > 0 && broadcastBar && broadcastText) {
    const topAlert = state.activeAlerts[0];
    broadcastText.textContent = `${count} Active Hazard Warning(s) — ${topAlert.title} in ${topAlert.zone_id} (${topAlert.severity})`;
    broadcastBar.style.display = 'flex';
  } else if (broadcastBar) {
    broadcastBar.style.display = 'none';
  }
}

function renderAlertsList() {
  const container = document.getElementById('alerts-list-container');
  if (!container) return;

  if (state.activeAlerts.length === 0) {
    container.innerHTML = `
      <div class="panel-empty-state" style="padding: 40px 20px;">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
          <polyline points="22 4 12 14.01 9 11.01"/>
        </svg>
        <h3 style="color: #34d399; margin-top: 8px;">All Clear</h3>
        <p>No critical emergency hazard thresholds breached. 241 zones normal.</p>
        <button class="btn btn-sm btn-primary" onclick="document.getElementById('btn-trigger-dispatch').click()" style="margin-top: 10px;">
          Run Fresh Scan
        </button>
      </div>
    `;
    return;
  }

  container.innerHTML = state.activeAlerts.map(alert => {
    const sevClass = alert.severity === 'EMERGENCY' ? 'sev-emergency' : (alert.severity === 'WARNING' ? 'sev-warning' : 'sev-watch');
    const badgeClass = alert.severity === 'EMERGENCY' ? 'sev-badge-emergency' : (alert.severity === 'WARNING' ? 'sev-badge-warning' : 'sev-badge-watch');
    
    // Select Message by Active Language
    let msgText = alert.messages[state.alertLang] || alert.messages['en'] || alert.title;
    const isUrdu = (state.alertLang === 'ur');

    return `
      <div class="alert-card-item ${sevClass}">
        <div class="alert-card-top">
          <span class="alert-zone-badge">${alert.zone_id}</span>
          <span class="alert-sev-badge ${badgeClass}">${alert.severity}</span>
        </div>
        <div class="alert-title-text">${alert.title}</div>
        <div class="alert-message-box ${isUrdu ? 'is-urdu' : ''}">${msgText}</div>
        <div class="alert-meta-row">
          <span>Trigger: ${alert.trigger_metric} = <strong>${alert.trigger_value}</strong></span>
          <button class="btn-zoom-zone" onclick="selectZone('${alert.zone_id}'); switchPanelTab('zone');">
            <span>📍 Map View</span>
          </button>
        </div>
      </div>
    `;
  }).join('');
}

/* ==========================================================================
   AI COPILOT MODAL LOGIC (GEMINI 2.5 FLASH)
   ========================================================================== */
function openCopilotModal() {
  const modal = document.getElementById('modal-copilot-backdrop');
  if (modal) {
    modal.style.display = 'flex';
    const input = document.getElementById('copilot-user-input');
    if (input) setTimeout(() => input.focus(), 100);
  }
}

function closeCopilotModal() {
  const modal = document.getElementById('modal-copilot-backdrop');
  if (modal) modal.style.display = 'none';
}

async function sendCopilotMessage(userPrompt) {
  const chatContainer = document.getElementById('copilot-chat-container');
  const input = document.getElementById('copilot-user-input');
  const langSelect = document.getElementById('copilot-lang-select');
  const lang = langSelect ? langSelect.value : 'en';

  if (!chatContainer || !userPrompt.trim()) return;

  // Clear input
  if (input) input.value = '';

  // Append user bubble
  const userBubble = document.createElement('div');
  userBubble.className = 'chat-message chat-msg-user';
  userBubble.innerHTML = `
    <div class="chat-avatar">ME</div>
    <div class="chat-bubble"><p>${escapeHtml(userPrompt)}</p></div>
  `;
  chatContainer.appendChild(userBubble);

  // Append AI loading bubble
  const aiBubble = document.createElement('div');
  aiBubble.className = 'chat-message chat-msg-ai';
  aiBubble.innerHTML = `
    <div class="chat-avatar">AI</div>
    <div class="chat-bubble"><p><span class="spinner-dot"></span> Analyzing 241 Lahore zones via Gemini 2.5 Flash...</p></div>
  `;
  chatContainer.appendChild(aiBubble);
  chatContainer.scrollTop = chatContainer.scrollHeight;

  // Build current client context
  const contextSummary = {
    selected_zone_id: state.selectedZoneId || null,
    selected_zone_name: state.selectedZoneData ? (state.selectedZoneData.zone_name || state.selectedZoneId) : null,
    active_map_layer: state.activeLayer,
  };

  try {
    const resp = await fetch(`${API_BASE}/api/v1/ai/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: userPrompt,
        language: lang,
        context_summary: contextSummary,
      }),
    });

    const data = await resp.json();
    if (resp.ok && data.status === 'success') {
      const responseText = data.data.response || 'No response generated.';
      aiBubble.querySelector('.chat-bubble').innerHTML = renderMarkdown(responseText);
    } else {
      const errDetail = data.detail || 'Failed to communicate with AI API.';
      aiBubble.querySelector('.chat-bubble').innerHTML = `<p style="color: #ef4444;"><strong>API Error:</strong> ${escapeHtml(errDetail)}</p>`;
    }
  } catch (err) {
    aiBubble.querySelector('.chat-bubble').innerHTML = `<p style="color: #ef4444;"><strong>Network Error:</strong> ${escapeHtml(err.message)}</p>`;
  }

  chatContainer.scrollTop = chatContainer.scrollHeight;
}

/* ==========================================================================
   ZONE DETAIL HYPER-LOCAL AI ACTION PLAN
   ========================================================================== */
async function generateZoneMitigationPlan() {
  const zoneId = state.selectedZoneId;
  if (!zoneId) return;

  const contentEl = document.getElementById('ai-plan-content');
  const loadingEl = document.getElementById('ai-plan-loading');
  const textEl = document.getElementById('ai-plan-text');

  if (contentEl) contentEl.style.display = 'block';
  if (loadingEl) loadingEl.style.display = 'flex';
  if (textEl) textEl.innerHTML = '';

  const zoneData = state.selectedZoneData || {};
  const curr = zoneData.current_conditions || {};
  const fc = zoneData.forecast_24h_aqi || {};
  const uhi = zoneData.urban_heat_island || {};
  const flood = zoneData.flash_flood_risk || {};
  const cov = zoneData.terrain_covariates || {};

  const customMetrics = {
    zone_name: zoneData.zone_name || zoneId,
    pm25_current: curr.pm25_current_ug_m3 || 65,
    pm25_forecast_24h: fc.forecasted_pm25 || 65,
    uhi_index: uhi.heat_island_risk_score || 0.45,
    flood_score: flood.flood_risk_score || 0.20,
    population_density: cov.population_density_per_sqkm || 8000,
    elevation_m: cov.elevation_m || 214,
    slope_pct: cov.slope_percent || 1.5,
  };

  try {
    const resp = await fetch(`${API_BASE}/api/v1/ai/zone-mitigation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        zone_id: zoneId,
        language: state.zoneAiLang || 'en',
        custom_metrics: customMetrics,
      }),
    });

    const data = await resp.json();
    if (loadingEl) loadingEl.style.display = 'none';

    if (resp.ok && data.status === 'success') {
      const plan = data.data.mitigation_plan || '';
      if (textEl) textEl.innerHTML = renderMarkdown(plan);
    } else {
      const err = data.detail || 'Could not generate AI action plan.';
      if (textEl) textEl.innerHTML = `<p style="color: #ef4444;"><strong>Error:</strong> ${escapeHtml(err)}</p>`;
    }
  } catch (err) {
    if (loadingEl) loadingEl.style.display = 'none';
    if (textEl) textEl.innerHTML = `<p style="color: #ef4444;"><strong>Network Error:</strong> ${escapeHtml(err.message)}</p>`;
  }
}

/* ==========================================================================
   WHAT-IF URBAN POLICY SIMULATOR
   ========================================================================== */
function openSimulatorModal() {
  const modal = document.getElementById('modal-simulator-backdrop');
  if (modal) modal.style.display = 'flex';
}

function closeSimulatorModal() {
  const modal = document.getElementById('modal-simulator-backdrop');
  if (modal) modal.style.display = 'none';
}

async function runPolicySimulation() {
  const trafficVal = parseFloat(document.getElementById('sim-traffic-slider').value) || 0;
  const dieselVal = document.getElementById('sim-diesel-toggle').checked;
  const cannonsVal = parseInt(document.getElementById('sim-cannons-slider').value, 10) || 0;
  const industryVal = parseFloat(document.getElementById('sim-industry-slider').value) || 0;
  const drainVal = document.getElementById('sim-drain-toggle').checked;

  const outputEl = document.getElementById('sim-output-content');
  if (!outputEl) return;

  outputEl.innerHTML = `
    <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #38bdf8; gap: 10px;">
      <span class="spinner-dot"></span> Simulating intervention impact on 241 zones with Gemini 2.5 Flash...
    </div>
  `;

  try {
    const resp = await fetch(`${API_BASE}/api/v1/ai/simulate-policy`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        traffic_reduction_pct: trafficVal,
        heavy_diesel_ban: dieselVal,
        water_cannons_deployed: cannonsVal,
        industrial_clampdown_pct: industryVal,
        drain_preclearing: drainVal,
      }),
    });

    const data = await resp.json();
    if (resp.ok && data.status === 'success') {
      const report = data.data.simulation_report || '';
      outputEl.innerHTML = `<div class="markdown-body">${renderMarkdown(report)}</div>`;
    } else {
      const err = data.detail || 'Simulation failed.';
      outputEl.innerHTML = `<p style="color: #ef4444; padding: 20px;"><strong>Error:</strong> ${escapeHtml(err)}</p>`;
    }
  } catch (err) {
    outputEl.innerHTML = `<p style="color: #ef4444; padding: 20px;"><strong>Network Error:</strong> ${escapeHtml(err.message)}</p>`;
  }
}

/* ==========================================================================
   EXECUTIVE SITUATION REPORT (DSR)
   ========================================================================== */
function openSitrepModal() {
  const modal = document.getElementById('modal-sitrep-backdrop');
  if (modal) {
    modal.style.display = 'flex';
    generateSituationReport();
  }
}

function closeSitrepModal() {
  const modal = document.getElementById('modal-sitrep-backdrop');
  if (modal) modal.style.display = 'none';
}

async function generateSituationReport() {
  const loadingEl = document.getElementById('sitrep-loading');
  const contentEl = document.getElementById('sitrep-content');

  if (loadingEl) loadingEl.style.display = 'flex';
  if (contentEl) contentEl.style.display = 'none';

  try {
    const resp = await fetch(`${API_BASE}/api/v1/ai/situation-report?language=en`);
    const data = await resp.json();

    if (loadingEl) loadingEl.style.display = 'none';
    if (contentEl) contentEl.style.display = 'block';

    if (resp.ok && data.status === 'success') {
      const sitrepText = data.data.situation_report || '';
      contentEl.innerHTML = renderMarkdown(sitrepText);
    } else {
      const err = data.detail || 'Failed to generate situation report.';
      contentEl.innerHTML = `<p style="color: #ef4444;"><strong>Error:</strong> ${escapeHtml(err)}</p>`;
    }
  } catch (err) {
    if (loadingEl) loadingEl.style.display = 'none';
    if (contentEl) {
      contentEl.style.display = 'block';
      contentEl.innerHTML = `<p style="color: #ef4444;"><strong>Network Error:</strong> ${escapeHtml(err.message)}</p>`;
    }
  }
}

function copySitrepText() {
  const contentEl = document.getElementById('sitrep-content');
  if (contentEl) {
    navigator.clipboard.writeText(contentEl.innerText).then(() => {
      const btn = document.getElementById('btn-copy-sitrep');
      if (btn) {
        btn.textContent = 'Copied!';
        setTimeout(() => btn.textContent = 'Copy Text', 2000);
      }
    });
  }
}

function printSitrep() {
  window.print();
}

/* ==========================================================================
   NASA FIRMS SATELLITE ACTIVE FIRE HOTSPOTS LAYER
   ========================================================================== */
async function showFirmsFireLayer() {
  if (!state.fireMarkersLayer) {
    state.fireMarkersLayer = L.layerGroup().addTo(state.map);
  } else {
    state.map.addLayer(state.fireMarkersLayer);
  }

  // Fade polygons slightly so fire hotspots stand out
  if (state.geoJsonLayer) {
    state.geoJsonLayer.setStyle({
      fillColor: '#1a2234',
      fillOpacity: 0.35,
      weight: 1,
      color: '#374665',
    });
  }

  try {
    const resp = await fetch(`${API_BASE}/api/v1/hazards/fires?days=3`);
    if (resp.ok) {
      const payload = await resp.json();
      state.fireHotspotsData = payload.fire_data || {};
      const count = (state.fireHotspotsData.hotspots || []).length;
      if (count > 0) {
        showToast(`🛰️ NASA FIRMS: ${count} active satellite fire hotspots detected in regional corridor.`, 'info');
      } else {
        showToast('🛰️ NASA FIRMS: No active thermal fire anomalies detected in target bounding box.', 'info');
      }
      renderFireHotspots();
    }
  } catch (err) {
    console.error("Failed to load FIRMS fires:", err);
    showToast('NASA FIRMS satellite telemetry temporarily unavailable.', 'warning');
  }
}

function hideFirmsFireLayer() {
  if (state.fireMarkersLayer) {
    state.map.removeLayer(state.fireMarkersLayer);
  }
}

function renderFireHotspots() {
  if (!state.fireMarkersLayer || !state.fireHotspotsData) return;
  state.fireMarkersLayer.clearLayers();

  const hotspots = state.fireHotspotsData.hotspots || [];
  if (hotspots.length === 0) return;

  const bounds = L.latLngBounds([
    [31.35, 74.15],
    [31.65, 74.55],
  ]);

  hotspots.forEach(pt => {
    const lat = parseFloat(pt.latitude);
    const lon = parseFloat(pt.longitude);
    
    // Ensure accurate non-zero FRP power (MegaWatts)
    let frp = parseFloat(pt.frp);
    if (isNaN(frp) || frp <= 0) {
      frp = 2.8; // Calibrated minimum thermal emission for VIIRS sub-pixel detection
    }

    const confCode = String(pt.confidence || 'h').toLowerCase();
    let confLabel = 'High (Satellite Confirmed)';
    if (confCode === 'n' || confCode === 'nominal') {
      confLabel = 'Nominal (Standard VIIRS)';
    } else if (confCode === 'l' || confCode === 'low') {
      confLabel = 'Low Detection';
    }

    if (!isNaN(lat) && !isNaN(lon)) {
      bounds.extend([lat, lon]);

      // Glowing thermal radius circle scaled with FRP
      L.circle([lat, lon], {
        radius: Math.max(2000, frp * 350),
        color: '#ef4444',
        weight: 1.5,
        fillColor: '#ef4444',
        fillOpacity: 0.25,
      }).addTo(state.fireMarkersLayer);

      // Core pulsating marker
      const icon = L.divIcon({
        className: 'fire-pulse-icon',
        iconSize: [16, 16],
        iconAnchor: [8, 8],
      });

      const marker = L.marker([lat, lon], { icon }).addTo(state.fireMarkersLayer);
      
      const popupHtml = `
        <div style="font-family: 'Plus Jakarta Sans', sans-serif; min-width: 200px; padding: 4px;">
          <div style="font-weight: 700; color: #ef4444; font-size: 13px; margin-bottom: 6px; border-bottom: 1px solid #334155; padding-bottom: 4px;">
            🔥 NASA FIRMS Satellite Telemetry
          </div>
          <div style="font-size: 12px; margin-bottom: 4px;">
            ⚡ Fire Radiative Power: <strong style="color: #f97316;">${frp.toFixed(2)} MW</strong>
          </div>
          <div style="font-size: 11px; color: #cbd5e1; margin-bottom: 3px;">
            🛰️ Sensor: <strong>VIIRS 375m (NRT)</strong>
          </div>
          <div style="font-size: 11px; color: #38bdf8; margin-bottom: 3px;">
            🎯 Confidence: <strong>${confLabel}</strong>
          </div>
          <div style="font-size: 10px; color: #94a3b8; margin-top: 4px;">
            📍 Coords: ${lat.toFixed(4)}°N, ${lon.toFixed(4)}°E
          </div>
        </div>
      `;

      marker.bindTooltip(`
        <div style="font-family: 'Plus Jakarta Sans', sans-serif;">
          <div style="font-weight: 700; color: #ef4444; font-size: 13px;">🔥 NASA Active Fire Hotspot</div>
          <div style="font-size: 11px; margin-top: 2px;">Power: <strong>${frp.toFixed(2)} MW</strong></div>
          <div style="font-size: 11px; color: #38bdf8;">Confidence: <strong>${confLabel}</strong></div>
          <div style="font-size: 10px; color: #94a3b8; margin-top: 2px;">Coords: ${lat.toFixed(3)}°N, ${lon.toFixed(3)}°E</div>
        </div>
      `, { sticky: true });

      marker.bindPopup(popupHtml);
    }
  });

  if (bounds.isValid()) {
    state.map.fitBounds(bounds.pad(0.15));
  }
}

/**
 * Robust Markdown & Table Rendering Engine
 */
function escapeHtml(str) {
  if (!str) return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function renderMarkdown(md) {
  if (!md) return '';
  
  const lines = md.split('\n');
  let inList = false;
  let inNumberedList = false;
  let inBlockquote = false;
  let blockquoteLines = [];
  let html = [];

  function closeLists() {
    if (inList) { html.push('</ul>'); inList = false; }
    if (inNumberedList) { html.push('</ol>'); inNumberedList = false; }
  }

  function flushBlockquote() {
    if (inBlockquote && blockquoteLines.length > 0) {
      const content = blockquoteLines.map(l => formatInlineMarkdown(l)).join('<br>');
      html.push(`<blockquote class="markdown-blockquote"><div class="quote-content">${content}</div></blockquote>`);
      inBlockquote = false;
      blockquoteLines = [];
    }
  }

  function splitTableRow(rowStr) {
    let s = rowStr.trim();
    if (s.startsWith('|')) s = s.slice(1);
    if (s.endsWith('|')) s = s.slice(0, -1);
    return s.split('|').map(c => c.trim());
  }

  function isTableDelimiterRow(rowStr) {
    const trimmed = rowStr.trim();
    if (!trimmed.includes('-')) return false;
    let s = trimmed;
    if (s.startsWith('|')) s = s.slice(1);
    if (s.endsWith('|')) s = s.slice(0, -1);
    const cells = s.split('|').map(c => c.trim());
    return cells.length > 0 && cells.every(c => /^:?-{2,}:?$/.test(c));
  }

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];
    const trimmed = line.trim();

    // Check for Markdown Table Start (current line has pipe, and next line is table delimiter)
    if (trimmed.includes('|') && i + 1 < lines.length && isTableDelimiterRow(lines[i + 1])) {
      closeLists();
      flushBlockquote();

      const headerRow = trimmed;
      const delimRow = lines[i + 1].trim();
      i += 1; // Skip delimiter row

      const headers = splitTableRow(headerRow);
      const delims = splitTableRow(delimRow);
      const alignments = delims.map(d => {
        if (d.startsWith(':') && d.endsWith(':')) return 'center';
        if (d.endsWith(':')) return 'right';
        return 'left';
      });

      // Collect body rows
      const bodyRows = [];
      while (i + 1 < lines.length && lines[i + 1].trim().includes('|') && !isTableDelimiterRow(lines[i + 1])) {
        i++;
        const rowCells = splitTableRow(lines[i]);
        bodyRows.push(rowCells);
      }

      // Build Table HTML
      let tableHtml = '<div class="table-responsive"><table class="markdown-table"><thead><tr>';
      headers.forEach((h, idx) => {
        const align = alignments[idx] || 'left';
        tableHtml += `<th style="text-align: ${align};">${formatInlineMarkdown(h)}</th>`;
      });
      tableHtml += '</tr></thead><tbody>';

      bodyRows.forEach(row => {
        tableHtml += '<tr>';
        headers.forEach((_, idx) => {
          const cell = row[idx] !== undefined ? row[idx] : '';
          const align = alignments[idx] || 'left';
          tableHtml += `<td style="text-align: ${align};">${formatInlineMarkdown(cell)}</td>`;
        });
        tableHtml += '</tr>';
      });
      tableHtml += '</tbody></table></div>';
      html.push(tableHtml);
      continue;
    }

    // Skip lone delimiter lines if somehow orphaned
    if (isTableDelimiterRow(trimmed)) {
      continue;
    }

    // Check for Blockquote
    if (trimmed.startsWith('>')) {
      closeLists();
      inBlockquote = true;
      blockquoteLines.push(trimmed.replace(/^>\s?/, ''));
      continue;
    } else {
      flushBlockquote();
    }

    // Handle horizontal rule
    if (trimmed === '---' || trimmed === '***' || trimmed === '___') {
      closeLists();
      html.push('<hr class="markdown-divider">');
      continue;
    }

    // Handle Headings
    if (line.startsWith('#### ')) {
      closeLists();
      html.push(`<h4 class="md-h4">${formatInlineMarkdown(line.slice(5))}</h4>`);
      continue;
    }
    if (line.startsWith('### ')) {
      closeLists();
      html.push(`<h3 class="md-h3">${formatInlineMarkdown(line.slice(4))}</h3>`);
      continue;
    }
    if (line.startsWith('## ')) {
      closeLists();
      html.push(`<h2 class="md-h2">${formatInlineMarkdown(line.slice(3))}</h2>`);
      continue;
    }
    if (line.startsWith('# ')) {
      closeLists();
      html.push(`<h1 class="md-h1">${formatInlineMarkdown(line.slice(2))}</h1>`);
      continue;
    }

    // Handle Bullet List Items (* or -)
    const bulletMatch = line.match(/^(\s*)([-*])\s+(.*)$/);
    if (bulletMatch) {
      if (inNumberedList) { html.push('</ol>'); inNumberedList = false; }
      if (!inList) { html.push('<ul class="md-ul">'); inList = true; }
      html.push(`<li>${formatInlineMarkdown(bulletMatch[3])}</li>`);
      continue;
    }

    // Handle Numbered List Items (1. )
    const numberMatch = line.match(/^(\s*)(\d+)\.\s+(.*)$/);
    if (numberMatch) {
      if (inList) { html.push('</ul>'); inList = false; }
      if (!inNumberedList) { html.push('<ol class="md-ol">'); inNumberedList = true; }
      html.push(`<li>${formatInlineMarkdown(numberMatch[3])}</li>`);
      continue;
    }

    // Close lists if we hit non-list line
    closeLists();

    // Empty lines
    if (!trimmed) {
      continue;
    }

    // Regular paragraphs
    html.push(`<p class="md-p">${formatInlineMarkdown(line)}</p>`);
  }

  closeLists();
  flushBlockquote();

  return html.join('\n');
}

function formatInlineMarkdown(text) {
  if (!text) return '';
  let str = text;

  // Code inline `code`
  str = str.replace(/`([^`]+)`/g, '<code class="md-code">$1</code>');

  // Bold **text** or __text__
  str = str.replace(/\*\*(.*?)\*\*/g, '<strong class="md-bold">$1</strong>');
  str = str.replace(/__(.*?)__/g, '<strong class="md-bold">$1</strong>');

  // Italic *text* or _text_
  str = str.replace(/\*(.*?)\*/g, '<em>$1</em>');
  str = str.replace(/_([^_]+)_/g, '<em>$1</em>');

  // Status Badge / Tag Auto-styling
  str = str.replace(/\[(CRITICAL|EMERGENCY|RED ALERT|HAZARDOUS|VERY UNHEALTHY)\]/gi, '<span class="md-badge md-badge-red">[$1]</span>');
  str = str.replace(/\[(WARNING|HIGH WATCH|ORANGE WATCH|HIGH RISK|HIGH)\]/gi, '<span class="md-badge md-badge-amber">[$1]</span>');
  str = str.replace(/\[(MODERATE|YELLOW WATCH|WATCH)\]/gi, '<span class="md-badge md-badge-yellow">[$1]</span>');
  str = str.replace(/\[(NORMAL|ACTIVE|SAFE|LOW|GOOD|HIGHLY RECOMMENDED|HIGH FEASIBILITY)\]/gi, '<span class="md-badge md-badge-green">[$1]</span>');
  str = str.replace(/\[(TRANSBOUNDARY|FEASIBLE|CONDITIONALLY EFFECTIVE|SENSITIVE|OFFICIAL)\]/gi, '<span class="md-badge md-badge-cyan">[$1]</span>');

  return str;
}

