import { FREQ_CONFIG, SCALAR_PLOTTER_CONFIG } from "../config.js";

export class ScalarPlotter {
  constructor(app, scalarNames) {
    this.app = app;
    this.scalarNames = scalarNames || [];
    if (this.scalarNames.length === 0) {
      console.warn("No scalar names provided.");
      return;
    }
    this.isExpanded = false;
    this.activeScalar = this.scalarNames[0];
    this.currentEndIndex = 0;
    this.currentFocusedBatch = 0;
    this.plotElements = {};
    this.tabElements = {};
    this.charts = new Map();
    this.scalarSeries = new Map();
    this.scalarBounds = new Map();
    this.fullDataPoints = new Map();
    this.times = [];
    this.indices = [];
    this.seriesRenderCallback = null;
    this.opacityRenderCallback = null;
    this.minRenderDelay = 1000 / FREQ_CONFIG.scalarPlotter;
    this.lastRenderTime = Number.NEGATIVE_INFINITY;
    this.renderTimeout = null; // To manage delayed rendering

    this._injectStyles();
    this._setupHTML();
    this._setupEventListeners();
  }

  _injectStyles() {
    const styleId = `scalar-styles`;
    if (document.getElementById(styleId)) return;
    const containerWidthPercentage = 40;
    const contentMaxHeightPercentage = 25;
    const chartHeightPercentage = 15;
    const css = `
        .scalar-dropdown-container {
            width: ${containerWidthPercentage}%;
            position: absolute;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            border-radius: 5px;
            font-family: Arial, sans-serif;
            font-size: 12px;
            z-index: 1000;
            /* overflow-y: auto; */ /* Let content handle overflow */
        }

        /* Header */
        .scalar-header {
            display: flex;
            justify-content: space-between;
            background-color: transparent;
            align-items: center;
            width: 100%;
            padding: 7px 10px;
            color: white;
            cursor: pointer;
            border-radius: 5px 5px 0 0; /* Adjust rounding */
            transition: background-color 0.2s ease;
            box-sizing: border-box;
        }
        .scalar-header:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
        .scalar-header-title {
            font-weight: bold;
            font-size: 1.1em;
        }
        .scalar-header-icon {
            font-size: 1.2em;
            transition: transform 0.3s ease;
        }
        .scalar-header-icon.expanded {
            transform: rotate(180deg);
        }

        /* Content area */
        .scalar-content {
            border-radius: 0 0 5px 5px;
            display: none;
            box-sizing: border-box;
            overflow: hidden;
        }

        .scalar-content.visible {
            display: block;
             max-height: ${contentMaxHeightPercentage}vh;
             overflow-y: auto;
        }

        /* Tab bar */
        .scalar-tab-bar {
            display: flex;
            width: 100%;
            flex-wrap: wrap;
            padding: 0;
            position: sticky;
            top: 0;
            z-index: 1;
        }

        /* Tabs */
        .scalar-tab {
            flex: 1;
            text-align: center;
            padding: 5px 10px; /* Adjusted padding */
            margin-right: 2px; /* Space between tabs */
            cursor: pointer;
            background-color: rgba(0, 0, 0, 1);
            border-radius: 2px 2px 0 0;
            font-size: 1.0em;
            color: white;
            transition: background-color 0.2s ease, color 0.2s ease;
            position: relative;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-bottom: 1px solid white;
            margin-bottom: -1px;
            font-weight: bold;
            white-space: nowrap; /* Prevent wrapping */
            overflow: hidden; /* Hide overflow */
            text-overflow: ellipsis; /* Add ellipsis if text too long */
        }
        .scalar-tab:first-child {
            margin-left: 0;
            border-left: none;
        }
        .scalar-tab:last-child {
            margin-right: 0;
            border-right: none;
        }
        .scalar-tab.active {
            border: 1px solid white;
            border-bottom: 1px solid rgba(0, 0, 0, 1);
        }
        .scalar-tab.active:hover {
            background-color: rgba(0, 0, 0, 1);
        }
        .scalar-tab.active:first-child {
            border-left: none;
        }
        .scalar-tab.active:last-child {
            border-right: none;
        }

        /* Plot area */
        .scalar-plot-area {
            width: 100%;
            height: ${chartHeightPercentage}vh;
          }

        .canvasjs-plot-div {
                     border-top: 1px solid white; /* Re-add border here */

            width: 100%;
            height: 99%;
            display: none;
        }
        .canvasjs-plot-div.visible {
            display: block;
        }
    `;
    const styleElement = document.createElement("style");
    styleElement.id = styleId;
    styleElement.textContent = css;
    document.head.appendChild(styleElement);
  }

  _setupHTML() {
    this.container = document.createElement("div");
    this.container.className = "scalar-dropdown-container";
    this.header = document.createElement("div");
    this.header.className = "scalar-header";
    const title = document.createElement("span");
    title.className = "scalar-header-title";
    title.textContent = "Scalars";
    this.icon = document.createElement("span");
    this.icon.className = "scalar-header-icon";
    this.icon.innerHTML = "↓";
    this.header.appendChild(title);
    this.header.appendChild(this.icon);

    this.content = document.createElement("div");
    this.content.className = "scalar-content";

    this.tabBar = document.createElement("div");
    this.tabBar.className = "scalar-tab-bar";

    this.plotArea = document.createElement("div");
    this.plotArea.className = "scalar-plot-area";

    this.scalarNames.forEach((name, index) => {
      const tabButton = document.createElement("button");
      tabButton.className = "scalar-tab";
      tabButton.textContent = name;
      tabButton.dataset.scalarName = name;
      if (index === 0) tabButton.classList.add("active");
      this.tabBar.appendChild(tabButton);
      this.tabElements[name] = tabButton;

      const plotDiv = document.createElement("div");
      plotDiv.id = `plot-${name}`;
      plotDiv.className = "canvasjs-plot-div";
      plotDiv.dataset.scalarName = name;
      if (index === 0) plotDiv.classList.add("visible");
      this.plotArea.appendChild(plotDiv);
      this.plotElements[name] = plotDiv;
    });

    this.content.appendChild(this.tabBar);
    this.content.appendChild(this.plotArea);
    this.container.appendChild(this.header);
    this.container.appendChild(this.content);
    document.body.appendChild(this.container);
  }

  _setupEventListeners() {
    this.header.addEventListener("click", () => this._toggleDropdown());
    this.tabBar.addEventListener("click", (event) => {
      const target = event.target;
      if (target.classList.contains("scalar-tab")) {
        const scalarName = target.dataset.scalarName;
        if (scalarName && scalarName !== this.activeScalar) {
          this._switchTab(scalarName);
        }
      }
      event.target.blur();
    });
  }

  _toggleDropdown() {
    this.isExpanded = !this.isExpanded;
    this.content.classList.toggle("visible", this.isExpanded);
    this.icon.classList.toggle("expanded", this.isExpanded);

    if (this.isExpanded && this.activeScalar) {
      this.setEndIndex(this.currentEndIndex, true);
      this.setFocusedBatch(this.currentFocusedBatch, true);
    }
  }

  _switchTab(newScalarName) {
    if (
      !this.scalarNames.includes(newScalarName) ||
      newScalarName === this.activeScalar
    ) {
      return;
    }

    const oldTab = this.tabElements[this.activeScalar];
    const oldPlot = this.plotElements[this.activeScalar];
    if (oldTab) oldTab.classList.remove("active");
    if (oldPlot) oldPlot.classList.remove("visible");

    const newTab = this.tabElements[newScalarName];
    const newPlot = this.plotElements[newScalarName];
    if (newTab) newTab.classList.add("active");
    if (newPlot) newPlot.classList.add("visible");

    this.activeScalar = newScalarName;
    this.setEndIndex(this.currentEndIndex, true);
    this.setFocusedBatch(this.currentFocusedBatch, true);
  }

  initFromStates(states) {
    const batchSize = this.app.batchManager.batchSize;
    this.indices = [...Array(batchSize).keys()];
    for (const scalarName of this.scalarNames) {
      this.scalarBounds.set(scalarName, [Number.MAX_VALUE, Number.MIN_VALUE]);
      this.scalarSeries.set(
        scalarName,
        new Array(batchSize).fill().map(() => [])
      );
    }
    for (const state of states) {
      for (const scalarName of this.scalarNames) {
        const scalarValues = state[scalarName];
        if (scalarValues === undefined) {
          throw new Error(`Scalar "${scalarName}" not found in state.`);
        }
        var [min, max] = this.scalarBounds.get(scalarName);
        for (let i = 0; i < batchSize; i++) {
          const scalarValue = scalarValues[i];
          if (scalarValue === undefined) {
            throw new Error(
              `Scalar "${scalarName}" not found in state at index ${i}.`
            );
          }
          this.scalarSeries.get(scalarName)[i].push(scalarValue);
          min = Math.min(min, scalarValue);
          max = Math.max(max, scalarValue);
        }
        this.scalarBounds.set(scalarName, [min, max]);
      }
      this.times.push(state.time);
    }

    for (const scalarName of this.scalarNames) {
      console.log(
        "Scalar",
        scalarName,
        " has bounds",
        this.scalarBounds.get(scalarName)
      );
    }

    this.fullDataPoints = new Map();
    for (const scalarName of this.scalarNames) {
      const series = this.scalarSeries.get(scalarName);
      const dataPointsPerBatch = series.map((batchSeries) =>
        batchSeries.map((value, timeIndex) => ({
          x: this.times[timeIndex],
          y: value,
        }))
      );
      this.fullDataPoints.set(scalarName, dataPointsPerBatch);
    }
    this._initializePlots();
    if (this.isExpanded) {
      this.setEndIndex(this.currentEndIndex, true);
      this.setFocusedBatch(this.currentFocusedBatch, true);
    }
  }

  getChartInterval(min, max) {
    const diff = max - min;
    if (diff === 0)
      return Math.max(
        Math.abs(max) / SCALAR_PLOTTER_CONFIG.stepsPerYAxis,
        1e-3
      );
    return diff / SCALAR_PLOTTER_CONFIG.stepsPerYAxis;
  }

  _initializePlots() {
    const limOffset = 1e-2;
    this.scalarNames.forEach((name) => {
      const plotDiv = this.plotElements[name];
      var [min, max] = this.scalarBounds.get(name);
      min = min - limOffset;
      max = max + limOffset;
      const chart = new CanvasJS.Chart(plotDiv, {
        backgroundColor: "rgba(0, 0, 0, 1)",
        axisX: {
          lineColor: "rgba(255, 255, 255, 0.00)",
          labelFontColor: "transparent",
          tickLength: 0,
          minimum: this.times[0],
          maximum: this.times[this.times.length - 1],
          zoomEnabled: false,
          margin: -10,
        },
        axisY: {
          gridColor: "rgb(53, 53, 53)",
          lineColor: "rgb(73, 73, 73)",
          labelFontColor: "white",
          minimum: min,
          maximum: max,
          zoomEnabled: false,
          labelFontFamily: "Arial",
          interval: this.getChartInterval(min, max),
        },
        toolTip: {
          fontFamily: "Arial",
          contentFormatter: function (e) {
            var series = e.entries[0].dataSeries;
            var batch = series.name.split(" ").at(-1);
            var time = e.entries[0].dataPoint.x.toFixed(3);
            var value = e.entries[0].dataPoint.y.toFixed(3);
            var color = series.color;
            return `<div style="color: ${color};">
                        Batch: ${batch}<br>
                        Time: ${time}<br>
                        Value: ${value}
                    </div>`;
          },
        },
        data: [],
      });
      for (let i = 0; i < this.app.batchManager.batchSize; i++) {
        chart.options.data.push({
          type: "line",
          markerSize: 0,
          lineThickness: 1,
          color: this.app.batchManager.getColorForBatch(i),
          name: `${name} ${i}`,
          dataPoints: [],
          click: () => {
            this.app.batchManager.setActiveBatch(i);
          },
          lineColor: this.app.batchManager.getColorForBatch(i),
          lineDashType: "solid",
        });
      }
      this.charts.set(name, chart);
    });
  }

  setEndIndex(newEndIndex, force = false) {
    if (newEndIndex < 0 || newEndIndex >= this.times.length) {
      console.warn(
        "Invalid end index. Must be within the range of time values."
      );
      return;
    }
    if (this.currentEndIndex === newEndIndex && !force) {
      return;
    }
    this.currentEndIndex = newEndIndex;
    const activeChart = this.charts.get(this.activeScalar);
    if (!activeChart) {
      console.warn(`No chart found for scalar "${this.activeScalar}".`);
      return;
    }
    const fullDataPoints = this.fullDataPoints.get(this.activeScalar);
    if (!fullDataPoints) {
      console.warn(`No data points found for scalar "${this.activeScalar}".`);
      return;
    }

    this.seriesRenderCallback = () => {
      for (let i = 0; i < this.app.batchManager.batchSize; i++) {
        activeChart.options.data[i].dataPoints = fullDataPoints[i].slice(
          0,
          newEndIndex + 1
        );
      }
    };
  }

  setFocusedBatch(batchIndex, force = false) {
    if (batchIndex < 0 || batchIndex >= this.app.batchManager.batchSize) {
      console.warn("Invalid batch index.");
      return;
    }
    if (this.currentFocusedBatch === batchIndex && !force) {
      return;
    }
    this.currentFocusedBatch = batchIndex;

    const activeChart = this.charts.get(this.activeScalar);
    if (!activeChart) {
      console.warn(`No chart found for scalar "${this.activeScalar}".`);
      return;
    }
    this.opacityRenderCallback = () => {
      for (let i = 0; i < this.app.batchManager.batchSize; i++) {
        const opacity =
          i === batchIndex ? 1 : SCALAR_PLOTTER_CONFIG.inactiveBatchOpacity;
        const baseColor = this.app.batchManager.getColorForBatch(i);
        const rgbaColor = this._hexToRgba(baseColor, opacity);
        activeChart.options.data[i].lineColor = rgbaColor;
      }
    };
  }

  _hexToRgba(hex, opacity) {
    hex = hex.replace("#", "");
    const r = parseInt(hex.substring(0, 2), 16);
    const g = parseInt(hex.substring(2, 4), 16);
    const b = parseInt(hex.substring(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${opacity})`;
  }

  _renderChart() {
    const activeChart = this.charts.get(this.activeScalar);
    if (!activeChart) return;

    const plotDiv = this.plotElements[this.activeScalar];
    if (!this.isExpanded || !plotDiv.classList.contains("visible")) {
      return;
    }

    var needRender = false;
    if (this.seriesRenderCallback) {
      this.seriesRenderCallback();
      this.seriesRenderCallback = null;
      needRender = true;
    }
    if (this.opacityRenderCallback) {
      this.opacityRenderCallback();
      this.opacityRenderCallback = null;
      needRender = true;
    }

    if (!needRender) return;
    activeChart.render();
  }

  animate(now) {
    if (now - this.lastRenderTime < this.minRenderDelay) return;
    this._renderChart();
    this.lastRenderTime = now;
  }

  dispose() {
    this.container.remove();
    this.charts.clear();
    this.scalarSeries.clear();
    this.scalarBounds.clear();
    this.fullDataPoints.clear();
  }
}
