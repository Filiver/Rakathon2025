import * as THREE from "three";
import { BATCH_PALETTE_GENERATION_CONFIG } from "../config.js";
import { generateDivergingPalette } from "../objects/utils.js";

export class BatchManager {
  constructor(app, modelData) {
    this.app = app;
    this.batchSize = 1; // Default to single batch
    this.currentlyActiveBatch = 0; // Default to the first batch
    this.collapsedMode = false; // Flag for collapsed mode, TODO implement this
    this.focusedMode = false; // Flag for focused mode, TODO implement this

    // Batch offset configuration
    this.spacing = 0.5; // Spacing between batches in meters
    this.batchOffsets = []; // Array of {x, y, z} offsets for each batch
    this.batchPalette = []; // Array of colors for each batch
    this._initialize(modelData);
  }

  _initialize(modelData) {
    if (!modelData) return;

    // Set batch count from model data
    if (modelData.batchSize !== undefined) {
      this.batchSize = Math.max(1, parseInt(modelData.batchSize));
      console.log(`Initializing with ${this.batchSize} simulation batches`);
      this.app.batchSize = this.batchSize;
    }
    const sideLength = Math.ceil(Math.sqrt(this.batchSize));
    const { extentX, extentY } = modelData.terrain.dimensions;

    // Initialize batch offsets
    this.batchOffsets = [];
    for (let i = 0; i < this.batchSize; i++) {
      const rowIdx = Math.floor(i / sideLength);
      const colIdx = i % sideLength;
      this.batchOffsets.push({
        x: colIdx * (extentX + this.spacing),
        y: rowIdx * (extentY + this.spacing),
        z: 0,
      });
    }
    console.debug("Batch offsets initialized:", this.batchOffsets);

    this.batchPalette = generateDivergingPalette(
      BATCH_PALETTE_GENERATION_CONFIG.colors,
      this.batchSize + 1,
      BATCH_PALETTE_GENERATION_CONFIG.correctLightness
    );
    console.debug("Batch palette initialized:", this.batchPalette);
  }

  getbatchSize() {
    return this.batchSize;
  }

  getColorForBatch(batchIndex) {
    if (batchIndex < 0 || batchIndex >= this.batchSize) {
      return new THREE.Color(0x000000); // Default to black for invalid index
    }
    return this.batchPalette[batchIndex];
  }

  getRowColFromBatchIndex(batchIndex) {
    if (batchIndex < 0 || batchIndex >= this.batchSize) {
      return { row: -1, col: -1 };
    }
    const sideLength = Math.ceil(Math.sqrt(this.batchSize));
    const rowIdx = Math.floor(batchIndex / sideLength);
    const colIdx = batchIndex % sideLength;
    return { row: rowIdx, col: colIdx };
  }

  getBatchIndexFromRowCol(rowIdx, colIdx) {
    const sideLength = Math.ceil(Math.sqrt(this.batchSize));
    if (
      rowIdx < 0 ||
      colIdx < 0 ||
      rowIdx >= sideLength ||
      colIdx >= sideLength
    ) {
      return -1;
    }
    const batchIndex = rowIdx * sideLength + colIdx;
    if (batchIndex < 0 || batchIndex >= this.batchSize) {
      return -1;
    }
    return batchIndex;
  }

  getOffsetByRowCol(rowIdx, colIdx) {
    const sideLength = Math.ceil(Math.sqrt(this.batchSize));
    if (
      rowIdx < 0 ||
      colIdx < 0 ||
      rowIdx >= sideLength ||
      colIdx >= sideLength
    ) {
      throw new Error("Invalid row or column index");
    }
    const batchIndex = rowIdx * sideLength + colIdx;
    return this.getBatchOffset(batchIndex);
  }

  // Get offset for a specific batch
  getBatchOffset(batchIndex) {
    if (batchIndex >= 0 && batchIndex < this.batchOffsets.length) {
      return this.batchOffsets[batchIndex];
    }
    return { x: 0, y: 0, z: 0 };
  }

  changeFocusOnBatchByRowCol(row, col) {
    const { camera, controls } = this.app.scene;
    const { x, y, z } = this.getOffsetByRowCol(row, col);
    const newCameraTarget = new THREE.Vector3(x, y, z);
    const oldCameraTarget = controls.target.clone();
    controls.target = newCameraTarget;
    camera.position.add(newCameraTarget.clone().sub(oldCameraTarget));
  }

  changeFocusOnBatchByIndex(batchIndex) {
    const { row, col } = this.getRowColFromBatchIndex(batchIndex);
    this.changeFocusOnBatchByRowCol(row, col);
  }

  setActiveBatch(batchIndex) {
    if (batchIndex >= 0 && batchIndex < this.batchSize) {
      this.currentlyActiveBatch = batchIndex;
      this.changeFocusOnBatchByIndex(batchIndex);
    } else {
      console.warn("Invalid batch index:", batchIndex);
    }
    this.app.bodyStateWindow.setSelectedBatch(batchIndex);
    if (this.app.scalarPlotter) {
      this.app.scalarPlotter.setFocusedBatch(batchIndex);
    }
  }

  setActiveBatchByRowCol(row, col) {
    const batchIndex = this.getBatchIndexFromRowCol(row, col);
    if (batchIndex === -1) {
      return;
    }
    this.setActiveBatch(batchIndex);
  }
}
