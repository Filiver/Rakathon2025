import * as THREE from "three";
import { TERRAIN_CONFIG } from "../config.js";
import { colorMapOptions, evaluate_cmap } from "../../lib/js-colormaps.js";

export class Terrain {
  constructor(terrainData, app) {
    this.app = app; // Store reference to app for accessing batch manager
    this.bounds = terrainData.bounds;
    this.dimensions = terrainData.dimensions;
    this.group = null;
    this.#createVisualRepresentations(
      terrainData.heightData,
      terrainData.normals
    );
  }

  #createMaterials() {
    // Create materials for the surface and wireframe
    const surfaceMaterial = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      flatShading: TERRAIN_CONFIG.flatShading || false,
      shininess: TERRAIN_CONFIG.shininess || 10,
    });

    const wireframeMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      wireframe: true,
      opacity: 0.2,
      transparent: true,
    });

    return { surfaceMaterial, wireframeMaterial };
  }

  #createVisualRepresentations(heightData, normals) {
    console.debug(
      `Creating terrain geometry: ${this.dimensions.extentX}x${this.dimensions.extentY} m, resolution: ${this.dimensions.shapeX}x${this.dimensions.shapeY}`
    );
    const { surfaceMaterial, wireframeMaterial } = this.#createMaterials();
    // Create geometry for each batch
    this.group = new THREE.Group();
    for (let i = 0; i < this.app.batchManager.batchSize; i++) {
      const batchGroup = new THREE.Group();
      batchGroup.name = `batch${i}`;
      const surfaceGeometry = this.#createSurfaceGeometryFromHeightData(
        heightData[i]
      );
      const surfaceMesh = new THREE.Mesh(surfaceGeometry, surfaceMaterial);
      surfaceMesh.name = "surface";
      surfaceMesh.receiveShadow = true;
      surfaceMesh.castShadow = true;
      batchGroup.add(surfaceMesh);
      const wireframeMesh = new THREE.Mesh(surfaceGeometry, wireframeMaterial);
      wireframeMesh.name = "wireframe";
      batchGroup.add(wireframeMesh);
      const surfaceNormals = this.#createNormalVectors(
        heightData[i],
        normals[i]
      );
      surfaceNormals.name = "normals";
      batchGroup.add(surfaceNormals);
      for (const [key, value] of Object.entries(
        this.app.uiState.terrainVisualizationModes
      )) {
        batchGroup.getObjectByName(key).visible = value;
      }
      // translate by the batch offset
      const batch_offset = this.app.batchManager.getBatchOffset(i);
      batchGroup.position.set(batch_offset.x, batch_offset.y, batch_offset.z);
      this.group.add(batchGroup);
    }
  }

  calculateNormalizedHeight(height) {
    var normalizedHeight =
      (height - this.bounds.minZ) / (this.bounds.maxZ - this.bounds.minZ);
    normalizedHeight = Number.isFinite(normalizedHeight) ? normalizedHeight : 0;
    normalizedHeight = Math.max(0, Math.min(1, normalizedHeight));
    return normalizedHeight;
  }

  /**
   *
   * @param {array} heightData - Height data for the terrain, a flattened array
   * @returns
   */
  #createSurfaceGeometryFromHeightData(heightData) {
    const { extentX, extentY, shapeX, shapeY } = this.dimensions;
    const { minX, minY, maxX, maxY } = this.bounds;
    // Create a plane geometry with the right number of segments
    const geometry = new THREE.PlaneGeometry(
      extentX,
      extentY,
      shapeX - 1,
      shapeY - 1
    );
    // Center the geometry based on bounds
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    geometry.translate(centerX, centerY, 0);
    // Get attributes for direct manipulation
    const position = geometry.attributes.position;
    // Create color buffer
    const colorAttribute = new THREE.BufferAttribute(
      new Float32Array(position.count * 3),
      3
    );
    // Apply height data to geometry
    // NOTE: THREE.js PlaneGeometry vertices are arranged in rows from bottom to top (Y increases)
    const callableColormap = this.getCallableFromColorMapName(
      this.app.uiState.terrainColorMap
    );
    for (let i = 0; i < position.count; i++) {
      // Convert vertex index to grid coordinates
      const col = i % shapeX;
      const invertedRow = Math.floor(i / shapeY);
      const row = shapeY - invertedRow - 1; // Invert row index
      // Calculate index in the flattened height data array
      const dataIndex = row * shapeX + col;
      // Set Z coordinate (height)
      position.setZ(i, heightData[dataIndex]);
      // Calculate color based on height
      const normalizedHeight = this.calculateNormalizedHeight(
        heightData[dataIndex]
      );
      const color = callableColormap(normalizedHeight);
      colorAttribute.setXYZ(i, color.r, color.g, color.b);
    }
    // Add colors to the geometry
    geometry.setAttribute("color", colorAttribute);
    // Make sure changes are applied
    position.needsUpdate = true;
    return geometry;
  }

  getCallableFromColorMapName(cmapName) {
    let reversed = false;
    // Check if this is a reversed colormap request
    if (cmapName.endsWith("_r")) {
      cmapName = cmapName.substring(0, cmapName.length - 2);
      reversed = true;
    }
    if (colorMapOptions.includes(cmapName))
      return (value) => {
        const [r, g, b] = evaluate_cmap(value, cmapName, reversed);
        return new THREE.Color(r / 255, g / 255, b / 255);
      };
    console.log(
      `Colormap ${cmapName} not found in colorMapOptions. Using default colormap instead.`
    );
    // Fallback
    switch (cmapName) {
      case "grayscale":
        return (value) => new THREE.Color(value, value, value);
      case "heatmap":
        // Simple heatmap: blue->cyan->green->yellow->red
        return (value) => {
          if (value < 0.25) {
            return new THREE.Color(0, value * 4, 1);
          } else if (value < 0.5) {
            return new THREE.Color(0, 1, 1 - (value - 0.25) * 4);
          } else if (value < 0.75) {
            return new THREE.Color((value - 0.5) * 4, 1, 0);
          } else {
            return new THREE.Color(1, 1 - (value - 0.75) * 4, 0);
          }
        };
      case "terrain":
        // Terrain color map - blues for low areas, greens for middle, browns/whites for high
        return (value) => {
          if (value < 0.2) {
            return new THREE.Color(0.0, 0.2, 0.5 + value); // Deep to shallow water
          } else if (value < 0.4) {
            const t = (value - 0.2) * 5; // 0-1 within this range
            return new THREE.Color(0.2 * t, 0.5 + 0.2 * t, 0.7 - 0.2 * t); // Shore transition
          } else if (value < 0.75) {
            const t = (value - 0.4) / 0.35; // 0-1 within this range
            return new THREE.Color(0.2 + 0.3 * t, 0.7 - 0.2 * t, 0.5 - 0.4 * t); // Green to brown
          } else {
            const t = (value - 0.75) * 4; // 0-1 within this range
            return new THREE.Color(0.5 + 0.5 * t, 0.5 + 0.5 * t, 0.1 + 0.9 * t); // Brown to white (snow)
          }
        };
      default:
        // Default blue to red gradient
        return (value) => new THREE.Color(value, 0.2, 1 - value);
    }
  }

  /**
   *
   * @param {array} heightData - Height data for the terrain, a flattened array
   * @param {array} normals - Normal data for the terrain, an array of 3D vectors
   * @returns {THREE.Group} - Group containing normal vectors
   */
  #createNormalVectors(heightData, normals) {
    const { extentX, extentY, shapeX, shapeY } = this.dimensions;
    const { minX, minY } = this.bounds;

    const normalVectors = new THREE.Group();
    normalVectors.visible =
      this.app.uiState.terrainVisualizationModes.normals || false;

    // Create a helper arrow for each normal
    const normalLength = TERRAIN_CONFIG.normalLength || 0.5;
    const skipFactor = Math.max(
      1,
      Math.floor(shapeX / TERRAIN_CONFIG.skipNormalCells)
    ); // Adaptive skip factor based on resolution

    // Sample normals at regular intervals
    for (let row = 0; row < shapeY; row += skipFactor) {
      for (let col = 0; col < shapeX; col += skipFactor) {
        const dataIndex = row * shapeX + col;

        if (dataIndex < heightData.length) {
          // Calculate real-world coordinates
          const x = minX + col * (extentX / (shapeX - 1));
          const y = minY + row * (extentY / (shapeY - 1));
          const z = heightData[dataIndex];

          // Get normal data
          const [nx, ny, nz] = normals[dataIndex];

          const origin = new THREE.Vector3(x, y, z);
          const direction = new THREE.Vector3(nx, ny, nz);

          const arrowHelper = new THREE.ArrowHelper(
            direction.normalize(),
            origin,
            normalLength,
            0xff0000
          );

          normalVectors.add(arrowHelper);
        }
      }
    }
    return normalVectors;
  }

  // Toggle methods for visualizations - now updated to support batched terrains
  toggleVisualization(type, visible) {
    // Toggle in the original terrain
    for (const batchGroup of this.group.children) {
      const object = batchGroup.getObjectByName(type);
      if (object) {
        object.visible = visible;
      }
    }
  }

  #updateSurfaceColor(geometry, callableColormap) {
    if (!geometry) return;
    const position = geometry.attributes.position;
    const colorAttribute = geometry.attributes.color;
    if (!position || !colorAttribute) return;
    // Update all colors based on the new colormap
    for (let i = 0; i < position.count; i++) {
      // Update color based on height
      const normalizedHeight = this.calculateNormalizedHeight(position.getZ(i));
      const color = callableColormap(normalizedHeight);
      colorAttribute.setXYZ(i, color.r, color.g, color.b);
    }
    // Update the buffer
    colorAttribute.needsUpdate = true;
  }

  // Update terrain colors with current colormap
  setColorMap(colormapName) {
    // Update the main terrain surface
    const callableColormap = this.getCallableFromColorMapName(colormapName);
    for (const batchGroup of this.group.children) {
      const surfaceMesh = batchGroup.getObjectByName("surface");
      const geometry = surfaceMesh.geometry;
      this.#updateSurfaceColor(geometry, callableColormap);
    }
  }

  // Get THREE.js group containing all visualizations
  getObject3D() {
    return this.group;
  }

  // Clean up resources when terrain is no longer needed
  dispose() {
    for (const batchGroup of this.group.children) {
      batchGroup.traverse((child) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
      });
      if (this.app && this.app.scene) {
        this.app.scene.removeObject3D(batchGroup);
      }
    }
    this.group = null;
  }
}
