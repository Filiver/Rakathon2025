import * as THREE from "three";
import { STATIC_OBJECT_CONFIG } from "../config.js"; // Assuming this exists
import {
  createGeometry,
  createMesh,
  createPoints,
  createWireframe,
} from "./utils.js";

export class StaticObject {
  constructor(objectData, app) {
    this.app = app;
    this.name = objectData.name;
    this.isSingleton = objectData.isSingleton || false; // Default to false if not specified
    this.batchSize = app.batchManager.batchSize;
    this.representations = { mesh: [], wireframe: [], points: [] }; // For visualization modes
    this.batchGroups = []; // Store batch groups

    // Validate input data
    if (this.isSingleton) {
      if (!objectData.shape || !objectData.shape.type) {
        throw new Error(
          "Singleton static object requires a 'shape' definition."
        );
      }
      this.type = objectData.shape.type;
    } else {
      if (
        !Array.isArray(objectData.shapes) ||
        objectData.shapes.length !== this.batchSize
      ) {
        throw new Error(
          `Batched static object requires a 'shapes' array of length ${this.batchSize}.`
        );
      }
      this.type = objectData.shapes[0].type; // Assume consistent types for simplicity
    }

    // Initialize and create visual objects
    this.initializeGroup();
    this.createBatchGroups(objectData);
  }

  /** Initialize the main Three.js group */
  initializeGroup() {
    this.group = new THREE.Group();
    this.group.name = this.name;
  }

  /** Create batch groups with visual representations */
  createBatchGroups(objectData) {
    // Create each batch
    for (let i = 0; i < this.batchSize; i++) {
      const batchGroup = new THREE.Group();
      batchGroup.name = `${this.name}_batch_${i}`;
      this.group.add(batchGroup);
      this.batchGroups.push(batchGroup);

      const shape = this.isSingleton ? objectData.shape : objectData.shapes[i];
      const geometryConfig = STATIC_OBJECT_CONFIG.geometry || {};
      const meshConfig = {
        ...STATIC_OBJECT_CONFIG.mesh,
        color: shape.color || STATIC_OBJECT_CONFIG.color || 0xffffff,
      };
      const pointsConfig = {
        ...STATIC_OBJECT_CONFIG.points,
        color: shape.color || STATIC_OBJECT_CONFIG.color || 0xffffff,
      };
      if (shape.type === "pointcloud") {
        const points = createPoints(shape.points, pointsConfig);
        if (points) {
          points.visible = this.app.uiState.bodyVisualizationMode === "points";
          batchGroup.add(points);
          this.representations["points"].push(points);
        }
      } else {
        const geometry = createGeometry(shape, geometryConfig);
        const mesh = createMesh(geometry, meshConfig);
        mesh.visible = this.app.uiState.bodyVisualizationMode === "mesh";
        batchGroup.add(mesh);
        this.representations["mesh"].push(mesh);
        const wireframe = createWireframe(
          geometry,
          STATIC_OBJECT_CONFIG.wireframe
        );
        wireframe.visible =
          this.app.uiState.bodyVisualizationMode === "wireframe";
        batchGroup.add(wireframe);
        this.representations["wireframe"].push(wireframe);
      }

      // Set fixed position with batch offset
      const offset = this.app.batchManager.getBatchOffset(i);
      batchGroup.position.set(offset.x, offset.y, offset.z);
    }
  }

  /** Update visualization mode (mesh, wireframe, points) */
  updateVisualizationMode(mode) {
    this.representations.mesh.forEach(
      (mesh) => (mesh.visible = mode === "mesh")
    );
    this.representations.wireframe.forEach(
      (wireframe) => (wireframe.visible = mode === "wireframe")
    );
    this.representations.points.forEach(
      (points) => (points.visible = mode === "points")
    );
  }

  /** Return the Three.js group for scene integration */
  getObject3D() {
    return this.group;
  }

  /** Clean up resources */
  dispose() {
    if (this.group) {
      if (
        this.app.scene &&
        typeof this.app.scene.removeObject3D === "function"
      ) {
        this.app.scene.removeObject3D(this.group);
      } else if (this.group.parent) {
        this.group.parent.remove(this.group);
      }

      this.group = null;
      this.representations = { mesh: [], wireframe: [], points: [] };
      this.batchGroups = [];
    }
  }
}
