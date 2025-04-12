import * as THREE from "three";
import { BODY_CONFIG, BODY_VECTOR_CONFIG } from "../config.js";
import {
  createGeometry,
  createMesh,
  createPoints,
  createWireframe,
  createContactPoints,
  createArrow,
} from "./utils.js";

export class Body {
  constructor(bodyData, app) {
    this.app = app;
    this.name = bodyData.name;
    this.batchSize = app.batchManager.batchSize;

    // Mandatory attributes
    this.positions = Array(this.batchSize)
      .fill()
      .map(() => new THREE.Vector3());
    this.quaternions = Array(this.batchSize)
      .fill()
      .map(() => new THREE.Quaternion());
    this.rotations = Array(this.batchSize)
      .fill()
      .map(() => new THREE.Euler()); // For debugging/display

    // Optional attributes management
    this.availableAttributes = new Set(bodyData.availableAttributes || []);
    this.attributeStorage = new Map();
    this.attributeUpdaters = new Map();
    this.hasContacts = this.availableAttributes.has("contacts");

    // Initialize storage and updaters for optional attributes
    if (bodyData.availableAttributes) {
      bodyData.availableAttributes.forEach((attr) => {
        if (attr === "contacts") {
          this.attributeStorage.set(
            attr,
            Array(this.batchSize)
              .fill()
              .map(() => [])
          );
          this.attributeUpdaters.set(attr, (data, batchIndex) =>
            this.updateContactPointsVisibility(data, batchIndex)
          );
        } else {
          // Vector attributes: velocity, angularVelocity, force, torque
          this.attributeStorage.set(
            attr,
            Array(this.batchSize)
              .fill()
              .map(() => new THREE.Vector3())
          );
          this.attributeUpdaters.set(attr, (data, batchIndex) =>
            this.updateBodyVector(attr, data, batchIndex)
          );
        }
      });
    }
    // Scene setup
    this.group = new THREE.Group();
    this.group.name = this.name;
    this.batchGroups = [];
    this.bodyVectors = [];
    this.contactPoints = [];
    this.contactPointSizes = [];
    this.representations = { mesh: [], wireframe: [], points: [] };
    this.createBatchGroups(bodyData);
  }

  createBatchGroups(bodyData) {
    for (let i = 0; i < this.batchSize; i++) {
      const batchGroup = new THREE.Group();
      batchGroup.name = `${this.name}_batch_${i}`;
      this.group.add(batchGroup);
      this.batchGroups.push(batchGroup);

      this.createVisualRepresentations(batchGroup, bodyData);
      this.initializeBodyVectors(batchGroup, BODY_VECTOR_CONFIG, i);

      if (
        this.hasContacts &&
        (bodyData.shape.type === "pointcloud" || bodyData.shape.type === "mesh")
      ) {
        const points =
          bodyData.shape.type === "pointcloud"
            ? bodyData.shape.points
            : bodyData.shape.vertices;
        this.initializeContactPoints(batchGroup, points, i);
      }

      const axes = new THREE.AxesHelper(1);
      axes.visible = this.app.uiState.axesVisible;
      batchGroup.add(axes);
    }
  }

  updateAttribute(attr, data, batchIndex) {
    if (!this.attributeStorage.has(attr) || batchIndex >= this.batchSize)
      return;
    const storage = this.attributeStorage.get(attr);
    if (attr === "contacts") {
      storage[batchIndex] = Array.isArray(data) ? data : [];
    } else {
      storage[batchIndex].fromArray(data.length >= 3 ? data : [0, 0, 0]);
    }
    const updater = this.attributeUpdaters.get(attr);
    if (updater) updater(storage[batchIndex], batchIndex);
  }

  updateState(bodyState) {
    // Update position (mandatory)
    if (bodyState.position) {
      if (Array.isArray(bodyState.position[0])) {
        for (
          let i = 0;
          i < Math.min(this.batchSize, bodyState.position.length);
          i++
        ) {
          this.setPosition(bodyState.position[i], i);
        }
      } else {
        this.setPosition(bodyState.position, 0);
      }
    }

    // Update orientation (mandatory)
    if (bodyState.orientation) {
      if (Array.isArray(bodyState.orientation[0])) {
        for (
          let i = 0;
          i < Math.min(this.batchSize, bodyState.orientation.length);
          i++
        ) {
          this.setOrientation(bodyState.orientation[i], i);
        }
      } else {
        this.setOrientation(bodyState.orientation, 0);
      }
    }

    // Update optional attributes
    for (const attr of this.attributeStorage.keys()) {
      if (bodyState[attr]) {
        const data = bodyState[attr];
        if (Array.isArray(data[0])) {
          for (let i = 0; i < Math.min(this.batchSize, data.length); i++) {
            this.updateAttribute(attr, data[i], i);
          }
        } else {
          this.updateAttribute(attr, data, 0);
        }
      }
    }
  }

  setPosition(positionData, batchIndex = 0) {
    if (
      !positionData ||
      positionData.length < 3 ||
      batchIndex >= this.batchSize
    )
      return;
    const [x, y, z] = positionData;
    this.positions[batchIndex].set(x, y, z);
    const offset = this.app.batchManager
      ? this.app.batchManager.getBatchOffset(batchIndex)
      : { x: 0, y: 0, z: 0 };
    this.batchGroups[batchIndex].position.set(
      x + offset.x,
      y + offset.y,
      z + offset.z
    );
  }

  setOrientation(orientationData, batchIndex = 0) {
    if (
      !orientationData ||
      orientationData.length < 4 ||
      batchIndex >= this.batchSize
    )
      return;
    const [qw, qx, qy, qz] = orientationData;
    this.quaternions[batchIndex].set(qx, qy, qz, qw);
    this.rotations[batchIndex].setFromQuaternion(this.quaternions[batchIndex]);
    this.batchGroups[batchIndex].quaternion.copy(this.quaternions[batchIndex]);
  }

  createVisualRepresentations(batchGroup, bodyData) {
    const shape = bodyData.shape;

    if (shape.type === "pointcloud") {
      const points = createPoints(shape.points, BODY_CONFIG.points);
      if (points) {
        points.visible = this.app.uiState.bodyVisualizationMode === "points";
        batchGroup.add(points);
        this.representations["points"].push(points);
      }
    } else {
      const geometry = createGeometry(shape, BODY_CONFIG.geometry);
      if (geometry) {
        const mesh = createMesh(geometry, BODY_CONFIG.mesh);
        mesh.visible = this.app.uiState.bodyVisualizationMode === "mesh";
        batchGroup.add(mesh);
        this.representations["mesh"].push(mesh);

        const wireframe = createWireframe(geometry, BODY_CONFIG.wireframe);
        wireframe.visible =
          this.app.uiState.bodyVisualizationMode === "wireframe";
        batchGroup.add(wireframe);
        this.representations["wireframe"].push(wireframe);
      }
    }
  }

  initializeBodyVectors(batchGroup, vectorConfigs, batchIndex) {
    if (!this.bodyVectors[batchIndex]) {
      this.bodyVectors[batchIndex] = new Map();
    }
    for (const attr of this.availableAttributes) {
      if (attr !== "contacts" && vectorConfigs[attr]) {
        const config = vectorConfigs[attr];
        const vector = createArrow(
          new THREE.Vector3(),
          new THREE.Vector3(0, 1, 0),
          config
        );
        vector.visible = this.app.uiState.attributeVisible[attr] || false;
        vector.userData = { scale: config.scale };
        batchGroup.add(vector);
        this.bodyVectors[batchIndex].set(attr, vector);
      }
    }
  }

  initializeContactPoints(batchGroup, points, batchIndex) {
    if (!points?.length) return;
    const contactPoints = createContactPoints(
      points,
      BODY_CONFIG.contactPoints
    );
    if (!contactPoints) return;

    const pointCount = points.length;
    this.contactPointSizes[batchIndex] = new Float32Array(pointCount).fill(0);
    contactPoints.geometry.setAttribute(
      "size",
      new THREE.Float32BufferAttribute(this.contactPointSizes[batchIndex], 1)
    );
    contactPoints.visible = this.app.uiState.attributeVisible.contacts || false;
    batchGroup.add(contactPoints);
    this.contactPoints[batchIndex] = contactPoints;
  }

  updateBodyVector(type, vector, batchIndex) {
    const arrow = this.bodyVectors[batchIndex]?.get(type);
    if (!arrow) return;
    const scale = arrow.userData.scale || 1.0;
    const length = vector.length() * scale;
    if (length < 1e-6) {
      arrow.setLength(0);
      return;
    }
    const normalizedVector = vector.clone().normalize();
    arrow.setDirection(normalizedVector);
    const headLength = Math.min(length * 0.2, 0.5);
    const headWidth = Math.min(length * 0.1, 0.25);
    arrow.setLength(length, headLength, headWidth);
  }

  updateContactPointsVisibility(contactIndices, batchIndex) {
    if (!this.contactPoints[batchIndex] || !this.contactPointSizes[batchIndex])
      return;
    this.contactPointSizes[batchIndex].fill(0);
    if (Array.isArray(contactIndices)) {
      contactIndices.forEach((index) => {
        if (index >= 0 && index < this.contactPointSizes[batchIndex].length) {
          this.contactPointSizes[batchIndex][index] =
            BODY_CONFIG.contactPoints.size;
        }
      });
    }
    const sizeAttribute =
      this.contactPoints[batchIndex].geometry.getAttribute("size");
    sizeAttribute.array = this.contactPointSizes[batchIndex];
    sizeAttribute.needsUpdate = true;
  }

  updateVisualizationMode(mode) {
    for (const [type, objects] of Object.entries(this.representations)) {
      objects.forEach((obj) => (obj.visible = type === mode));
    }
  }

  toggleContactPoints(visible) {
    this.contactPoints.forEach((cp) => cp && (cp.visible = visible));
  }

  toggleAxes(visible) {
    this.batchGroups.forEach((group) => {
      const axes = group.children.find(
        (child) => child instanceof THREE.AxesHelper
      );
      if (axes) axes.visible = visible;
    });
  }

  toggleBodyVector(type, visible) {
    if (!this.availableAttributes.has(type)) return;
    this.bodyVectors.forEach((map) => {
      const vector = map.get(type);
      if (vector) vector.visible = visible;
    });
  }

  getObject3D() {
    return this.group;
  }

  getAvailableAttributes() {
    return this.availableAttributes;
  }

  dispose() {
    if (this.group && this.group.parent) {
      this.group.parent.remove(this.group);
      this.group.traverse((child) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          Array.isArray(child.material)
            ? child.material.forEach((m) => m.dispose())
            : child.material.dispose();
        }
      });
    }
    this.group = null;
    this.batchGroups = [];
    this.bodyVectors = [];
    this.contactPoints = [];
    this.contactPointSizes = [];
  }
}
