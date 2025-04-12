import * as THREE from "three";
import chroma from "chroma";

// Default configurations
const DEFAULT_GEOMETRY_CONFIG = {
  box: {
    widthSegments: 1,
    heightSegments: 1,
    depthSegments: 1,
  },
  sphere: {
    widthSegments: 32,
    heightSegments: 16,
  },
  cylinder: {
    radialSegments: 32,
    heightSegments: 1,
  },
};

const DEFAULT_POINTS_CONFIG = {
  size: 1,
  opacity: 1,
  alphaTest: 0.5,
  transparent: true,
  sizeAttenuation: true,
};

const DEFAULT_WIREFRAME_CONFIG = {
  linewidth: 1,
  color: 0x000000,
};

const DEFAULT_MESH_CONFIG = {
  opacity: 1,
  color: 0xffffff,
  roughness: 0.5,
  metalness: 0.5,
  envMapIntensity: 1,
  transparent: false,
};

const DEFAULT_ARROW_CONFIG = {
  lineWidth: 1,
  headWidth: 0.2,
  headLength: 0.2,
  color: 0xff0000,
};

/**
 * Creates a THREE.js geometry based on the shape type and configuration
 * @param {Object} shape - Object containing shape parameters (type, dimensions)
 * @param {GeometryConfig} [geometryConfig={}] - Configuration for geometry segments
 * @returns {THREE.BufferGeometry|null} The created geometry or null if shape type is invalid
 */
export function createGeometry(shape, geometryConfig) {
  if (!shape || !shape.type) return null;
  const config = { ...DEFAULT_GEOMETRY_CONFIG, ...geometryConfig };
  let geometry;

  switch (shape.type) {
    case "box":
      geometry = new THREE.BoxGeometry(
        shape.hx * 2,
        shape.hy * 2,
        shape.hz * 2,
        config.box.widthSegments,
        config.box.heightSegments,
        config.box.depthSegments
      );
      break;
    case "sphere":
      geometry = new THREE.SphereGeometry(
        shape.radius,
        config.sphere.widthSegments,
        config.sphere.heightSegments
      );
      break;
    case "cylinder":
      geometry = new THREE.CylinderGeometry(
        shape.radius,
        shape.radius,
        shape.height,
        config.cylinder.radialSegments,
        config.cylinder.heightSegments
      );
      geometry.rotateX(Math.PI / 2);
      break;
    case "mesh":
      geometry = new THREE.BufferGeometry();
      const positions = new Float32Array(shape.vertices.flat());
      geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(positions, 3)
      );
      const indices = new Uint16Array(shape.faces.flat());
      geometry.setIndex(new THREE.BufferAttribute(indices, 1));
      geometry.computeVertexNormals();
      break;
    case "pointcloud":
      geometry = null; // Handled separately in createVisualRepresentations
      break;
    default:
      console.error("Invalid shape type:", shape.type);
      return null;
  }
  return geometry;
}

/**
 * Creates a THREE.js Points object from a point cloud
 * @param {Array<Array<number>>} pointCloud - Array of 3D points
 * @param {PointsConfig} [pointsConfig={}] - Configuration for points appearance
 * @param {boolean} [visible=false] - Initial visibility of the points
 * @returns {THREE.Points|null} The created Points object or null if pointCloud is empty
 */
export function createPoints(pointCloud, pointsConfig, visible = true) {
  if (!pointCloud || pointCloud.length === 0) return null;
  const config = { ...DEFAULT_POINTS_CONFIG, ...pointsConfig };
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(pointCloud.flat());
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3)
  );

  const material = new THREE.PointsMaterial({
    size: config.size,
    opacity: config.opacity,
    alphaTest: config.alphaTest,
    transparent: config.transparent,
    sizeAttenuation: config.sizeAttenuation,
  });

  if (config.texture) {
    const texture = new THREE.TextureLoader().load(config.texture);
    texture.colorSpace = THREE.SRGBColorSpace;
    material.map = texture;
  }

  if (config.color) {
    material.color = new THREE.Color(config.color);
  }
  // Create points object
  const points = new THREE.Points(geometry, material);
  points.isPoints = true;
  points.visible = visible;
  return points;
}

/**
 * Creates a THREE.js Points object from a point cloud
 * @param {Array<Array<number>>} pointCloud - Array of 3D points
 * @param {PointsConfig} [pointsConfig={}] - Configuration for points appearance
 * @param {boolean} [visible=false] - Initial visibility of the points
 * @returns {THREE.Points|null} The created Points object or null if pointCloud is empty
 */
export function createContactPoints(pointCloud, pointsConfig) {
  if (!pointCloud || pointCloud.length === 0) return null;

  const config = { ...DEFAULT_POINTS_CONFIG, ...pointsConfig };

  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(pointCloud.flat());
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3)
  );

  const material = new THREE.ShaderMaterial({
    uniforms: {
      color: { value: new THREE.Color(config.color) },
      opacity: { value: config.opacity },
      sizeAttenuation: { value: config.sizeAttenuation !== false },
      useTexture: { value: false },
      pointTexture: { value: null },
      alphaTest: { value: 0.5 }, // Add alphaTest uniform
    },
    vertexShader: `
      attribute float size;
      uniform bool sizeAttenuation;

      void main() {
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mvPosition;

        // Handle size attenuation
        if (sizeAttenuation) {
          gl_PointSize = size * (300.0 / -mvPosition.z);
        } else {
          gl_PointSize = size;
        }
      }
    `,
    fragmentShader: `
      uniform vec3 color;
      uniform float opacity;
      uniform bool useTexture;
      uniform sampler2D pointTexture;
      uniform float alphaTest;

      void main() {
        // Create a circular point with smooth edges
        vec2 center = gl_PointCoord - vec2(0.5);
        float dist = length(center) * 2.0;

        // Smooth circle with anti-aliasing
        float alpha = 1.0 - smoothstep(0.8, 1.0, dist);

        // Apply alpha test
        if (alpha < alphaTest) discard;

        vec4 outputColor = vec4(color, alpha * opacity);
        if (useTexture) {
          vec4 texColor = texture2D(pointTexture, gl_PointCoord);
          outputColor *= texColor;
        }

        gl_FragColor = outputColor;
      }
    `,
    transparent: true,
    depthWrite: false, // Important for proper transparency
    depthTest: true,
  });

  // If texture is provided, load and set it
  if (config.texture) {
    const texture = new THREE.TextureLoader().load(config.texture);
    material.uniforms.pointTexture.value = texture;
    material.uniforms.useTexture.value = true;
  }

  const points = new THREE.Points(geometry, material);
  points.visible = config.visible !== false;

  return points;
}

/**
 * Creates a wireframe representation of a geometry
 * @param {THREE.BufferGeometry} geometry - The geometry to create wireframe from
 * @param {WireframeConfig} [wireframeConfig={}] - Configuration for wireframe appearance
 * @param {boolean} [visible=false] - Initial visibility of the wireframe
 * @returns {THREE.LineSegments} The created wireframe object
 */
export function createWireframe(geometry, wireframeConfig, visible = true) {
  if (!geometry) return null;

  const config = { ...DEFAULT_WIREFRAME_CONFIG, ...wireframeConfig };

  const wireframe = new THREE.LineSegments(
    new THREE.WireframeGeometry(geometry),
    new THREE.LineBasicMaterial(config)
  );
  wireframe.visible = visible;
  wireframe.isWireframe = true;
  return wireframe;
}

/**
 * Creates a THREE.js mesh with standard material and environment mapping
 * @param {THREE.BufferGeometry} geometry - The geometry for the mesh
 * @param {MeshConfig} [meshConfig={}] - Configuration for mesh appearance
 * @param {boolean} [visible=true] - Initial visibility of the mesh
 * @returns {THREE.Mesh} The created mesh object
 */
export function createMesh(geometry, meshConfig, visible = true) {
  if (!geometry) return null;

  const config = { ...DEFAULT_MESH_CONFIG, ...meshConfig };

  let envMap = null;
  if (config.envMapPath) {
    const format = ".jpg";
    const urls = [
      config.envMapPath + "nx" + format,
      config.envMapPath + "px" + format,
      config.envMapPath + "pz" + format,
      config.envMapPath + "nz" + format,
      config.envMapPath + "py" + format,
      config.envMapPath + "ny" + format,
    ];
    envMap = new THREE.CubeTextureLoader().load(urls);
  }

  const material = new THREE.MeshStandardMaterial({
    color: config.color,
    roughness: config.roughness,
    metalness: config.metalness,
    opacity: config.opacity,
    envMapIntensity: config.envMapIntensity,
    transparent: config.transparent,
    envMap: envMap,
    side: THREE.DoubleSide,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.isMesh = true;
  mesh.visible = visible;
  return mesh;
}

/**
 * Creates a single arrow
 * @param {THREE.Vector3} start - Starting point of the arrow
 * @param {THREE.Vector3} end - End point of the arrow
 * @param {ArrowConfig} [arrowConfig={}] - Arrow configuration
 * @returns {THREE.ArrowHelper} The created arrow object
 */
export function createArrow(start, end, arrowConfig = {}) {
  const config = { ...DEFAULT_ARROW_CONFIG, ...arrowConfig };
  const dir = end.clone().sub(start);
  const length = dir.length();

  const arrow = new THREE.ArrowHelper(
    dir.normalize(),
    start,
    length,
    config.color,
    config.headLength,
    config.headWidth
  );

  arrow.line.material.linewidth = config.lineWidth;
  return arrow;
}

/**
 * Creates multiple arrows from arrays of start and end points
 * @param {THREE.Vector3[]} starts - Array of starting points
 * @param {THREE.Vector3[]} ends - Array of end points
 * @param {ArrowConfig|ArrowConfig[]} configs - Single config or array of configs for each arrow
 * @returns {THREE.Group} Group containing all created arrows
 */
export function createArrows(starts, ends, configs = {}) {
  if (starts.length !== ends.length) {
    console.error("Number of start and end points must match");
    return null;
  }

  const arrowGroup = new THREE.Group();

  starts.forEach((start, index) => {
    const config = Array.isArray(configs) ? configs[index] : configs;
    const arrow = createArrow(start, ends[index], config);
    arrowGroup.add(arrow);
  });

  return arrowGroup;
}

export function generateDivergingPalette(colors, numColors, correctLightness) {
  // Split colors into left/right gradients
  const midpointIndex = Math.floor(colors.length / 2);
  const leftColors = colors.slice(0, midpointIndex + 1);
  const rightColors = colors.slice(midpointIndex);

  // Create two separate scales
  const leftScale = chroma
    .bezier(leftColors)
    .scale()
    .correctLightness(correctLightness)
    .mode("lab");

  const rightScale = chroma
    .bezier(rightColors)
    .scale()
    .correctLightness(correctLightness)
    .mode("lab");

  // Generate and combine halves
  const numEach = Math.ceil(numColors / 2);
  return [
    ...leftScale.colors(numEach).slice(0, -1),
    ...rightScale.colors(numEach),
  ];
}
