import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CONTROLS_CONFIG, SELECTION_CONFIG } from "../config.js";

export function setupControls(camera, renderer) {
  const controls = new OrbitControls(camera, renderer.domElement);

  // Apply basic configuration
  Object.assign(controls, CONTROLS_CONFIG);

  // Add custom event handlers
  setupControlEvents(controls);

  return controls;
}

function setupControlEvents(controls) {
  let activeSelectionMode = null;

  // Prevent OrbitControls from interfering during selection
  const keyDownHandler = (event) => {
    const keyPressed = event.key.toLowerCase().replace("control", "ctrl");

    // Check if the pressed key matches any selection mode
    for (const [mode, config] of Object.entries(SELECTION_CONFIG)) {
      if (keyPressed === config.key) {
        activeSelectionMode = mode;
        controls.enabled = false; // Disable controls
        document.body.style.cursor = "crosshair"; // Change cursor to indicate selection mode
        break;
      }
    }
  };

  const keyUpHandler = (event) => {
    const keyReleased = event.key.toLowerCase().replace("control", "ctrl");
    // Check if the released key matches the active selection mode
    for (const [mode, config] of Object.entries(SELECTION_CONFIG)) {
      if (keyReleased === config.key && activeSelectionMode === mode) {
        activeSelectionMode = null;
        controls.enabled = true; // Enable controls
        document.body.style.cursor = "auto"; // Reset cursor
        break;
      }
    }
  };

  // Add event listeners for keypress
  window.addEventListener("keydown", keyDownHandler);
  window.addEventListener("keyup", keyUpHandler);

  // Update cursor during drag
  controls.addEventListener("start", () => {
    if (!activeSelectionMode) {
      document.body.style.cursor = "grabbing";
    }
  });

  controls.addEventListener("end", () => {
    if (!activeSelectionMode) {
      document.body.style.cursor = "auto";
    }
  });

  // Clean up function to remove event listeners
  controls.dispose = () => {
    window.removeEventListener("keydown", keyDownHandler);
    window.removeEventListener("keyup", keyUpHandler);
    controls.dispose();
  };

  // Optional: Add more custom event handlers if needed
  controls.addEventListener("change", () => {
    // Handle control changes
  });
}

// Helper functions for direct control manipulation
export function zoomToFit(controls, boundingSphere, offset = 1.5) {
  const fov = controls.object.fov * (Math.PI / 180);
  const distance = (boundingSphere.radius / Math.sin(fov / 2)) * offset;

  controls.object.position.copy(boundingSphere.center);
  controls.object.position.z += distance;
  controls.target.copy(boundingSphere.center);
  controls.update();
}

export function resetControls(controls) {
  controls.reset();
}

export function updateControlsConfig(controls, config) {
  Object.assign(controls, config);
  controls.update();
}

// Optional: Animation functions
export function enableAutoRotate(controls, speed = 2.0) {
  controls.autoRotate = true;
  controls.autoRotateSpeed = speed;
}

export function disableAutoRotate(controls) {
  controls.autoRotate = false;
}

// Cleanup function
export function disposeControls(controls) {
  if (controls) {
    controls.dispose();
  }
}
