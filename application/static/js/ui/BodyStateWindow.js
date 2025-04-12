import { FREQ_CONFIG } from "../config.js";

export class BodyStateWindow {
  static cssInjected = false;
  static styleId = "body-state-window-styles"; // Unique ID for the style tag
  constructor(app) {
    this.app = app;
    this.selectedBodies = new Set();
    this.window = null;
    this.batchSelector = null;
    this.header = null; // Keep track of header element
    this.content = null;
    this.bodyListContainer = null;
    this.detailsContainer = null;
    this.bodyListItems = new Map(); // Map body name -> list item element
    this.detailContainers = new Map(); // Map body name -> detail container element
    this.minRenderDelay = 1000 / FREQ_CONFIG.bodyStateWindow; // 60 FPS
    this.lastRenderTime = Number.NEGATIVE_INFINITY;

    this.injectCSS(); // Inject CSS styles
    this.initWindow();
  }

  injectCSS() {
    // Only attempt injection if it hasn't been done successfully before
    if (BodyStateWindow.cssInjected) {
      return;
    }

    // Check if the style tag already exists in the document (e.g., from a previous session or manual insertion)
    if (document.getElementById(BodyStateWindow.styleId)) {
      BodyStateWindow.cssInjected = true; // Mark as injected if found
      return;
    }

    // CSS rules as a string (using template literal for multiline)
    const css = `
      /* Base styles for the body state window */
      .body-state-window {
          position: fixed;
          top: 10px;
          left: 1rem;
          background-color: rgba(0, 0, 0, 0.75); /* Slightly adjusted alpha */
          color: white;
          border-radius: 8px; /* Slightly larger radius */
          font-family: Arial, sans-serif;
          z-index: 1000;
          display: flex; /* Use flexbox for layout */
          flex-direction: column; /* Stack header and content vertically */
          width: 320px;
          max-height: clamp(300px, 85vh, 90vh);
          font-size: clamp(0.8rem, 0.9vw + 0.5rem, 1rem); /* Responsive base font */
          padding: 1.0rem; /* ~20px if 1rem=16px */
      }

      /* Allow content area to scroll */
      .body-state-window-content {
          overflow-y: auto; /* Enable vertical scrolling ONLY for content */
          overflow-x: hidden; /* Prevent horizontal scroll */
          flex-grow: 1; /* Allow content to fill available space */
          padding-top: 0.5rem;
          margin-top: 0.5rem;
          border-top: 1px solid rgba(255, 255, 255, 0.5); /* Separator line */
      }

      /* Header styles */
      .body-state-window-header {
          font-weight: bold;
          font-size: 1.1em; /* Relative to parent window font size */
          display: flex;
          justify-content: space-between;
          align-items: center;
          flex-shrink: 0; /* Prevent header from shrinking */
      }

      /* Batch selector container */
      .batch-selector-container {
          display: flex;
          align-items: center;
          font-size: 0.8em; /* Smaller font size relative to header */
          font-weight: normal; /* Normal weight for selector part */
      }

      .batch-selector-label {
          margin-right: 0.5em; /* Relative spacing */
      }

      .batch-selector {
          background-color: rgba(50, 50, 50, 0.8);
          color: white;
          border: 1px solid white;
          padding: 0.25em 0.5em; /* Relative padding */
          border-radius: 3px;
          font-size: 1em; /* Inherit from container */
      }
      .batch-selector:focus {
          outline: 1px solid skyblue; /* Add focus indicator */
      }

      .body-list-title {
          font-weight: bold;
          margin-bottom: 0.5rem; /* Use rem */
          font-size: 1.0em; /* Slightly larger */
      }

      .body-list {
          margin: 0 0 1rem 0; /* Bottom margin */
          padding: 0 0 0 1.25rem; /* ~20px indent */
          list-style-type: "→ "; /* Remove default bullets */
          font-weight: bold;
      }

      .body-list-item {
          margin-bottom: 0.3rem; /* Use rem */
          cursor: pointer;
          padding: 0.25rem 0.5rem; /* Use rem */
          border-radius: 3px;
          transition: background-color 0.2s;
      }

      /* Hover effect for non-selected items */
      .body-list-item:not(.selected):hover {
          background-color: rgba(255, 255, 255, 0.1);
      }

      /* Style for selected items */
      .body-list-item.selected {
          background-color: rgba(100, 150, 255, 0.2); /* Use a selection color */
      }

      /* Details container */
      .details-container { } /* Container for all detail blocks */

      /* Individual body detail block */
      .body-detail-container {
          margin-top: 0.75rem; /* Use rem */
          padding: 0.75rem; /* Use rem */
          padding-top: 1.5rem; /* Extra top padding for close button */
          background-color: rgba(50, 50, 50, 0.6); /* Slightly more contrast */
          border-radius: 5px;
          position: relative;
          border: 1px solid rgba(255, 255, 255, 0.1); /* Subtle border */
      }

      /* Close button */
      .body-detail-close-button {
          position: absolute;
          top: 0.3rem;   /* Use rem */
          right: 0.5rem; /* Use rem */
          cursor: pointer;
          font-size: 1.1em; /* Relative size */
          color: #aaa;
          width: 1.5em;   /* Relative size */
          height: 1.5em;  /* Relative size */
          text-align: center;
          line-height: 1.5em; /* Center vertically */
          border-radius: 50%;
          transition: background-color 0.2s, color 0.2s;
      }

      .body-detail-close-button:hover {
          background-color: rgba(255, 255, 255, 0.2);
          color: white;
      }

      /* Header within the detail block */
      .body-detail-header {
          font-weight: bold;
          margin-bottom: 0.75rem; /* Use rem */
          font-size: 1.1em; /* Slightly larger */
          color: #eee;
      }

      /* Table within the detail block */
      .body-detail-table {
          width: 100%;
          font-size: 0.9em; /* Slightly smaller relative font */
          border-collapse: collapse;
      }

      .body-detail-table td {
          padding: 0.25rem 0; /* Use rem for padding */
      }

      .body-detail-table td:first-child { /* Label cell */
          font-weight: bold;
          color: #ccc; /* Lighter gray */
          white-space: nowrap; /* Prevent labels wrapping */
      }

      .body-detail-table td:last-child { /* Value cell */
          font-family: monospace;
          color: #f0f0f0; /* Brighter value text */
      }
    `;

    // Create the <style> element
    const styleElement = document.createElement("style");
    styleElement.id = BodyStateWindow.styleId; // Assign the ID
    styleElement.textContent = css; // Add the CSS rules

    // Append the <style> element to the document's <head>
    try {
      document.head.appendChild(styleElement);
      BodyStateWindow.cssInjected = true; // Mark as successfully injected
    } catch (error) {
      console.error("Failed to inject BodyStateWindow CSS:", error);
      // Optionally handle the error, e.g., by falling back to inline styles
    }
  }

  initWindow() {
    // --- Window setup ---
    this.window = document.createElement("div");
    this.window.classList.add("body-state-window"); // Use CSS class

    // --- Header setup ---
    this.header = document.createElement("div");
    this.header.classList.add("body-state-window-header"); // Use CSS class
    this.window.appendChild(this.header);

    const title = document.createElement("span");
    title.textContent = "Body states";
    // No specific class needed for title span unless more styling is required
    this.header.appendChild(title);

    // --- Batch Selector (if needed) ---
    if (this.app.batchManager && this.app.batchManager.getbatchSize) {
      const batchSize = this.app.batchManager.getbatchSize();
      if (batchSize > 1) {
        const selectorContainer = document.createElement("div");
        selectorContainer.classList.add("batch-selector-container"); // Use CSS class

        const label = document.createElement("span");
        label.textContent = "Batch: ";
        label.classList.add("batch-selector-label"); // Use CSS class
        selectorContainer.appendChild(label);

        const batchSelector = document.createElement("select");
        batchSelector.classList.add("batch-selector"); // Use CSS class

        for (let i = 0; i < batchSize; i++) {
          const option = document.createElement("option");
          option.value = i;
          option.textContent = `${i}`;
          batchSelector.appendChild(option);
        }

        batchSelector.addEventListener("change", (e) => {
          const batchIndex = parseInt(e.target.value);
          this.app.batchManager.setActiveBatch(batchIndex);
          e.target.blur(); // Remove focus from the selector
        });

        selectorContainer.appendChild(batchSelector);
        this.header.appendChild(selectorContainer); // Append to header
        this.batchSelector = batchSelector;
      }
    }

    // --- Content Area (for scrolling) ---
    this.content = document.createElement("div");
    this.content.classList.add("body-state-window-content"); // Use CSS class for scrolling
    this.window.appendChild(this.content);

    // --- Body List and Details Containers (inside the content area) ---
    this.bodyListContainer = document.createElement("div");
    this.bodyListContainer.classList.add("body-list-container"); // Use CSS class
    this.content.appendChild(this.bodyListContainer);

    this.detailsContainer = document.createElement("div");
    this.detailsContainer.classList.add("details-container"); // Use CSS class
    this.content.appendChild(this.detailsContainer);

    document.body.appendChild(this.window);

    // --- Event Listener (unchanged) ---
    window.addEventListener("keydown", (event) => {
      if (event.code === "Space") {
        const activeElement = document.activeElement;
        if (activeElement === this.batchSelector) {
          return;
        }
        event.preventDefault();
      }
    });

    // Initialize body list once
    this.updateBodyList();
  }

  updateBodyList() {
    this.bodyListContainer.innerHTML = ""; // Clear previous content
    this.bodyListItems.clear();

    const title = document.createElement("div");
    title.textContent = "Bodies:";
    title.classList.add("body-list-title"); // Use CSS class
    this.bodyListContainer.appendChild(title);

    const list = document.createElement("ul");
    list.classList.add("body-list"); // Use CSS class

    for (const name of this.app.bodies.keys()) {
      const item = document.createElement("li");
      item.classList.add("body-list-item"); // Use CSS class
      item.textContent = name;
      item.dataset.bodyName = name; // Store name for easier access if needed

      // Add click handler for selection toggle
      item.addEventListener("click", () => {
        // Use arrow function for concise 'this'
        if (this.selectedBodies.has(name)) {
          this.deselectBody(name);
        } else {
          this.selectBody(name);
        }
      });

      list.appendChild(item);
      this.bodyListItems.set(name, item);

      // Initial style update (if it might be pre-selected somehow)
      this.updateBodyListItemStyle(name);
    }

    this.bodyListContainer.appendChild(list);
  }

  selectBody(name) {
    if (!this.selectedBodies.has(name)) {
      this.selectedBodies.add(name);
      const body = this.app.bodies.get(name);
      if (body) {
        const container = this.createBodyDetailContainer(body);
        this.detailsContainer.appendChild(container);
        this.detailContainers.set(name, container);
        this.updateBodyListItemStyle(name); // Update style via class
      }
    }
  }

  deselectBody(name) {
    if (this.selectedBodies.has(name)) {
      this.selectedBodies.delete(name);
      const container = this.detailContainers.get(name);
      if (container) {
        this.detailsContainer.removeChild(container);
        this.detailContainers.delete(name);
      }
      this.updateBodyListItemStyle(name); // Update style via class
    }
  }

  updateBodyListItemStyle(name) {
    const item = this.bodyListItems.get(name);
    if (item) {
      // Toggle 'selected' class based on the set
      if (this.selectedBodies.has(name)) {
        item.classList.add("selected");
      } else {
        item.classList.remove("selected");
      }
      // Hover styles are now handled purely by CSS :hover pseudo-class
    }
  }

  createBodyDetailContainer(body) {
    const container = document.createElement("div");
    container.classList.add("body-detail-container"); // Use CSS class

    // Close button
    const closeButton = document.createElement("div");
    closeButton.innerHTML = "✕"; // Keep content
    closeButton.classList.add("body-detail-close-button"); // Use CSS class
    // Hover effect is handled by CSS

    closeButton.addEventListener("click", (e) => {
      e.stopPropagation(); // Prevent triggering other clicks
      this.deselectBody(body.name);
    });
    container.appendChild(closeButton);

    // Details header
    const detailsHeader = document.createElement("div");
    detailsHeader.classList.add("body-detail-header"); // Use CSS class
    container.appendChild(detailsHeader);
    container.header = detailsHeader; // Keep reference for updates

    // Details table
    const table = document.createElement("table");
    table.classList.add("body-detail-table"); // Use CSS class

    const properties = [
      { key: "positions", label: "Position" },
      { key: "rotations", label: "Rotation" },
      { key: "velocity", label: "Velocity" }, // Abbreviate slightly if needed
      { key: "angularVelocity", label: "Ang. Vel." },
      { key: "force", label: "Force" },
      { key: "torque", label: "Torque" },
    ];

    const valueCells = {}; // Keep reference for updates
    for (const prop of properties) {
      if (body.availableAttributes.has(prop.key) || body[prop.key]) {
        const row = table.insertRow();
        const labelCell = row.insertCell(0);
        labelCell.textContent = prop.label;
        // Styling handled by CSS '.body-detail-table td:first-child'
        const valueCell = row.insertCell(1);
        // Styling handled by CSS '.body-detail-table td:last-child'
        valueCells[prop.key] = valueCell; // Store reference to the cell
      }
    }

    container.appendChild(table);
    container.valueCells = valueCells; // Attach value cell references to the container

    // Initial update of content
    this.updateBodyDetailContainer(container, body);
    return container;
  }

  // --- update, updateBodyDetailContainer, setSelectedBatch, show, hide, dispose, forceRedraw, animate ---
  // These methods should remain largely unchanged as they deal with data logic,
  // not the initial setup and styling which we've refactored.

  update() {
    // Only update existing detail containers
    for (const [name, container] of this.detailContainers) {
      const body = this.app.bodies.get(name);
      if (body) {
        this.updateBodyDetailContainer(container, body);
      }
    }
  }

  updateBodyDetailContainer(container, body) {
    // Check if batchManager exists and has the required properties/methods
    const batchManager = this.app.batchManager;
    let batchText = "";
    const batchIndex = batchManager.currentlyActiveBatch;
    const batchSize = batchManager.getbatchSize();
    if (batchSize > 1) {
      batchText = ` (Batch ${batchIndex})`;
    }

    container.header.textContent = `${body.name}${batchText}`;

    for (const [key, cell] of Object.entries(container.valueCells)) {
      // Ensure the property and the specific batch index exist before accessing
      var vector = null;
      if (body[key] && body[key][batchIndex]) {
        vector = body[key][batchIndex];
      } else if (body.availableAttributes.has(key)) {
        vector = body.attributeStorage.get(key)[batchIndex];
      }
      // Check if vector has x, y, z properties before calling toFixed
      if (
        vector &&
        typeof vector.x === "number" &&
        typeof vector.y === "number" &&
        typeof vector.z === "number"
      ) {
        cell.textContent = `(${vector.x.toFixed(3)}, ${vector.y.toFixed(
          3
        )}, ${vector.z.toFixed(3)})`;
      } else {
        cell.textContent = "N/A"; // Handle cases where data might be missing/malformed
      }
    }
  }

  setSelectedBatch(batchIndex) {
    if (this.batchSelector) {
      this.batchSelector.value = batchIndex;
    }
    this.update(); // Update displayed details for the new batch
  }

  show() {
    if (this.window) {
      this.window.style.display = "flex"; // Use flex since the class uses it
    }
  }

  hide() {
    if (this.window) {
      this.window.style.display = "none";
    }
  }

  dispose() {
    if (this.window && this.window.parentNode) {
      this.window.parentNode.removeChild(this.window);
      this.window = null; // Clear reference
      // Could also remove the keydown listener here if necessary
    }
  }

  forceRedraw() {
    this.lastRenderTime = Number.NEGATIVE_INFINITY;
    this.update();
  }

  animate(now) {
    if (!this.window || this.window.style.display === "none") return; // Don't update if hidden

    if (now - this.lastRenderTime < this.minRenderDelay) return;
    this.lastRenderTime = now;
    this.update();
  }
}
