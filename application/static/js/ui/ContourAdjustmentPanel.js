// Color constants for each anatomical structure
const COLORS = {
  GTV: "#FF4500", // Orange-Red
  CTV: "#FFD700", // Gold
  PTV: "#32CD32", // Lime Green
  SPINAL_CORD: "#1E90FF", // Dodger Blue
  PAROTID: "#9370DB", // Medium Purple
};

export class ContourAdjustmentPanel {
  constructor(parentElementId = "contour-panel-container", imagePanel = null) {
    // Accept parent ID and image panel reference
    this.imagePanel = imagePanel;

    const parentElement = document.getElementById(parentElementId);
    if (!parentElement) {
      console.error(
        `ContourAdjustmentPanel: Parent element #${parentElementId} not found.`
      );
      return;
    }

    this.container = document.createElement("div");
    this.container.id = "contour-adjustment-panel";

    // Style container similar to the measurement selector
    Object.assign(this.container.style, {
      padding: "8px 12px", // Reduced from 15px
      backgroundColor: "rgba(40, 40, 40, 0.85)",
      borderRadius: "4px", // Reduced from 8px
      boxShadow: "0 2px 5px rgba(0, 0, 0, 0.3)",
      border: "1px solid #000", // Added black border
      color: "white",
      fontFamily: "sans-serif",
      display: "flex",
      flexDirection: "row",
      alignItems: "center",
      gap: "10px", // Reduced from 15px
      width: "100%",
      height: "46px", // Reduced from 56px to match MeasurementSelector
      boxSizing: "border-box",
      justifyContent: "flex-start",
      fontSize: "0.9rem",
      zIndex: "10", // Add high z-index to ensure visibility
      position: "relative", // Add positioning context
    });

    // Create title element ALL CAPS
    const title = document.createElement("span");
    title.textContent = "DISP:";
    title.style.fontWeight = "bold";
    title.style.fontSize = "0.9rem"; // Match container font size
    this.container.appendChild(title);

    // Create checkboxes for each structure with full names
    this.createCheckbox("gtv", "GTV", COLORS.GTV);
    this.createCheckbox("ctv", "CTV", COLORS.CTV);
    this.createCheckbox("ptv", "PTV", COLORS.PTV);
    this.createCheckbox("spinal_cord", "SPINAL CORD", COLORS.SPINAL_CORD); // Full name
    this.createCheckbox("parotid", "PAROTID", COLORS.PAROTID); // Full name

    // Add a divider
    const divider = document.createElement("div");
    divider.style.height = "30px";
    divider.style.width = "1px";
    divider.style.backgroundColor = "rgba(255, 255, 255, 0.3)";
    this.container.appendChild(divider);

    // Add projection controls with clearer labels
    this.createProjectionCheckbox("ref_on_meas", "R → M", "#FFFFFF");
    this.createProjectionCheckbox("meas_on_ref", "M → R", "#FFFFFF");

    // REMOVED: Cross-projection settings section that was causing errors

    parentElement.appendChild(this.container);
  }

  /**
   * Creates a checkbox with label and adds it to the container
   * @param {string} id - Identifier for the checkbox
   * @param {string} label - Display text for the checkbox (in ALL CAPS)
   * @param {string} color - Color associated with this structure
   */
  createCheckbox(id, label, color) {
    const checkboxContainer = document.createElement("div");
    checkboxContainer.style.display = "flex";
    checkboxContainer.style.alignItems = "center";
    checkboxContainer.style.gap = "3px"; // Reduced from 5px
    checkboxContainer.style.height = "22px"; // Reduced from 25px
    checkboxContainer.style.padding = "0 3px"; // Reduced from 0 4px

    // Create the checkbox
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = `contour-${id.toLowerCase()}`;
    checkbox.checked = true;
    checkbox.style.marginRight = "2px";
    // Don't scale down the checkbox, keep normal size
    checkbox.addEventListener("change", (e) =>
      this.handleCheckboxChange(e, id, color)
    );

    // Create label for the checkbox (ALL CAPS)
    const checkboxLabel = document.createElement("label");
    checkboxLabel.htmlFor = checkbox.id;
    checkboxLabel.textContent = label;
    checkboxLabel.style.fontSize = "0.9rem"; // Match font size

    // Create color indicator
    const colorIndicator = document.createElement("span");
    colorIndicator.style.display = "inline-block";
    colorIndicator.style.width = "12px"; // Slightly larger indicator
    colorIndicator.style.height = "12px"; // Slightly larger indicator
    colorIndicator.style.backgroundColor = color;
    colorIndicator.style.borderRadius = "50%";
    colorIndicator.style.marginLeft = "2px";

    // Append elements to checkbox container
    checkboxContainer.appendChild(checkbox);
    checkboxContainer.appendChild(checkboxLabel);
    checkboxContainer.appendChild(colorIndicator);

    // Add to main container
    this.container.appendChild(checkboxContainer);

    return checkbox;
  }

  /**
   * Creates a projection checkbox with label
   * @param {string} id - Identifier for the projection checkbox
   * @param {string} label - Display text for the checkbox (in ALL CAPS)
   * @param {string} color - Color associated with this control
   */
  createProjectionCheckbox(id, label, color) {
    const checkboxContainer = document.createElement("div");
    checkboxContainer.style.display = "flex";
    checkboxContainer.style.alignItems = "center";
    checkboxContainer.style.gap = "3px"; // Reduced from 5px
    checkboxContainer.style.height = "22px"; // Reduced from 25px
    checkboxContainer.style.padding = "0 3px"; // Reduced from 0 4px

    // Create the checkbox
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = `projection-${id.toLowerCase()}`;
    checkbox.checked = false;
    checkbox.style.marginRight = "2px";
    // Don't scale down the checkbox, keep normal size
    checkbox.addEventListener("change", (e) =>
      this.handleProjectionCheckboxChange(e, id)
    );

    // Create label for the checkbox with arrow notation
    const checkboxLabel = document.createElement("label");
    checkboxLabel.htmlFor = checkbox.id;
    checkboxLabel.textContent = label; // Use provided label with arrow (REF → MEAS)
    checkboxLabel.style.fontSize = "0.9rem"; // Match font size

    // Append elements to checkbox container
    checkboxContainer.appendChild(checkbox);
    checkboxContainer.appendChild(checkboxLabel);

    // Add to main container
    this.container.appendChild(checkboxContainer);

    return checkbox;
  }

  /**
   * Generic handler for contour checkbox changes
   * @param {Event} event - The change event
   * @param {string} structureId - The ID of the anatomical structure
   * @param {string} color - The color associated with this structure
   */
  handleCheckboxChange(event, structureId, color) {
    const isChecked = event.target.checked;
    console.log(
      `${structureId} visualization ${
        isChecked ? "enabled" : "disabled"
      } (${color})`
    );

    // Update contour visibility in the ImagePanel if available
    if (this.imagePanel) {
      this.imagePanel.setContourVisibility(
        structureId.toLowerCase(),
        isChecked
      );
    }
  }

  /**
   * Handler for projection checkbox changes
   */
  handleProjectionCheckboxChange(event, projectionId) {
    const isChecked = event.target.checked;
    console.log(
      `Projection ${projectionId} ${isChecked ? "enabled" : "disabled"}`
    );

    // Update projection settings in the ImagePanel
    if (this.imagePanel) {
      if (projectionId === "ref_on_meas") {
        this.imagePanel.setProjectRefOnMeas(isChecked);
      } else if (projectionId === "meas_on_ref") {
        this.imagePanel.setProjectMeasOnRef(isChecked);
      }
    }
  }

  /**
   * Set the reference to the ImagePanel for contour toggling
   * @param {ImagePanel} imagePanel - The image panel to control
   */
  setImagePanel(imagePanel) {
    this.imagePanel = imagePanel;
  }

  dispose() {
    this.container.remove();
  }
}
