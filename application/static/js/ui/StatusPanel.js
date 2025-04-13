export class StatusPanel {
  /**
   * Creates an instance of StatusPanel and appends its container to the specified parent element.
   * @param {string} parentElementId - The ID of the HTML element to append the panel to.
   */
  constructor(parentElementId = "status-panel-container") {
    // Get parent element
    const parentElement = document.getElementById(parentElementId);
    if (!parentElement) {
      console.error(
        `StatusPanel: Parent element #${parentElementId} not found.`
      );
      return;
    }

    // Create the main container with dark theme styling
    this.container = document.createElement("div");
    this.container.id = "status-dropdown";
    Object.assign(this.container.style, {
      width: "100%", // Take full width
      marginTop: "10px",
      height: "7%", // Fixed height for dropdown
      marginBottom: "5px",
      borderRadius: "4px",
      boxShadow: "0 2px 5px rgba(0, 0, 0, 0.3)",
      overflow: "visible", // Changed from hidden to allow dropdown to overflow
      transition: "all 0.3s ease",
      cursor: "pointer",
      border: "1px solid #000",
      backgroundColor: "rgba(60, 60, 60, 0.85)", // Dark background like measurement selector
      borderLeft: "4px solid #3498db", // Keep indicator color for status
      position: "relative",
      zIndex: "500", // Increased from 10 to ensure everything related to status is above image elements
    });

    // Create header bar with dark theme
    this.header = document.createElement("div");
    Object.assign(this.header.style, {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      padding: "10px 15px",
      backgroundColor: "rgba(40, 40, 40, 0.85)", // Slightly lighter than container
      borderBottom: "1px solid #000",
      fontFamily: "sans-serif",
      fontSize: "14px",
      color: "white",
      fontWeight: "500",
    });

    // Status text with label
    this.statusText = document.createElement("div");
    this.statusText.innerHTML =
      '<span style="font-weight: bold; margin-right: 5px; color: white;">STATUS:</span><span id="status-message" style="color: white;">Loading...</span> <span id="status-emoji"></span>';

    // Arrow indicator
    this.arrow = document.createElement("span");
    this.arrow.id = "dropdown-arrow";
    this.arrow.innerHTML = "▼";
    Object.assign(this.arrow.style, {
      transition: "transform 0.3s ease",
      display: "inline-block",
      marginLeft: "10px",
      fontSize: "12px",
      color: "white",
    });

    this.header.appendChild(this.statusText);
    this.header.appendChild(this.arrow);

    // Content container - now with absolute positioning to float over other content
    this.content = document.createElement("div");
    this.content.id = "status-content";
    Object.assign(this.content.style, {
      position: "absolute",
      top: "100%", // Position directly below the header
      left: "0",
      right: "0",
      padding: "15px",
      backgroundColor: "rgba(45, 45, 45, 0.95)", // Dark background for content
      maxHeight: "0",
      overflow: "hidden",
      transition: "max-height 0.3s ease",
      fontSize: "13px",
      color: "white",
      zIndex: "999", // Increased significantly to be above all normal page elements
      boxShadow: "0 4px 8px rgba(0, 0, 0, 0.3)", // Shadow for floating effect
      borderBottomLeftRadius: "4px",
      borderBottomRightRadius: "4px",
      border: "1px solid #000",
      borderTop: "none",
      visibility: "hidden", // Start hidden
    });

    // Toggle dropdown on click
    let isOpen = false;
    this.header.addEventListener("click", () => {
      isOpen = !isOpen;
      if (isOpen) {
        this.content.style.maxHeight = "300px"; // Fixed height for dropdown
        this.content.style.visibility = "visible";
        this.arrow.style.transform = "rotate(180deg)";
      } else {
        this.content.style.maxHeight = "0";
        setTimeout(() => {
          if (!isOpen) this.content.style.visibility = "hidden";
        }, 300); // Match transition duration
        this.arrow.style.transform = "rotate(0deg)";
      }
    });

    // Assemble and append to parent
    this.container.appendChild(this.header);
    this.container.appendChild(this.content);

    // Place at the beginning of the parent element
    if (parentElement.firstChild) {
      parentElement.insertBefore(this.container, parentElement.firstChild);
    } else {
      parentElement.appendChild(this.container);
    }

    // Initialize with default status
    this.updateStatus({
      message: "System ready",
      severity: 0,
      content: {
        "System State": "Initialized",
        "Last Check": "Never",
      },
    });
  }

  /**
   * Updates the status display with new message
   * @param {Object} statusData - Object containing message, severity, and content
   */
  updateStatus(statusData) {
    if (!statusData) return;

    const { message, severity, content } = statusData;
    const statusMessage = document.getElementById("status-message");
    const statusEmoji = document.getElementById("status-emoji");

    // Update message text
    if (statusMessage) {
      statusMessage.textContent = message || "Status unknown";
    }

    // Update emoji and color based on severity
    if (statusEmoji) {
      let emoji = "✅";
      let color = "#2ecc71"; // Green for OK

      switch (severity) {
        case 0: // OK
          emoji = "✅";
          color = "#2ecc71";
          break;
        case 1: // Warning
          emoji = "⚠️";
          color = "#f39c12";
          // Show warning popup for non-standard situation
          this.showBlockingPopup(
            "WARNING: Non-Standard Situation",
            message,
            color
          );
          break;
        case 2: // Danger
          emoji = "❌";
          color = "#e74c3c";
          // Show danger popup for non-standard situation
          this.showBlockingPopup(
            "DANGER: Non-Standard Situation",
            message,
            color
          );
          break;
        default:
          emoji = "ℹ️";
          color = "#3498db";
      }

      statusEmoji.textContent = emoji;
      this.container.style.borderLeft = `4px solid ${color}`;
    }

    // Update content details with dark theme styling
    if (this.content && content) {
      // Clear previous content
      this.content.innerHTML = "";

      // Create a table for structured display with dark theme styling
      const table = document.createElement("table");
      Object.assign(table.style, {
        width: "100%",
        borderCollapse: "collapse",
        fontFamily: "sans-serif",
        fontSize: "13px",
        color: "white",
      });

      // Add entries from content object
      Object.entries(content).forEach(([key, value]) => {
        const row = document.createElement("tr");

        const keyCell = document.createElement("td");
        keyCell.textContent = key;
        Object.assign(keyCell.style, {
          padding: "8px 12px",
          borderBottom: "1px solid #555", // Darker border for cells
          fontWeight: "600",
          width: "35%",
          color: "#ddd", // Light gray for keys
        });

        const valueCell = document.createElement("td");
        valueCell.textContent = value;
        Object.assign(valueCell.style, {
          padding: "8px 12px",
          borderBottom: "1px solid #555", // Darker border for cells
          color: "white", // White for values
        });

        row.appendChild(keyCell);
        row.appendChild(valueCell);
        table.appendChild(row);
      });

      this.content.appendChild(table);
    }
  }

  /**
   * Shows a blocking popup for warning/danger situations
   * @param {string} title - The popup title
   * @param {string} message - The message to display
   * @param {string} color - Border color for the popup
   */
  showBlockingPopup(title, message, color) {
    // Remove any existing popup
    const existingPopup = document.getElementById("status-blocking-popup");
    if (existingPopup) {
      existingPopup.remove();
    }

    // Create overlay
    const overlay = document.createElement("div");
    overlay.id = "status-blocking-popup";
    Object.assign(overlay.style, {
      position: "fixed",
      top: "0",
      left: "0",
      width: "100%",
      height: "100%",
      backgroundColor: "rgba(0, 0, 0, 0.75)",
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      zIndex: "1000", // Very high z-index to be above everything
    });

    // Create popup container
    const popup = document.createElement("div");
    Object.assign(popup.style, {
      width: "500px",
      maxWidth: "80%",
      backgroundColor: "rgba(40, 40, 40, 0.95)",
      borderRadius: "8px",
      boxShadow: "0 4px 20px rgba(0, 0, 0, 0.5)",
      border: `3px solid ${color}`,
      overflow: "hidden",
    });

    // Create popup header
    const popupHeader = document.createElement("div");
    Object.assign(popupHeader.style, {
      padding: "15px 20px",
      backgroundColor: color,
      color: "white",
      fontFamily: "sans-serif",
      fontSize: "18px",
      fontWeight: "bold",
      textAlign: "center",
    });
    popupHeader.textContent = title;

    // Create popup content
    const popupContent = document.createElement("div");
    Object.assign(popupContent.style, {
      padding: "20px",
      color: "white",
      fontFamily: "sans-serif",
      fontSize: "16px",
      textAlign: "center",
    });
    popupContent.textContent = message;

    // Create dismissal button
    const dismissButton = document.createElement("button");
    dismissButton.textContent = "ACKNOWLEDGE";
    Object.assign(dismissButton.style, {
      display: "block",
      margin: "0 auto 20px auto",
      padding: "10px 20px",
      backgroundColor: color,
      color: "white",
      border: "none",
      borderRadius: "4px",
      cursor: "pointer",
      fontFamily: "sans-serif",
      fontSize: "14px",
      fontWeight: "bold",
    });

    // Add hover effect
    dismissButton.addEventListener("mouseover", () => {
      dismissButton.style.backgroundColor = this.adjustColor(color, -20); // Darken on hover
    });
    dismissButton.addEventListener("mouseout", () => {
      dismissButton.style.backgroundColor = color;
    });

    // Add click handler to dismiss
    dismissButton.addEventListener("click", () => {
      overlay.remove();
    });

    // Assemble popup
    popup.appendChild(popupHeader);
    popup.appendChild(popupContent);
    popup.appendChild(dismissButton);
    overlay.appendChild(popup);

    // Add to document
    document.body.appendChild(overlay);
  }

  /**
   * Adjusts a hex color by the given percent
   * @param {string} color - Hex color
   * @param {number} percent - Percent to adjust brightness (-100 to 100)
   * @returns {string} - Adjusted hex color
   */
  adjustColor(color, percent) {
    let R = parseInt(color.substring(1, 3), 16);
    let G = parseInt(color.substring(3, 5), 16);
    let B = parseInt(color.substring(5, 7), 16);

    R = parseInt((R * (100 + percent)) / 100);
    G = parseInt((G * (100 + percent)) / 100);
    B = parseInt((B * (100 + percent)) / 100);

    R = R < 255 ? R : 255;
    G = G < 255 ? G : 255;
    B = B < 255 ? B : 255;

    R = R > 0 ? R : 0;
    G = G > 0 ? G : 0;
    B = B > 0 ? B : 0;

    const RR =
      R.toString(16).length === 1 ? "0" + R.toString(16) : R.toString(16);
    const GG =
      G.toString(16).length === 1 ? "0" + G.toString(16) : G.toString(16);
    const BB =
      B.toString(16).length === 1 ? "0" + B.toString(16) : B.toString(16);

    return "#" + RR + GG + BB;
  }

  /**
   * Cleanup resources
   */
  dispose() {
    if (this.container) {
      this.container.remove();
    }

    // Remove any popup that might be open
    const popup = document.getElementById("status-blocking-popup");
    if (popup) {
      popup.remove();
    }
  }
}
