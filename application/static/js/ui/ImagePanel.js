export class ImagePanel {
  /**
   * Creates an instance of ImagePanel and appends its container to the specified parent element.
   * @param {string} parentElementId - The ID of the HTML element to append the panel to.
   */
  constructor(parentElementId = "image-panel-container") {
    // Accept parent ID
    const parentElement = document.getElementById(parentElementId);
    if (!parentElement) {
      console.error(
        `ImagePanel: Parent element #${parentElementId} not found.`
      );
      return;
    }

    // Create the main container element
    this.container = document.createElement("div");
    this.container.id = "image-panel-content"; // Changed ID to avoid conflict with parent
    parentElement.appendChild(this.container); // Append to the specific parent

    // Basic styling for side-by-side canvases within the container
    Object.assign(this.container.style, {
      display: "flex",
      justifyContent: "center", // Center images horizontally
      alignItems: "center", // Center images vertically
      flexWrap: "nowrap",
      width: "90%", // Take full width of parent
      height: "100%", // Take full height of parent
      padding: "10px", // Reduced padding from 20px
      paddingTop: "0",
      marginTop: "-20px", // Added negative margin to move images up
      boxSizing: "border-box",
      overflow: "visible", // Prevent clipping
      gap: "80px", // Increased gap between images
    });

    // Create canvases for reference and measurement images
    this.refCanvas = this.createCanvas("reference-canvas");
    this.measCanvas = this.createCanvas("measurement-canvas");

    // Store current image data
    this.currentRefData = null;
    this.currentMeasData = null;

    // Track contour visibility (default all to visible)
    this.contourVisibility = {
      gtv: true,
      ctv: true,
      ptv: true,
      spinal_cord: true,
      parotid: true,
    };

    // Add projection settings (default to off)
    this.projectRefOnMeas = false;
    this.projectMeasOnRef = false;

    // Add zoom tracking
    this.zoomLevels = {
      "reference-canvas": 1,
      "measurement-canvas": 1,
    };
    this.maxZoom = 5; // Maximum zoom level
    this.minZoom = 1; // Minimum zoom level
  }

  /**
   * Creates a canvas element with proper styling
   * @param {string} id - Canvas ID
   * @returns {HTMLCanvasElement} - The created canvas
   */
  createCanvas(id) {
    const canvasContainer = document.createElement("div");
    canvasContainer.style.position = "relative"; // For absolute positioning of the label
    canvasContainer.style.width = "45%";
    canvasContainer.style.height = "auto";
    canvasContainer.style.display = "flex";
    canvasContainer.style.justifyContent = "center";
    canvasContainer.style.alignItems = "center";
    canvasContainer.style.overflow = "hidden"; // Hide overflow when zooming

    // Create the canvas with wrapper for zoom
    const canvasWrapper = document.createElement("div");
    canvasWrapper.style.position = "relative";
    canvasWrapper.style.width = "100%";
    canvasWrapper.style.height = "100%";
    canvasWrapper.style.overflow = "hidden"; // Hide overflow when zooming

    const canvas = document.createElement("canvas");
    canvas.id = id;
    canvas.width = 512; // Default size, will be updated when image loads
    canvas.height = 512;
    Object.assign(canvas.style, {
      maxWidth: "100%",
      maxHeight: "85%",
      borderRadius: "8px",
      boxShadow: "0 4px 8px rgba(0, 0, 0, 0.3)",
      transformOrigin: "center center", // Default zoom origin
      transition: "transform 0.1s ease-out", // Smooth zoom effect
    });

    // Create and add the label box (outside the canvas wrapper)
    const labelBox = document.createElement("div");
    const isReference = id.includes("reference");
    labelBox.textContent = isReference ? "REFERENCE" : "MEASUREMENT";
    Object.assign(labelBox.style, {
      position: "absolute",
      top: "10px",
      left: "10px",
      backgroundColor: "rgba(255, 255, 255, 0.7)",
      color: "#000",
      padding: "3px 6px",
      fontSize: "11px",
      fontWeight: "bold",
      fontFamily: "monospace",
      borderRadius: "4px",
      zIndex: "50", // Reduced from 100 to be below status dropdown but above canvas
      pointerEvents: "none", // Don't interfere with mouse events
    });

    // Add wheel event listener for zooming
    canvasContainer.addEventListener("wheel", (event) => {
      event.preventDefault(); // Prevent page scrolling
      this.handleZoom(event, canvas);
    });

    // Add reset zoom on double click
    canvasContainer.addEventListener("dblclick", (event) => {
      this.resetZoom(canvas);
    });

    canvasWrapper.appendChild(canvas);
    canvasContainer.appendChild(canvasWrapper);
    canvasContainer.appendChild(labelBox); // Add label outside the canvas wrapper
    this.container.appendChild(canvasContainer);
    return canvas;
  }

  /**
   * Handle zoom event on canvas
   * @param {WheelEvent} event - The wheel event
   * @param {HTMLCanvasElement} canvas - The canvas being zoomed
   */
  handleZoom(event, canvas) {
    // Get current zoom level
    const currentZoom = this.zoomLevels[canvas.id];

    // Calculate zoom delta based on wheel direction
    const delta = event.deltaY < 0 ? 0.1 : -0.1;

    // Calculate new zoom level with bounds
    let newZoom = Math.max(
      this.minZoom,
      Math.min(this.maxZoom, currentZoom + delta)
    );

    // Get mouse position relative to canvas
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    // Calculate mouse position as percentage of canvas dimensions
    const mouseXPercent = mouseX / rect.width;
    const mouseYPercent = mouseY / rect.height;

    // Set transform origin to mouse position
    canvas.style.transformOrigin = `${mouseXPercent * 100}% ${
      mouseYPercent * 100
    }%`;

    // Apply the new zoom level
    canvas.style.transform = `scale(${newZoom})`;
    this.zoomLevels[canvas.id] = newZoom;

    // Store zoom origin for when frames change
    if (!this.zoomOrigins) this.zoomOrigins = {};
    this.zoomOrigins[canvas.id] = {
      x: mouseXPercent * 100,
      y: mouseYPercent * 100,
    };
  }

  /**
   * Reset zoom level to default
   * @param {HTMLCanvasElement} canvas - The canvas to reset zoom for
   */
  resetZoom(canvas) {
    canvas.style.transformOrigin = "center center";
    canvas.style.transform = "scale(1)";
    this.zoomLevels[canvas.id] = 1;

    // Reset label transform
    const label = canvas.parentElement.querySelector("div");
    if (label) {
      label.style.transform = "scale(1)";
    }
  }

  /**
   * Updates the canvases with the provided image data and contours
   * @param {Object} refData - Reference image data object with dataUri and contours
   * @param {Object} measData - Measurement image data object with dataUri and contours
   */
  update(refData, measData) {
    // Store current data for redrawing when contour visibility changes
    this.currentRefData = refData;
    this.currentMeasData = measData;

    // Draw both canvases
    this.drawCanvas(this.refCanvas, refData, measData, true);
    this.drawCanvas(this.measCanvas, measData, refData, false);
  }

  /**
   * Draw an image and its contours on a canvas
   * @param {HTMLCanvasElement} canvas - The canvas to draw on
   * @param {Object} imageData - Primary image data object containing dataUri and contours
   * @param {Object} otherImageData - The other image's data for projection
   * @param {boolean} isReference - Whether this is the reference canvas (affects projection)
   */
  drawCanvas(canvas, imageData, otherImageData, isReference) {
    if (!imageData || !imageData.dataUri) return;

    const ctx = canvas.getContext("2d");
    const img = new Image();

    img.onload = () => {
      // Don't reset zoom when loading new images
      // Instead, maintain the current zoom level and origin
      const currentZoom = this.zoomLevels[canvas.id] || 1;

      // Resize canvas to match image dimensions
      canvas.width = img.width;
      canvas.height = img.height;

      // Clear canvas and draw image
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      // Draw native contours based on visibility settings
      this.drawContours(ctx, imageData);

      // Draw projected contours if enabled
      if (isReference && this.projectMeasOnRef && otherImageData) {
        this.drawProjectedContours(ctx, otherImageData);
      } else if (!isReference && this.projectRefOnMeas && otherImageData) {
        this.drawProjectedContours(ctx, otherImageData);
      }

      // Reapply zoom after drawing new frame
      if (currentZoom !== 1) {
        // Use stored origin if available, otherwise default to center
        const origin =
          this.zoomOrigins && this.zoomOrigins[canvas.id]
            ? `${this.zoomOrigins[canvas.id].x}% ${
                this.zoomOrigins[canvas.id].y
              }%`
            : "center center";

        canvas.style.transformOrigin = origin;
        canvas.style.transform = `scale(${currentZoom})`;
      }
    };

    img.src = imageData.dataUri;
  }

  /**
   * Draw native contours on canvas context based on visibility settings
   * @param {CanvasRenderingContext2D} ctx - Canvas 2D context
   * @param {Object} imageData - Image data with contours
   */
  drawContours(ctx, imageData) {
    // Define colors for each contour type (matching our constants in ContourAdjustmentPanel)
    const contourColors = {
      gtv: "#FF4500",
      ctv: "#FFD700",
      ptv: "#32CD32",
      spinal_cord: "#1E90FF",
      parotid: "#9370DB",
    };

    // Draw each visible contour
    Object.keys(this.contourVisibility).forEach((contourType) => {
      if (this.contourVisibility[contourType] && imageData[contourType]) {
        const points = imageData[contourType];
        if (points && points.length > 0) {
          ctx.beginPath();
          ctx.moveTo(points[0][0], points[0][1]);

          // Draw lines connecting all points
          for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i][0], points[i][1]);
          }

          // Style and stroke the path - solid line for native contours
          ctx.strokeStyle = contourColors[contourType];
          ctx.lineWidth = 3;
          ctx.setLineDash([]); // Solid line
          ctx.stroke();
        }
      }
    });
  }

  /**
   * Draw projected contours with dashed lines
   * @param {CanvasRenderingContext2D} ctx - Canvas 2D context
   * @param {Object} imageData - Image data with contours to project
   */
  drawProjectedContours(ctx, imageData) {
    // Define colors for each contour type (matching our constants in ContourAdjustmentPanel)
    const contourColors = {
      gtv: "#FF4500",
      ctv: "#FFD700",
      ptv: "#32CD32",
      spinal_cord: "#1E90FF",
      parotid: "#9370DB",
    };

    // Draw each visible contour as projection (dashed)
    Object.keys(this.contourVisibility).forEach((contourType) => {
      if (this.contourVisibility[contourType] && imageData[contourType]) {
        const points = imageData[contourType];
        if (points && points.length > 0) {
          ctx.beginPath();
          ctx.moveTo(points[0][0], points[0][1]);

          // Draw lines connecting all points
          for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i][0], points[i][1]);
          }

          // Style and stroke the path - dashed line for projected contours
          ctx.strokeStyle = contourColors[contourType];
          ctx.lineWidth = 2;
          ctx.setLineDash([5, 5]); // Dashed line pattern
          ctx.stroke();
        }
      }
    });

    // Reset line dash to avoid affecting other drawings
    ctx.setLineDash([]);
  }

  /**
   * Update the visibility of a specific contour type
   * @param {string} contourType - Type of contour (e.g., 'gtv', 'ctv')
   * @param {boolean} isVisible - Whether the contour should be visible
   */
  setContourVisibility(contourType, isVisible) {
    if (this.contourVisibility.hasOwnProperty(contourType)) {
      this.contourVisibility[contourType] = isVisible;

      // Redraw canvases with updated visibility settings
      if (this.currentRefData && this.currentMeasData) {
        this.drawCanvas(
          this.refCanvas,
          this.currentRefData,
          this.currentMeasData,
          true
        );
        this.drawCanvas(
          this.measCanvas,
          this.currentMeasData,
          this.currentRefData,
          false
        );
      }
    }
  }

  /**
   * Set whether to project reference contours on measurement image
   * @param {boolean} enabled - Whether projection is enabled
   */
  setProjectRefOnMeas(enabled) {
    this.projectRefOnMeas = enabled;
    // Redraw if we have data
    if (this.currentRefData && this.currentMeasData) {
      this.drawCanvas(
        this.measCanvas,
        this.currentMeasData,
        this.currentRefData,
        false
      );
    }
  }

  /**
   * Set whether to project measurement contours on reference image
   * @param {boolean} enabled - Whether projection is enabled
   */
  setProjectMeasOnRef(enabled) {
    this.projectMeasOnRef = enabled;
    // Redraw if we have data
    if (this.currentRefData && this.currentMeasData) {
      this.drawCanvas(
        this.refCanvas,
        this.currentRefData,
        this.currentMeasData,
        true
      );
    }
  }

  // Add a dispose method for cleanup if needed
  dispose() {
    this.container.remove();
  }
}
