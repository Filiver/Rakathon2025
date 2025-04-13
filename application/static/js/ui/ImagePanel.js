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
      alignItems: "flex-start", // Align images to the top instead of center
      flexWrap: "nowrap",
      width: "93%", // Take full width of parent
      height: "100%", // Take full height of parent
      padding: "10px", // Reduced padding from 20px
      paddingTop: "30px", // Added padding at top
      marginTop: "-20px", // Added negative margin to move images up
      marginBottom: "10px", // Added margin at bottom
      boxSizing: "border-box",
      overflow: "visible", // Prevent clipping
      gap: "80px", // Increased gap between images
    });

    // Add scale factor for larger images
    this.scaleFactor = 1.4; // Increase this value to make images larger

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
    canvasContainer.style.width = "45%"; // Original width
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
    canvasWrapper.style.borderRadius = "12px"; // Move border radius from canvas to wrapper
    canvasWrapper.style.boxShadow =
      "0 8px 16px rgba(0, 77, 64, 0.15), 0 2px 4px rgba(0, 77, 64, 0.1)"; // Softer teal-tinted shadow

    const canvas = document.createElement("canvas");
    canvas.id = id;
    canvas.width = 512; // Default size, will be updated when image loads
    canvas.height = 512;
    Object.assign(canvas.style, {
      maxWidth: "100%",
      maxHeight: "100%",
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
      backgroundColor: "rgba(224, 242, 241, 0.85)", // Light teal background
      color: "#004d40", // Dark teal text for better contrast
      padding: "4px 8px",
      fontSize: "11px",
      fontWeight: "bold",
      fontFamily: "monospace",
      borderRadius: "6px",
      zIndex: "100", // Higher z-index to stay above zoomed canvas
      pointerEvents: "none", // Don't interfere with mouse events
      boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)", // Subtle shadow
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

      // Apply scaling factor to make images larger
      const scaledWidth = Math.round(img.width * this.scaleFactor);
      const scaledHeight = Math.round(img.height * this.scaleFactor);

      // Resize canvas to match scaled image dimensions
      canvas.width = scaledWidth;
      canvas.height = scaledHeight;

      // Clear canvas and draw scaled image
      ctx.clearRect(0, 0, scaledWidth, scaledHeight);
      ctx.drawImage(img, 0, 0, scaledWidth, scaledHeight);

      // Scale contour coordinates
      const scaleContourPoints = (points) => {
        if (!points || points.length === 0) return points;
        return points.map((point) => [
          point[0] * this.scaleFactor,
          point[1] * this.scaleFactor,
        ]);
      };

      // Create scaled contour data for drawing
      const scaledImageData = { ...imageData };
      const scaledOtherImageData = otherImageData
        ? { ...otherImageData }
        : null;

      // Scale contour coordinates for each contour type
      Object.keys(this.contourVisibility).forEach((type) => {
        if (imageData[type])
          scaledImageData[type] = scaleContourPoints(imageData[type]);
        if (otherImageData && otherImageData[type]) {
          scaledOtherImageData[type] = scaleContourPoints(otherImageData[type]);
        }
      });

      // Draw native contours based on visibility settings (with scaled coordinates)
      this.drawContours(ctx, scaledImageData);

      // Draw projected contours if enabled (with scaled coordinates)
      if (isReference && this.projectMeasOnRef && scaledOtherImageData) {
        this.drawProjectedContours(ctx, scaledOtherImageData);
      } else if (
        !isReference &&
        this.projectRefOnMeas &&
        scaledOtherImageData
      ) {
        this.drawProjectedContours(ctx, scaledOtherImageData);
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
        if (!points || points.length < 3) return; // Skip if too few points

        // Check if this is likely a closed shape (first and last points are close)
        const firstPoint = points[0];
        const lastPoint = points[points.length - 1];
        const isClosedShape = this.isCloseEnough(firstPoint, lastPoint, 5);

        // Detect potential sub-contours by finding large jumps between points
        const subContours = this.splitIntoSubContours(points);

        // Draw each sub-contour separately
        subContours.forEach((subPoints) => {
          if (subPoints.length < 2) return; // Need at least 2 points to draw a line

          ctx.beginPath();
          ctx.moveTo(subPoints[0][0], subPoints[0][1]);

          // Draw lines connecting all points in this sub-contour
          for (let i = 1; i < subPoints.length; i++) {
            ctx.lineTo(subPoints[i][0], subPoints[i][1]);
          }

          // Style and stroke the path - solid line for native contours
          ctx.strokeStyle = contourColors[contourType];
          ctx.lineWidth = 3;
          ctx.setLineDash([]); // Solid line

          // Close the path if it appears to be a closed shape and has enough points
          if (isClosedShape && subPoints.length > 2) {
            ctx.closePath();
          }

          ctx.stroke();
        });
      }
    });
  }

  /**
   * Draw projected contours with darker tones and dashed lines
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
        if (!points || points.length < 3) return; // Skip if too few points

        // Check if this is likely a closed shape (first and last points are close)
        const firstPoint = points[0];
        const lastPoint = points[points.length - 1];
        const isClosedShape = this.isCloseEnough(firstPoint, lastPoint, 5);

        // Detect potential sub-contours by finding large jumps between points
        const subContours = this.splitIntoSubContours(points);

        // Draw each sub-contour separately
        subContours.forEach((subPoints) => {
          if (subPoints.length < 2) return; // Need at least 2 points to draw a line

          ctx.beginPath();
          ctx.moveTo(subPoints[0][0], subPoints[0][1]);

          // Draw lines connecting all points in this sub-contour
          for (let i = 1; i < subPoints.length; i++) {
            ctx.lineTo(subPoints[i][0], subPoints[i][1]);
          }

          // Get base color
          const baseColor = contourColors[contourType];

          // Darken color by 30%
          const darkColor = this.darkenColor(baseColor, 0.7);

          // Style and stroke the path - dashed line for projected contours
          ctx.strokeStyle = darkColor;
          ctx.lineWidth = 3; // Thicker lines for better visibility
          ctx.setLineDash([5, 5]); // Re-enable dashed line pattern for cross-projections

          // Close the path if it appears to be a closed shape and has enough points
          if (isClosedShape && subPoints.length > 2) {
            ctx.closePath();
          }

          ctx.stroke();
        });
      }
    });

    // Reset line dash to avoid affecting other drawings
    ctx.setLineDash([]);
  }

  /**
   * Darken a color by a specified factor
   * @param {string} hexColor - The hex color to darken
   * @param {number} factor - The darkening factor (0-1)
   * @returns {string} - The darkened color as hex
   */
  darkenColor(hexColor, factor) {
    // Handle invalid inputs
    if (
      !hexColor ||
      typeof hexColor !== "string" ||
      !hexColor.startsWith("#")
    ) {
      return hexColor; // Return original if invalid
    }

    try {
      // Convert hex to RGB
      let r = parseInt(hexColor.substring(1, 3), 16);
      let g = parseInt(hexColor.substring(3, 5), 16);
      let b = parseInt(hexColor.substring(5, 7), 16);

      // Darken RGB values
      r = Math.max(0, Math.floor(r * factor));
      g = Math.max(0, Math.floor(g * factor));
      b = Math.max(0, Math.floor(b * factor));

      // Convert back to hex
      const darkHex =
        "#" +
        r.toString(16).padStart(2, "0") +
        g.toString(16).padStart(2, "0") +
        b.toString(16).padStart(2, "0");

      return darkHex;
    } catch (error) {
      console.error("Error darkening color:", error);
      return hexColor; // Return original on error
    }
  }

  /**
   * Check if two points are close enough to be considered the same point
   * @param {Array} point1 - First point [x, y]
   * @param {Array} point2 - Second point [x, y]
   * @param {number} threshold - Distance threshold
   * @returns {boolean} - True if points are close enough
   */
  isCloseEnough(point1, point2, threshold = 3) {
    const dx = point1[0] - point2[0];
    const dy = point2[1] - point1[1];
    const distanceSquared = dx * dx + dy * dy;
    return distanceSquared <= threshold * threshold;
  }

  /**
   * Split a contour into sub-contours where there are large jumps between points
   * @param {Array} points - Array of points [[x, y], ...]
   * @returns {Array} - Array of sub-contours [[[x, y], ...], ...]
   */
  splitIntoSubContours(points) {
    if (!points || points.length < 2) return [points];

    const subContours = [];
    let currentSubContour = [points[0]];

    // Distance threshold for considering a point to be part of a new sub-contour
    const jumpThreshold = 20 * this.scaleFactor; // Scale with the image scaling

    for (let i = 1; i < points.length; i++) {
      const prevPoint = points[i - 1];
      const currentPoint = points[i];

      // Calculate distance between consecutive points
      const dx = currentPoint[0] - prevPoint[0];
      const dy = currentPoint[1] - prevPoint[1];
      const distance = Math.sqrt(dx * dx + dy * dy);

      if (distance > jumpThreshold) {
        // Large jump detected, end current sub-contour and start a new one
        if (currentSubContour.length > 1) {
          subContours.push(currentSubContour);
        }
        currentSubContour = [currentPoint];
      } else {
        // Continue current sub-contour
        currentSubContour.push(currentPoint);
      }
    }

    // Add the last sub-contour if it has points
    if (currentSubContour.length > 1) {
      subContours.push(currentSubContour);
    }

    return subContours.length > 0 ? subContours : [points];
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
