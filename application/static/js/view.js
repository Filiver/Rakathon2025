import { ImagePanel } from "./ui/ImagePanel.js";
import { AnimationController } from "./components/AnimationController.js"; // Import AnimationController
import { PlaybackControls } from "./ui/PlaybackControls.js"; // Import PlaybackControls
import { ContourAdjustmentPanel } from "./ui/ContourAdjustmentPanel.js"; // Import Contour Panel
import { MeasurementSelector } from "./ui/MeasurementSelector.js"; // Import Measurement Selector
import { StatusPanel } from "./ui/StatusPanel.js"; // Import Status Panel

export class View {
  constructor() {
    this.socket = null;
    this.imageSet = null; // { references: [uri1, uri2,...], measurements: [uri1, uri2,...] }
    this.availableFiles = null;
    this.imagePanel = null;
    this.animationController = null; // Add animation controller property
    this.playbackControls = null; // Add playback controls property
    this.contourPanel = null; // Add contour panel property
    this.measurementSelector = null; // Add measurement selector property
    this.statusPanel = null; // Add status panel property
    this.isInitialized = false;
    this.statusUpdateInterval = null; // For periodic status updates
  }

  static run() {
    // Wait for the DOM to be fully loaded before initializing the view
    document.addEventListener("DOMContentLoaded", () => {
      const simView = new View();
      simView.init();
    });
  }

  /**
   * Handles the 'images_data' event from the server.
   * Stores the data, initializes controllers, and updates the ImagePanel.
   * Removes the loading splash screen on first successful data load.
   * @param {object} imagesData - Object containing 'references' and 'measurements' arrays of data URIs.
   */
  imagesDataHandler(imagesData) {
    console.log("Received images data");

    // Re-enable the measurement selector if it was disabled
    if (this.measurementSelector) {
      this.measurementSelector.enableConfirmButton();
    }

    // Clear previous animation/playback state if controllers exist
    if (this.animationController) {
      this.animationController.pause(); // Stop playback
      // Optionally reset frame to 0, or let the new setFrame handle it
    }
    // Dispose old playback controls if they exist, before creating new ones
    if (this.playbackControls) {
      this.playbackControls.dispose();
      this.playbackControls = null;
    }

    this.initFromImagesData(imagesData);

    // Check if data is valid before proceeding
    const hasReferences = this.imageSet?.references?.length > 0;
    const hasMeasurements = this.imageSet?.measurements?.length > 0;

    if (hasReferences && hasMeasurements && this.imagePanel) {
      // Ensure both arrays have the same length for synchronized playback
      const frameCount = Math.min(
        this.imageSet.references.length,
        this.imageSet.measurements.length
      );

      if (frameCount <= 0) {
        console.warn("No frames available to display.");
        return; // Exit if no frames
      }

      console.log(`Initializing controllers for ${frameCount} frames.`);

      // Initialize AnimationController and PlaybackControls only once *per data load*
      if (!this.animationController) {
        this.animationController = new AnimationController(this, frameCount);
      } else {
        // Update existing controller
        this.animationController.totalFrames = frameCount;
        this.animationController.setFrame(0, false); // Reset to first frame without updating controls yet
      }

      // Always create new playback controls for the new data set
      this.playbackControls = new PlaybackControls(this.animationController);

      // Set the initial frame (frame 0) - this also updates the new playback controls
      this.animationController.setFrame(0);

      // Remove splash screen only once after the first successful data load
      if (!this.isInitialized) {
        const splash = document.getElementById("loading-splash");
        if (splash) {
          splash.remove();
          console.log("Initial data loaded, splash screen removed.");
        }
        this.isInitialized = true; // Mark as initialized
      }
    } else {
      console.warn("Image data is missing, incomplete, or panel not ready.");
    }
  }

  /**
   * Displays the images corresponding to the given frame index.
   * @param {number} frameIndex - The index of the frame to display.
   */
  displayFrame(frameIndex) {
    if (
      !this.imageSet ||
      !this.imageSet.references ||
      !this.imageSet.measurements ||
      !this.imagePanel
    ) {
      return; // Not ready
    }

    const refIndex = Math.min(frameIndex, this.imageSet.references.length - 1);
    const measIndex = Math.min(
      frameIndex,
      this.imageSet.measurements.length - 1
    );

    if (refIndex >= 0 && measIndex >= 0) {
      this.imagePanel.update(
        this.imageSet.references[refIndex],
        this.imageSet.measurements[measIndex]
      );

      // If capturing, capture the newly displayed frame (after images are potentially loaded/rendered)
      // Use a small timeout to allow the browser to render the images before capture
      if (this.playbackControls?.isCapturing) {
        requestAnimationFrame(() => {
          // Ensure rendering cycle completes
          setTimeout(() => this.playbackControls.captureFrame(), 50); // Small delay
        });
      }
    } else {
      console.warn(`Invalid frame index requested: ${frameIndex}`);
    }
  }

  initializeSocket() {
    const socket = io();
    this.socket = socket;
    socket.on("disconnect", () => console.log("Disconnected from server"));
    socket.on("error", (error) => {
      console.error("WebSocket Error:", error);

      // Re-enable the measurement selector if it was disabled
      if (this.measurementSelector) {
        this.measurementSelector.enableConfirmButton();
      }
    });
    socket.on("connect", () => {
      console.log("Connected to server");
      // Get available files to populate the selector
      socket.emit("get_available_files");
      // DO NOT request images immediately anymore, let the selector handle it
    });

    // Listen for available files
    socket.on("available_files", (filesData) => {
      console.log("Received available files");
      this.initFromAvailableFiles(filesData);
      // Populate the measurement selector with both reference and measurement dates
      if (
        this.measurementSelector &&
        filesData.references &&
        filesData.measurements
      ) {
        this.measurementSelector.populateOptions(
          filesData.references,
          filesData.measurements
        );
      }
      // No automatic 'get_images' request here - user must click confirm
    });

    // Listen for the actual image data
    // Bind the handler to 'this' context
    socket.on("images_data", this.imagesDataHandler.bind(this));

    // Add status event listener
    socket.on("status_update", (statusData) => {
      console.log("Received status update");
      if (this.statusPanel) {
        this.statusPanel.updateStatus(statusData);
      }
    });
  }

  initFromAvailableFiles(availableFiles) {
    this.availableFiles = availableFiles;
    // TODO: Update UI to show available files if needed
    console.log("Available:", this.availableFiles);
  }

  initFromImagesData(imagesData) {
    this.imageSet = imagesData;
    console.log(
      `Stored ${this.imageSet?.references?.length || 0} reference images and ${
        this.imageSet?.measurements?.length || 0
      } measurement images.`
    );
  }

  // Simplified initialization
  init() {
    // Set body background to white
    document.body.style.backgroundColor = "#FFFFFF";

    // Create status panel before other UI components
    this.statusPanel = new StatusPanel("image-panel-container"); // Pass parent ID for status panel

    // Instantiate UI Components
    this.imagePanel = new ImagePanel("image-panel-container"); // Pass parent ID
    this.contourPanel = new ContourAdjustmentPanel("contour-panel-container"); // Updated parent ID

    // Connect the contour panel to the image panel
    this.contourPanel.setImagePanel(this.imagePanel);

    this.initializeSocket(); // Initialize socket *before* measurement selector needs it
    this.measurementSelector = new MeasurementSelector(
      this.socket,
      "left-panel-container"
    ); // Pass socket and parent ID

    // Start periodic status updates
    this.startStatusUpdates();

    // Removed splash screen removal from here
    console.log("View initialized, waiting for available files...");
  }

  /**
   * Fetch the current system status from the server
   */
  fetchStatus() {
    if (!this.socket) return;

    this.socket.emit("get_status");
  }

  /**
   * Start periodic status updates
   */
  startStatusUpdates() {
    // Clear any existing interval first
    if (this.statusUpdateInterval) {
      clearInterval(this.statusUpdateInterval);
    }

    // Set up periodic status updates (every 5 seconds)
    this.statusUpdateInterval = setInterval(() => this.fetchStatus(), 5000);

    // Also fetch immediately
    this.fetchStatus();
  }

  /**
   * Called when the measurement selector sends a request for new data
   * Updates the UI to show loading state
   */
  onDataRequested() {
    // Show loading in status panel
    if (this.statusPanel) {
      this.statusPanel.showDataLoading("Requesting image data...");
    }

    // Disable confirm button in measurement selector
    if (this.measurementSelector) {
      this.measurementSelector.disableConfirmButton();
    }
  }

  // Removed disposeOfAll and animate methods - Add proper disposal if needed
  dispose() {
    // Stop status updates
    if (this.statusUpdateInterval) {
      clearInterval(this.statusUpdateInterval);
      this.statusUpdateInterval = null;
    }

    // Dispose UI components
    this.statusPanel?.dispose();
    this.imagePanel?.dispose(); // Assuming ImagePanel might have listeners/elements to clean up
    this.playbackControls?.dispose();
    this.contourPanel?.dispose();
    this.measurementSelector?.dispose();
    this.animationController?.dispose();

    // Disconnect socket
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    console.log("View disposed.");
  }
}
