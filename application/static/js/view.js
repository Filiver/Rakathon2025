import { ImagePanel } from "./ui/ImagePanel.js";
import { AnimationController } from "./components/AnimationController.js"; // Import AnimationController
import { PlaybackControls } from "./ui/PlaybackControls.js"; // Import PlaybackControls

export class View {
  constructor() {
    this.socket = null;
    this.imageSet = null; // { references: [uri1, uri2,...], measurements: [uri1, uri2,...] }
    this.availableFiles = null;
    this.imagePanel = null;
    this.animationController = null; // Add animation controller property
    this.playbackControls = null; // Add playback controls property
    this.isInitialized = false;
  }

  static run() {
    const simView = new View();
    simView.init();
  }

  /**
   * Handles the 'images_data' event from the server.
   * Stores the data, initializes controllers, and updates the ImagePanel.
   * Removes the loading splash screen on first successful data load.
   * @param {object} imagesData - Object containing 'references' and 'measurements' arrays of data URIs.
   */
  imagesDataHandler(imagesData) {
    console.log("Received images data");
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

      // Initialize AnimationController and PlaybackControls only once
      if (!this.animationController) {
        this.animationController = new AnimationController(this, frameCount);
      } else {
        // Update existing controller if data is reloaded (optional)
        this.animationController.totalFrames = frameCount;
        this.animationController.setFrame(0, false); // Reset to first frame without updating controls yet
      }

      if (!this.playbackControls) {
        // Append controls to the body or a specific container
        this.playbackControls = new PlaybackControls(
          this,
          document.body,
          frameCount
        );
      } else {
        // Update existing controls if data is reloaded (optional)
        this.playbackControls.totalFrames = frameCount;
        this.playbackControls.slider.max = Math.max(0, frameCount - 1);
        // Update slider+counter via animation controller's setFrame
      }

      // Set the initial frame (frame 0)
      this.animationController.setFrame(0); // This will call displayFrame and update controls

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
    socket.on("error", (error) => console.error("WebSocket Error:", error));
    socket.on("connect", () => {
      console.log("Connected to server");
      // Initially get available files (optional, depends on workflow)
      socket.emit("get_available_files");
      // Then request the images (assuming we want them immediately)
      // You might want to trigger this based on user selection later
      socket.emit("get_images", {}); // Request images
    });

    // Listen for available files (if needed for selection later)
    socket.on("available_files", (filesData) => {
      console.log("Received available files");
      this.initFromAvailableFiles(filesData);
      // Potentially update a UI element to show available files here
    });

    // Listen for the actual image data
    // Bind the handler to 'this' context
    socket.on("images_data", this.imagesDataHandler.bind(this));
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

    // Instantiate the ImagePanel, assuming 'image-container' exists in index.html
    this.imagePanel = new ImagePanel("image-container");
    this.initializeSocket();
    // Removed splash screen removal from here
    console.log("View initialized, waiting for data...");
  }

  // Removed disposeOfAll and animate methods
}
