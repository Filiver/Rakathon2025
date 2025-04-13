import { PlaybackControls } from "../ui/PlaybackControls.js";

const DEFAULT_FPS = 15;

export class AnimationController {
  /**
   * @param {View} view - The main view instance.
   * @param {number} totalFrames - Total number of frames available.
   * @param {number} fps - Target frames per second for playback.
   */
  constructor(view, totalFrames, fps = DEFAULT_FPS) {
    this.view = view;
    this.totalFrames = totalFrames;
    this.fps = fps;
    this.interval = 1000 / this.fps; // Milliseconds per frame

    this.isPlaying = false;
    this.currentFrame = 0;
    this.lastFrameTime = 0;
    this.animationFrameId = null; // Store requestAnimationFrame ID

    // Bind methods
    this.animate = this.animate.bind(this);
    this.play = this.play.bind(this);
    this.pause = this.pause.bind(this);
    this.setFrame = this.setFrame.bind(this);
    this.goToTime = this.goToTime.bind(this); // Bind goToTime
    this.setSpeed = this.setSpeed.bind(this); // Bind setSpeed

    this.isRecording = false; // Added recording state
    this.recordingFormat = "jpg"; // Default recording format
    this.capturedFrames = []; // To store captured frames during recording
  }

  play() {
    if (this.isPlaying || this.totalFrames <= 1) return;
    this.isPlaying = true;
    this.lastFrameTime = performance.now(); // Reset time for smooth start
    console.log("Animation playing");
    // Start the animation loop
    this.animationFrameId = requestAnimationFrame(this.animate);
    this.view.playbackControls?.updatePlayButton(true); // Update UI
  }

  pause() {
    if (!this.isPlaying) return;
    this.isPlaying = false;
    console.log("Animation paused");
    cancelAnimationFrame(this.animationFrameId); // Stop the loop
    this.animationFrameId = null;
    this.view.playbackControls?.updatePlayButton(false); // Update UI
  }

  togglePlayPause() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }

  /**
   * Sets the current frame index and updates the view.
   * @param {number} frameIndex - The desired frame index.
   * @param {boolean} [updateControls=true] - Whether to update playback controls UI.
   */
  setFrame(frameIndex, updateControls = true) {
    const newFrame = Math.max(0, Math.min(frameIndex, this.totalFrames - 1));
    if (newFrame !== this.currentFrame || updateControls) {
      // Update if frame changed or forced
      this.currentFrame = newFrame;
      this.view.displayFrame(this.currentFrame); // Tell view to update images
      if (updateControls && this.view.playbackControls) {
        this.view.playbackControls.updateSliderAndCounter(this.currentFrame);
      }
    }
  }

  /**
   * Jumps to a specific time in the animation.
   * @param {number} time - The target time in seconds.
   */
  goToTime(time) {
    const targetFrame = Math.round(time * this.fps);
    this.setFrame(targetFrame);
  }

  /**
   * Sets the playback speed multiplier.
   * @param {number} speedMultiplier - The desired speed multiplier (e.g., 1 for normal, 2 for double speed).
   */
  setSpeed(speedMultiplier) {
    const baseFps = DEFAULT_FPS; // Assuming DEFAULT_FPS is the 1x speed reference
    this.fps = baseFps * speedMultiplier;
    this.interval = 1000 / this.fps;
    console.log(
      `Animation speed set to ${speedMultiplier}x (FPS: ${this.fps})`
    );
  }

  nextFrame() {
    let nextFrameIndex = this.currentFrame + 1;
    if (nextFrameIndex >= this.totalFrames) {
      nextFrameIndex = 0; // Loop back to the start
    }
    this.setFrame(nextFrameIndex);
  }

  // The core animation loop
  animate(now) {
    if (!this.isPlaying) return;

    // Request the next frame immediately
    this.animationFrameId = requestAnimationFrame(this.animate);

    const elapsed = now - this.lastFrameTime;

    // Check if enough time has passed based on the desired FPS
    if (elapsed >= this.interval) {
      this.lastFrameTime = now - (elapsed % this.interval); // Adjust for smoother timing
      this.nextFrame();
    }
  }

  getCurrentTime() {
    return this.currentFrame / this.fps;
  }

  getTotalTime() {
    return this.totalFrames / this.fps;
  }

  dispose() {
    this.pause(); // Ensure animation loop is stopped
  }

  /**
   * Start recording frames for export
   */
  startRecording() {
    if (this.isRecording) return; // Prevent starting recording if already recording

    console.log("Starting recording");
    this.isRecording = true;
    this.capturedFrames = []; // Reset captured frames

    // If not playing, capture the current frame immediately
    if (!this.isPlaying) {
      this.captureCurrentFrame();
    }
    // If playing, frames will be captured during the animation loop
  }

  /**
   * Stop recording and prepare data for export
   */
  stopRecording() {
    if (!this.isRecording) return; // Prevent stopping if not recording

    console.log(
      `Stopping recording. Captured ${this.capturedFrames.length} frames.`
    );
    this.isRecording = false;

    // If frames were captured, trigger download
    if (this.capturedFrames.length > 0) {
      this.exportRecording();
    }
  }

  /**
   * Capture the current frame for recording
   */
  captureCurrentFrame() {
    if (!this.isRecording) return;

    // Get the canvas or element to capture
    // This implementation depends on how your view is structured
    // For example, if the view has a method to get the current frame as an image:
    const frameData = this.getCurrentFrameData();
    if (frameData) {
      this.capturedFrames.push(frameData);
      console.log(
        `Captured frame ${this.currentFrame}. Total: ${this.capturedFrames.length}`
      );
    }
  }

  /**
   * Get current frame data for recording
   * This should be customized based on what you want to capture
   */
  getCurrentFrameData() {
    // Example implementation - customize this based on your actual view structure
    // This might involve creating a canvas with the current view state, or
    // getting the data from existing canvases or image elements

    // For demonstration, we'll just save the current frame index
    // In a real implementation, you'd get the actual image data
    return {
      frameIndex: this.currentFrame,
      timestamp: Date.now(),
    };
  }

  /**
   * Export the recorded frames
   */
  exportRecording() {
    // This is a placeholder for the actual export logic
    // In a real implementation, you'd generate files based on this.recordingFormat
    console.log(
      `Exporting ${this.capturedFrames.length} frames in ${this.recordingFormat} format`
    );

    // Example: You might use a library like FileSaver.js to download the files
    // or create a ZIP archive with JSZip

    // Reset captured frames after export
    this.capturedFrames = [];
  }

  /**
   * Set the recording format (jpg, png, etc.)
   */
  setRecordingFormat(format) {
    this.recordingFormat = format;
    console.log(`Recording format set to: ${format}`);
  }
}
