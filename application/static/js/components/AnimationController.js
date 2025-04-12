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
}
