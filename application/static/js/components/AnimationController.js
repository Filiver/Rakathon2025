import { PlaybackControls } from "../ui/PlaybackControls.js";

export class AnimationController {
  constructor(app, simulationTimestep) {
    this.app = app;
    this.playbackControls = null;
    this.states = null;
    this.isPlaying = false;
    this.playbackSpeed = 1;
    this.isRecording = false;
    this.capturer = null;
    this.startTime = null;
    this.recordingFormat = "jpg"; // Default recording format
    this.currentStateIndex = 0;
    this.totalTime = 0; // Total animation time
    this.currentTime = 0; // Current time in the animation
    this.simulationTimestep = simulationTimestep; // Simulation timestep
    this.lastUpdateTime = null;
  }

  loadAnimation(states) {
    this.states = states;
    this.totalTime = this.states[this.states.length - 1].time;
    this.playbackControls = new PlaybackControls(this);
    this.goToTime(0);
  }

  play() {
    if (!this.isPlaying) {
      this.isPlaying = true;
      this.lastUpdateTime = performance.now();
    }
  }

  pause() {
    this.isPlaying = false;
  }

  setSpeed(speed) {
    this.playbackSpeed = speed;
  }

  setRecordingFormat(format) {
    this.recordingFormat = format;
  }

  getRecordingOptionsForFormat(format) {
    const options = {
      framerate: 30,
      verbose: false,
      motionBlurFrames: 0,
    };

    switch (format) {
      case "jpg":
        options.format = "jpg";
        options.quality = 600;
        break;
      case "png":
        options.format = "png";
        break;
      default:
        throw new Error(`Unsupported format: ${format}`);
    }
    return options;
  }

  forceRedrawStaticElements() {
    this.playbackControls.forceRedraw();
    this.app.bodyStateWindow.forceRedraw();
  }

  getStateIndexForTime(targetTime) {
    var q = targetTime / this.simulationTimestep;
    q = Math.round(q);
    if (q < 0) q = 0;
    if (q >= this.states.length) q = this.states.length - 1;
    return q;
  }

  seekToIndex(index) {
    this.currentStateIndex = index;
    this.currentTime = this.states[index].time;
    this.updateScene();
  }

  stepForward() {
    if (this.isPlaying) return; // Ignore if playing
    const newIndex = (this.currentStateIndex + 1) % this.states.length;
    this.seekToIndex(newIndex);
    this.forceRedrawStaticElements();
  }

  stepBackward() {
    if (this.isPlaying) return; // Ignore if playing
    const newIndex =
      (this.currentStateIndex - 1 + this.states.length) % this.states.length;
    this.seekToIndex(newIndex);
    this.forceRedrawStaticElements();
  }

  goToTime(time) {
    if (time < 0 || time > this.totalTime) return; // Ignore out-of-bounds time
    const index = this.getStateIndexForTime(time);
    this.seekToIndex(index);
    if (!this.isPlaying) {
      this.forceRedrawStaticElements();
    }
  }

  startRecording() {
    if (this.isRecording) return;
    // Reset animation to start
    this.seekToIndex(0);
    const options = this.getRecordingOptionsForFormat(this.recordingFormat);
    // Initialize CCapture
    this.capturer = new CCapture(options);
    this.isRecording = true;
    this.startTime = performance.now();
    // Start recording and play animation if not already playing
    this.capturer.start();
    if (!this.isPlaying) {
      this.play();
    }
  }

  getTotalTime() {
    return this.totalTime;
  }

  getCurrentTime() {
    return this.currentTime;
  }

  getCurrentStateIndex() {
    return this.currentStateIndex;
  }

  getCurrentState() {
    return this.states[this.currentStateIndex];
  }

  stopRecording() {
    if (!this.isRecording) return;
    this.isRecording = false;
    this.capturer.stop();
    this.capturer.save();
    // Reset recording state
    this.startTime = null;
  }

  animate(now) {
    if (!this.isPlaying || !this.states) return;
    if (this.lastUpdateTime === null) {
      this.lastUpdateTime = now;
    }
    const dt = (now - this.lastUpdateTime) / 1000;
    this.lastUpdateTime = now;
    this.currentTime += dt * this.playbackSpeed;
    this.currentTime = this.currentTime % this.totalTime;
    const newStateIndex = this.getStateIndexForTime(this.currentTime);
    // Update state if changed
    if (newStateIndex !== this.currentStateIndex) {
      this.seekToIndex(newStateIndex);
      this.playbackControls.animate(now);
      if (this.app.bodyStateWindow) {
        this.app.bodyStateWindow.animate(now);
      }
    }

    if (this.isRecording && this.capturer) {
      const elapsed = now - this.startTime;
      const duration = this.totalTime * 1000; // Convert to milliseconds
      this.capturer.capture(this.app.scene.renderer.domElement);
      // Stop recording if we've completed one loop
      if (elapsed >= duration) {
        this.playbackControls.recordButton.click();
      }
    }
  }

  updateScene() {
    if (!this.app.bodies || !this.states[this.currentStateIndex]) return;
    const state = this.states[this.currentStateIndex];
    // Update all body states
    state.bodies.forEach((bodyState) => {
      const body = this.app.bodies.get(bodyState.name);
      if (body) {
        // Update the body state - this handles both batched and non-batched formats
        body.updateState(bodyState);
      }
    });
    if (this.app.scalarPlotter) {
      this.app.scalarPlotter.setEndIndex(this.currentStateIndex);
    }
  }

  dispose() {
    if (this.playbackControls) {
      this.playbackControls.dispose();
      this.playbackControls = null;
    }
    if (this.capturer) {
      this.capturer.stop();
      this.capturer = null;
    }
  }
}
