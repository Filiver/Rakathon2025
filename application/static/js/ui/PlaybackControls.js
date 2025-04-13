const buttonHeight = 25;
export class PlaybackControls {
  constructor(animationController) {
    this.animationController = animationController;
    this.minRenderDelay = 1000 / 15;
    this.lastRenderTime = Number.NEGATIVE_INFINITY;
    this.container = document.createElement("div");
    Object.assign(this.container.style, {
      position: "absolute",
      bottom: "5px",
      left: "50%",
      transform: "translateX(-50%)",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: "3px",
      backgroundColor: "rgba(242, 152, 34, 0.85)", // Dark blue/navy color
      padding: "8px 15px",
      borderRadius: "10px",
      boxShadow: "0 6px 16px rgba(0, 0, 0, 0.15), 0 2px 4px rgba(0, 0, 0, 0.1)",
      backdropFilter: "blur(5px)", // Adds a modern frosted glass effect
      transition: "all 0.2s ease",
    });

    this.controlsRow = document.createElement("div");
    Object.assign(this.controlsRow.style, {
      display: "flex",
      alignItems: "center",
      flexDirection: "row",
      gap: "15px",
    });

    this.isDragging = false; // Add state for dragging

    // **Cache listener functions**
    this.recordButtonClick = () => {
      if (this.animationController.isRecording) {
        this.animationController.stopRecording();
        this.recordButton.textContent = "⚫ REC";
        this.recordButton.style.backgroundColor = "#444";
      } else {
        this.animationController.startRecording();
        this.recordButton.textContent = "⬛ STOP";
        this.recordButton.style.backgroundColor = "#aa0000";
      }
    };

    this.formatSelectChange = (e) => {
      this.animationController.setRecordingFormat(e.target.value);
    };

    this.playButtonClick = () => {
      this.animationController.togglePlayPause();
      this.playButton.textContent = this.animationController.isPlaying
        ? "Pause"
        : "Play";
    };

    this.stepBackButtonClick = () => {
      this.animationController.pause();
      this.animationController.setFrame(
        this.animationController.currentFrame - 1
      );
      this.playButton.textContent = "Play";
    };

    this.stepForwardButtonClick = () => {
      this.animationController.pause();
      this.animationController.setFrame(
        this.animationController.currentFrame + 1
      );
      this.playButton.textContent = "Play";
    };

    this.speedSelectChange = (e) => {
      console.debug("Speed changed to", e.target.value);
      this.animationController.setSpeed(parseFloat(e.target.value));
    };

    // ** Refactor progress bar interaction **
    this.handleProgressBarInteraction = (event) => {
      const rect = this.progressBarContainer.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const progress = Math.max(0, Math.min(1, x / rect.width)); // Clamp progress between 0 and 1
      const targetTime = progress * this.animationController.getTotalTime();

      if (event.type === "click" && event.altKey) {
        // Pause only on Alt+Click
        this.animationController.pause();
        this.playButton.textContent = "Play";
      }
      // Update frame immediately on click or drag
      this.animationController.goToTime(targetTime);
    };

    this.progressBarMouseDown = (event) => {
      this.isDragging = true;
      this.progressBarContainer.style.cursor = "grabbing";
      this.handleProgressBarInteraction(event); // Update on initial click
      // Add temporary listeners to window for smoother dragging
      window.addEventListener("mousemove", this.progressBarMouseMove);
      window.addEventListener("mouseup", this.progressBarMouseUp);
    };

    this.progressBarMouseMove = (event) => {
      if (this.isDragging) {
        this.handleProgressBarInteraction(event);
      }
    };

    this.progressBarMouseUp = () => {
      if (this.isDragging) {
        this.isDragging = false;
        this.progressBarContainer.style.cursor = "pointer";
        // Remove temporary listeners
        window.removeEventListener("mousemove", this.progressBarMouseMove);
        window.removeEventListener("mouseup", this.progressBarMouseUp);
      }
    };
    // ** End Refactor **

    this.keydownListener = (event) => {
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey)
        return;
      switch (event.key) {
        case "r":
          this.recordButton.click();
          break;
        case "ArrowRight":
          // Directly call setFrame for stepping
          this.animationController.pause();
          this.animationController.setFrame(
            this.animationController.currentFrame + 1
          );
          this.playButton.textContent = "Play";
          break;
        case "ArrowLeft":
          // Directly call setFrame for stepping
          this.animationController.pause();
          this.animationController.setFrame(
            this.animationController.currentFrame - 1
          );
          this.playButton.textContent = "Play";
          break;
        case " ":
          this.playButton.click();
          event.target.blur(); // Prevent space bar scrolling page
          break;
      }
    };

    // **Create elements and attach cached listeners**
    this.recordButton = this.#createButton(
      "⚫ REC",
      this.recordButtonClick,
      "100px"
    );

    // Format select dropdown styling
    this.formatSelect = document.createElement("select");
    ["jpg", "png"].forEach((format) => {
      const option = document.createElement("option");
      option.value = format;
      option.text = format.toUpperCase();
      this.formatSelect.appendChild(option);
    });
    Object.assign(this.formatSelect.style, {
      width: "80px",
      textAlign: "center",
      height: buttonHeight + "px",
      backgroundColor: "#1e3a8a", // Dark blue like the buttons
      color: "white",
      border: "none",
      borderRadius: "6px",
      cursor: "pointer",
      boxShadow: "0 2px 4px rgba(0, 0, 0, 0.2)",
      padding: "0 5px",
    });
    this.formatSelect.addEventListener("change", this.formatSelectChange);

    this.playButton = this.#createButton("Play", this.playButtonClick, "70px");

    this.stepBackButton = this.#createButton(
      "←",
      this.stepBackButtonClick,
      "40px"
    );

    this.stepForwardButton = this.#createButton(
      "→",
      this.stepForwardButtonClick,
      "40px"
    );

    // Speed select dropdown styling
    this.speedSelect = document.createElement("select");
    [0.1, 0.25, 0.5, 1, 2, 5].forEach((speed) => {
      const option = document.createElement("option");
      option.value = speed;
      option.text = `${speed}x`;
      if (speed === 1) option.selected = true;
      this.speedSelect.appendChild(option);
    });
    Object.assign(this.speedSelect.style, {
      width: "80px",
      textAlign: "center",
      height: buttonHeight + "px",
      backgroundColor: "#1e3a8a", // Dark blue like the buttons
      color: "white",
      border: "none",
      borderRadius: "6px",
      cursor: "pointer",
      boxShadow: "0 2px 4px rgba(0, 0, 0, 0.2)",
      padding: "0 5px",
    });
    this.speedSelect.addEventListener("change", this.speedSelectChange);

    this.frameCounter = document.createElement("span");
    Object.assign(this.frameCounter.style, {
      color: "white",
      display: "flex",
      alignItems: "center",
      justifyContent: "center", // Center the text
      height: buttonHeight + "px",
      marginLeft: "5px",
      fontFamily: "monospace",
      width: "100px", // Fixed width
      backgroundColor: "#1e3a8a", // Dark blue like buttons
      borderRadius: "6px",
      padding: "0 5px",
      boxShadow: "0 2px 4px rgba(0, 0, 0, 0.2)",
      textAlign: "center",
      whiteSpace: "nowrap", // Prevent wrapping
    });

    this.progressBarContainer = document.createElement("div");
    Object.assign(this.progressBarContainer.style, {
      position: "relative",
      width: "100%",
      marginLeft: "15px",
      marginRight: "15px",
      display: "flex",
      height: "10px",
      backgroundColor: "rgba(209, 213, 219, 0.3)", // Light gray with transparency
      borderRadius: "6px",
      cursor: "pointer",
      marginTop: "5px",
      overflow: "hidden",
      boxShadow: "inset 0 1px 3px rgba(0, 0, 0, 0.2)",
      transition: "background-color 0.2s ease",
    });

    this.progressBarContainer.addEventListener("mouseenter", () => {
      this.progressBarContainer.style.backgroundColor =
        "rgba(209, 213, 219, 0.4)";
    });

    this.progressBarContainer.addEventListener("mouseleave", () => {
      this.progressBarContainer.style.backgroundColor =
        "rgba(209, 213, 219, 0.3)";
    });

    this.progressBar = document.createElement("div");
    Object.assign(this.progressBar.style, {
      width: "0%",
      height: "100%",
      backgroundColor: "#1e40af", // Dark blue color
      borderRadius: "6px",
      transition: "width 0.1s ease-out",
      backgroundImage: "linear-gradient(to right, #1e40af, #3b82f6)", // Blue gradient
      boxShadow: "0 0 4px rgba(59, 130, 246, 0.5)", // Blue glow
    });
    this.progressBarContainer.appendChild(this.progressBar);
    // Replace single click listener with mousedown
    this.progressBarContainer.addEventListener(
      "mousedown",
      this.progressBarMouseDown
    );
    // Add mouseleave to cancel drag if mouse leaves the bar while pressed
    this.progressBarContainer.addEventListener(
      "mouseleave",
      this.progressBarMouseUp
    );

    // Add frame markers to the progress bar
    this.renderFrameMarkers = () => {
      const totalFrames = this.animationController.totalFrames;
      if (!this.frameMarkers && totalFrames > 0) {
        this.frameMarkers = document.createElement("div");
        this.frameMarkers.className = "frame-markers";
        Object.assign(this.frameMarkers.style, {
          position: "absolute",
          top: "0",
          left: "0",
          width: "100%",
          height: "100%",
          pointerEvents: "none",
          display: "flex",
          alignItems: "center",
        });

        // Only add markers if we have a reasonable number (avoid visual clutter)
        if (totalFrames <= 100) {
          for (let i = 0; i < totalFrames; i++) {
            const marker = document.createElement("div");
            const position = (i / (totalFrames - 1)) * 100;
            Object.assign(marker.style, {
              position: "absolute",
              left: `${position}%`,
              height: i % 5 === 0 ? "70%" : "40%", // Taller markers every 5 frames
              width: "1px",
              backgroundColor: "rgba(255, 255, 255, 0.4)",
              transform: "translateX(-50%)",
            });
            this.frameMarkers.appendChild(marker);
          }
        }

        this.progressBarContainer.appendChild(this.frameMarkers);
      }
    };

    // Assemble controls row
    [
      this.recordButton,
      this.formatSelect,
      this.stepBackButton,
      this.playButton,
      this.stepForwardButton,
      this.speedSelect,
      this.frameCounter,
    ].forEach((element) => this.controlsRow.appendChild(element));

    this.container.appendChild(this.controlsRow);
    this.container.appendChild(this.progressBarContainer);
    document.body.appendChild(this.container);

    // Attach document-level listener
    document.addEventListener("keydown", this.keydownListener);

    this.updateElements();
  }

  updateElements() {
    const currentFrame = this.animationController.currentFrame + 1; // Add 1 for human-readable indexing (1-based)
    const totalFrames = this.animationController.totalFrames;
    // Format with fixed width using padStart for consistent display
    this.frameCounter.textContent = `${currentFrame}/${totalFrames}`;

    // We still need time values for progress calculation
    const currentTime = this.animationController.getCurrentTime();
    const totalTime = this.animationController.getTotalTime();
    const progress = currentTime / totalTime;
    this.progressBar.style.width = `${(progress * 100).toFixed(1)}%`;

    // Ensure frame markers are rendered
    this.renderFrameMarkers();

    this.lastRenderTime = Number.NEGATIVE_INFINITY;
  }

  /**
   * Updates the text of the play/pause button based on the animation state.
   * @param {boolean} isPlaying - Whether the animation is currently playing.
   */
  updatePlayButton(isPlaying) {
    this.playButton.textContent = isPlaying ? "Pause" : "Play";
  }

  updateSliderAndCounter(frameIndex) {
    this.updateElements(); // Reuse existing logic to update time and progress bar
  }

  forceRedraw() {
    this.updateElements();
    this.lastRenderTime = Number.NEGATIVE_INFINITY;
  }

  animate(now) {
    if (now - this.lastRenderTime < this.minRenderDelay) return;
    this.updateElements();
    this.lastRenderTime = now;
  }

  dispose() {
    // Remove the container from the DOM
    this.container.remove();

    // Remove all event listeners using cached functions
    this.recordButton.removeEventListener("click", this.recordButtonClick);
    this.formatSelect.removeEventListener("change", this.formatSelectChange);
    this.playButton.removeEventListener("click", this.playButtonClick);
    this.stepBackButton.removeEventListener("click", this.stepBackButtonClick);
    this.stepForwardButton.removeEventListener(
      "click",
      this.stepForwardButtonClick
    );
    this.speedSelect.removeEventListener("change", this.speedSelectChange);
    // Remove old progress bar listener and add new ones
    this.progressBarContainer.removeEventListener(
      "mousedown",
      this.progressBarMouseDown
    );
    this.progressBarContainer.removeEventListener(
      "mouseleave",
      this.progressBarMouseUp
    );
    // Ensure window listeners are removed if dispose is called during a drag
    window.removeEventListener("mousemove", this.progressBarMouseMove);
    window.removeEventListener("mouseup", this.progressBarMouseUp);

    document.removeEventListener("keydown", this.keydownListener);
  }

  // Update button style to match the dark blue theme
  #createButton(text, onClick, width = "auto") {
    const button = document.createElement("button");
    Object.assign(button.style, {
      padding: "5px 10px",
      backgroundColor: "#1e3a8a", // Dark blue background
      color: "white",
      border: "none",
      borderRadius: "6px",
      cursor: "pointer",
      width: width,
      minWidth: "40px",
      height: buttonHeight + "px",
      display: "inline-flex",
      justifyContent: "center",
      alignItems: "center",
      boxShadow: "0 2px 4px rgba(0, 0, 0, 0.2)",
      transition: "all 0.2s ease",
    });

    button.addEventListener("mouseenter", () => {
      button.style.backgroundColor = "#2563eb"; // Lighter blue on hover
      button.style.boxShadow = "0 3px 6px rgba(0, 0, 0, 0.25)";
    });

    button.addEventListener("mouseleave", () => {
      button.style.backgroundColor = "#1e3a8a"; // Back to dark blue
      button.style.boxShadow = "0 2px 4px rgba(0, 0, 0, 0.2)";
    });

    button.textContent = text;
    button.addEventListener("click", onClick);
    return button;
  }
}
