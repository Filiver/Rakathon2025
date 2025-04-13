const buttonHeight = 25;
export class PlaybackControls {
  constructor(animationController) {
    this.animationController = animationController;
    this.minRenderDelay = 1000 / 15;
    this.lastRenderTime = Number.NEGATIVE_INFINITY;
    this.container = document.createElement("div");
    Object.assign(this.container.style, {
      position: "absolute",
      bottom: "20px",
      left: "50%",
      transform: "translateX(-50%)",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: "5px",
      backgroundColor: "rgba(0, 0, 0, 0.5)",
      padding: "10px 20px",
      borderRadius: "5px",
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
    });
    this.speedSelect.addEventListener("change", this.speedSelectChange);

    this.frameCounter = document.createElement("span");
    Object.assign(this.frameCounter.style, {
      color: "white",
      display: "flex",
      alignItems: "center",
      height: "30px",
      marginLeft: "5px",
      fontFamily: "monospace",
    });

    this.progressBarContainer = document.createElement("div");
    Object.assign(this.progressBarContainer.style, {
      width: "100%",
      marginLeft: "15px",
      marginRight: "15px",
      display: "flex",
      height: "8px",
      backgroundColor: "#222",
      borderRadius: "4px",
      cursor: "pointer",
      marginTop: "5px",
    });

    this.progressBar = document.createElement("div");
    Object.assign(this.progressBar.style, {
      width: "0%",
      height: "100%",
      backgroundColor: "#cc0000", // Changed from #888 (gray) to red
      borderRadius: "4px",
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
    const currentTime = this.animationController.getCurrentTime().toFixed(3);
    const totalTime = this.animationController.getTotalTime().toFixed(3);
    this.frameCounter.textContent = `time: ${currentTime} / ${totalTime}`;
    const progress = currentTime / totalTime;
    this.progressBar.style.width = `${(progress * 100).toFixed(1)}%`;
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

  #createButton(text, onClick, width = "auto") {
    const button = document.createElement("button");
    Object.assign(button.style, {
      padding: "5px 5px",
      backgroundColor: "#444",
      color: "white",
      border: "none",
      borderRadius: "4px",
      cursor: "pointer",
      width: width,
      minWidth: "40px",
      height: buttonHeight + "px",
      display: "inline-flex",
      justifyContent: "center",
      alignItems: "center",
    });
    button.textContent = text;
    button.addEventListener("click", onClick);
    return button;
  }
}
