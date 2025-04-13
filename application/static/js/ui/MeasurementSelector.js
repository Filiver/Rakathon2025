export class MeasurementSelector {
  constructor(socket, parentElementId = "left-panel-container") {
    this.socket = socket;
    const parentElement = document.getElementById(parentElementId);
    if (!parentElement) {
      console.error(
        `MeasurementSelector: Parent element #${parentElementId} not found.`
      );
      return;
    }

    this.container = document.createElement("div");
    this.container.id = "measurement-selector-block";
    Object.assign(this.container.style, {
      padding: "8px 12px", // Reduced padding from 15px
      backgroundColor: "rgba(40, 40, 40, 0.85)",
      borderRadius: "4px", // Reduced from 8px
      boxShadow: "0 2px 5px rgba(0, 0, 0, 0.3)",
      border: "1px solid #000", // Added black border
      color: "white",
      fontFamily: "sans-serif",
      display: "flex",
      flexDirection: "row", // Keep horizontal layout for dropdowns
      alignItems: "center",
      gap: "10px", // Reduced gap from 15px
      width: "auto", // Ensure it doesn't stretch unnecessarily
      flexShrink: 0, // Prevent shrinking
      height: "46px", // Reduced height from 56px
      boxSizing: "border-box",
      zIndex: "10", // Add high z-index to ensure visibility
      position: "relative", // Add positioning context
    });

    // State tracking
    this.isLoading = false;
    this.isLoaded = false;
    this.optionsChanged = false;

    // --- Reference Selector ---
    const refLabel = document.createElement("label");
    refLabel.textContent = "REF:"; // ALL CAPS label
    this.container.appendChild(refLabel);

    this.referenceSelect = document.createElement("select");
    this.referenceSelect.id = "reference-selector";
    this.applySelectStyles(this.referenceSelect);
    this.container.appendChild(this.referenceSelect);

    // --- Measurement Selector ---
    const measLabel = document.createElement("label");
    measLabel.textContent = "MEAS:"; // ALL CAPS label
    this.container.appendChild(measLabel);

    this.measurementSelect = document.createElement("select");
    this.measurementSelect.id = "measurement-selector";
    this.applySelectStyles(this.measurementSelect);
    this.container.appendChild(this.measurementSelect);

    // --- Confirm Button ---
    this.confirmButton = document.createElement("button");
    this.confirmButton.textContent = ""; // ALL CAPS button text
    Object.assign(this.confirmButton.style, {
      padding: "8px 15px",
      marginLeft: "auto", // Push button to the right if desired, or adjust gap
      backgroundColor: "#d9534f", // Default to red (needs loading)
      color: "white",
      border: "none",
      borderRadius: "4px",
      cursor: "pointer",
      fontSize: "1em",
      whiteSpace: "nowrap", // Prevent button text wrapping
      width: "120px", // Slightly wider to accommodate the spinner
      textAlign: "center", // Center text within the fixed width
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      position: "relative",
    });

    // Create spinner element (hidden by default)
    this.spinner = document.createElement("span");
    this.spinner.textContent = "↻"; // Unicode rotating arrow
    Object.assign(this.spinner.style, {
      marginRight: "5px",
      display: "none",
      animation: "spin 1s linear infinite",
    });

    // Add keyframes for spinner animation
    const styleSheet = document.createElement("style");
    styleSheet.textContent = `
      @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
      }
    `;
    document.head.appendChild(styleSheet);

    this.confirmButton.appendChild(this.spinner);

    // Create text span for button label
    this.buttonText = document.createElement("span");
    this.buttonText.textContent = "LOAD DATA";
    this.confirmButton.appendChild(this.buttonText);

    // No mouseover/mouseout events here as we'll control colors based on state
    this.container.appendChild(this.confirmButton);

    parentElement.appendChild(this.container); // Append to the specific parent

    // Add listener for the confirm button
    this.confirmButton.addEventListener(
      "click",
      this.handleConfirmClick.bind(this)
    );

    // Add change event listeners to detect when options change
    this.referenceSelect.addEventListener("change", () => {
      this.optionsChanged = true;
      this.updateConfirmButtonText();
    });

    this.measurementSelect.addEventListener("change", () => {
      this.optionsChanged = true;
      this.updateConfirmButtonText();
    });
  }

  applySelectStyles(selectElement) {
    Object.assign(selectElement.style, {
      padding: "6px 8px", // Reduced padding from 8px 10px
      borderRadius: "3px", // Reduced from 4px
      border: "none",
      backgroundColor: "rgba(40, 40, 40, 0.85)",
      color: "white",
      fontFamily: "sans-serif",
      minWidth: "130px", // Reduced from 150px
      boxSizing: "border-box",
    });
  }

  populateOptions(referenceDates, measurementDates) {
    const firstRef = this.populateSelect(
      this.referenceSelect,
      referenceDates,
      "No references available"
    );
    const firstMeas = this.populateSelect(
      this.measurementSelect,
      measurementDates,
      "No measurements available"
    );

    // Demo mode auto-loading has been removed
    // User must now explicitly click the Load Data button

    // Reset state when new options are loaded
    this.isLoaded = false;
    this.optionsChanged = false;
    this.updateConfirmButtonText();
  }

  populateSelect(selectElement, dateList, emptyText) {
    selectElement.innerHTML = ""; // Clear existing options
    let firstValue = null; // Keep track of the first valid value

    if (!dateList || dateList.length === 0) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = emptyText;
      option.disabled = true;
      selectElement.appendChild(option);
      // return null; // Return null if no valid options
    } else {
      dateList.forEach((date, index) => {
        const option = document.createElement("option");
        option.value = date; // Use date string as value
        option.textContent = date; // Display the date string
        selectElement.appendChild(option);
        if (index === 0) {
          firstValue = date; // Store the first date as the default
        }
      });
    }
    // Set the selected value after populating
    if (firstValue) {
      selectElement.value = firstValue;
    }
    return firstValue; // Return the first valid value found
  }

  handleConfirmClick() {
    const selectedReference = this.referenceSelect.value;
    const selectedMeasurement = this.measurementSelect.value;

    if (selectedReference && selectedMeasurement) {
      console.log(
        `Confirm button clicked. Requesting data for Ref: ${selectedReference}, Meas: ${selectedMeasurement}`
      );

      // Disable button while loading
      this.disableConfirmButton();

      // Notify view for UI updates (especially needed for splash screen)
      if (this.view) {
        this.view.onDataRequested();
      } else {
        // Fallback behavior if view isn't connected
        const splashSpinner = document.getElementById("splash-spinner");
        const splashStatus = document.getElementById("splash-status");
        if (splashSpinner) splashSpinner.style.display = "block";
        if (splashStatus)
          splashStatus.textContent = "Loading data, please wait...";
      }

      // Request the images
      this.requestImages(selectedReference, selectedMeasurement);
    } else {
      console.warn("Please select both a reference and a measurement date.");

      // Show an error message in splash status if available
      const splashStatus = document.getElementById("splash-status");
      if (splashStatus) {
        splashStatus.textContent =
          "Please select both a reference and measurement dataset";
        splashStatus.style.color = "#e74c3c";
      }
    }
  }

  requestImages(referenceDate, measurementDate) {
    console.log(
      `Requesting images for Ref: ${referenceDate}, Meas: ${measurementDate}`
    );
    this.socket.emit("get_images", {
      reference: referenceDate,
      measurement: measurementDate,
    });
  }

  dispose() {
    this.confirmButton.removeEventListener("click", this.handleConfirmClick);
    // No change listeners on selects anymore
    this.container.remove();
  }

  /**
   * Updates the text of the confirm button based on current state
   */
  updateConfirmButtonText() {
    if (!this.confirmButton) return;

    if (this.isLoading) {
      this.buttonText.textContent = "LOADING";
      this.confirmButton.disabled = true;
      this.confirmButton.style.backgroundColor = "#5bc0de"; // Blue for loading
      this.spinner.style.display = "inline-block"; // Show spinner
    } else if (this.isLoaded && !this.optionsChanged) {
      this.buttonText.textContent = "LOADED";
      this.confirmButton.disabled = false;
      this.confirmButton.style.backgroundColor = "#5cb85c"; // Green for loaded
      this.spinner.style.display = "none"; // Hide spinner
    } else {
      this.buttonText.textContent = "LOAD DATA";
      this.confirmButton.disabled = false;
      this.confirmButton.style.backgroundColor = "#d9534f"; // Red for needs loading
      this.spinner.style.display = "none"; // Hide spinner
    }
  }

  /**
   * Mark data as successfully loaded
   */
  markAsLoaded() {
    this.isLoading = false;
    this.isLoaded = true;
    this.optionsChanged = false;
    this.updateConfirmButtonText();
  }

  /**
   * Disable the confirm button during loading
   */
  disableConfirmButton() {
    this.isLoading = true;
    this.updateConfirmButtonText();
  }

  /**
   * Enable the confirm button after loading completes
   */
  enableConfirmButton() {
    this.isLoading = false;
    this.updateConfirmButtonText();
  }

  /**
   * Set a reference to the View for communication
   */
  setView(view) {
    this.view = view;
  }
}
