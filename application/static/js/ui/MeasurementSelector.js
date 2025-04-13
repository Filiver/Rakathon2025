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
    });

    // --- Reference Selector ---
    const refLabel = document.createElement("label");
    refLabel.textContent = "REFERENCE:"; // ALL CAPS label
    this.container.appendChild(refLabel);

    this.referenceSelect = document.createElement("select");
    this.referenceSelect.id = "reference-selector";
    this.applySelectStyles(this.referenceSelect);
    this.container.appendChild(this.referenceSelect);

    // --- Measurement Selector ---
    const measLabel = document.createElement("label");
    measLabel.textContent = "MEASUREMENT:"; // ALL CAPS label
    this.container.appendChild(measLabel);

    this.measurementSelect = document.createElement("select");
    this.measurementSelect.id = "measurement-selector";
    this.applySelectStyles(this.measurementSelect);
    this.container.appendChild(this.measurementSelect);

    // --- Confirm Button ---
    this.confirmButton = document.createElement("button");
    this.confirmButton.textContent = "LOAD DATA"; // ALL CAPS button text
    Object.assign(this.confirmButton.style, {
      padding: "8px 15px",
      marginLeft: "auto", // Push button to the right if desired, or adjust gap
      backgroundColor: "#5cb85c", // Green color
      color: "white",
      border: "none",
      borderRadius: "4px",
      cursor: "pointer",
      fontSize: "1em",
      whiteSpace: "nowrap", // Prevent button text wrapping
    });
    this.confirmButton.addEventListener(
      "mouseover",
      () => (this.confirmButton.style.backgroundColor = "#4cae4c")
    );
    this.confirmButton.addEventListener(
      "mouseout",
      () => (this.confirmButton.style.backgroundColor = "#5cb85c")
    );
    this.container.appendChild(this.confirmButton);

    parentElement.appendChild(this.container); // Append to the specific parent

    // Add listener for the confirm button
    this.confirmButton.addEventListener(
      "click",
      this.handleConfirmClick.bind(this)
    );
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

    // --- Demo Mode Fix ---
    // Automatically load the first available combination if both exist
    if (firstRef && firstMeas) {
      console.log(
        `Demo mode: Automatically loading first combination - Ref: ${firstRef}, Meas: ${firstMeas}`
      );
      this.requestImages(firstRef, firstMeas);
    } else {
      console.log(
        "Demo mode: Not loading data automatically as reference or measurement dates are missing."
      );
    }
    // --- End Demo Mode Fix ---
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

      // Request the images
      this.requestImages(selectedReference, selectedMeasurement);
    } else {
      console.warn("Please select both a reference and a measurement date.");
      // Optionally show a message to the user
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
   * Disables the confirm button and shows a loading spinner
   */
  disableConfirmButton() {
    if (this.confirmButton) {
      this.confirmButton.disabled = true;

      // Store original button text
      if (!this.confirmButton._originalText) {
        this.confirmButton._originalText = this.confirmButton.innerHTML;
      }

      // Replace with loading spinner
      this.confirmButton.innerHTML =
        '<span class="loading-spinner">⟳</span> Loading...';

      // Make sure spinner styles exist
      if (!document.getElementById("spinner-animation")) {
        const style = document.createElement("style");
        style.id = "spinner-animation";
        style.textContent = `
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
          .loading-spinner {
            display: inline-block;
            animation: spin 1.5s linear infinite;
          }
        `;
        document.head.appendChild(style);
      }
    }
  }

  /**
   * Re-enables the confirm button and restores its original text
   */
  enableConfirmButton() {
    if (this.confirmButton) {
      this.confirmButton.disabled = false;

      // Restore original text if available
      if (this.confirmButton._originalText) {
        this.confirmButton.innerHTML = this.confirmButton._originalText;
      } else {
        this.confirmButton.innerHTML = "LOAD DATA";
      }
    }
  }
}
