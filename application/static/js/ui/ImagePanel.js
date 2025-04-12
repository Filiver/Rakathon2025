export class ImagePanel {
  /**
   * Creates an instance of ImagePanel and appends its container to the document body.
   */
  constructor() {
    // Create the main container element
    this.container = document.createElement("div");
    this.container.id = "image-panel-container"; // Assign an ID for potential styling/selection
    document.body.appendChild(this.container); // Append directly to the body

    // Basic styling for side-by-side images
    this.container.style.display = "flex";
    this.container.style.justifyContent = "space-around";
    this.container.style.alignItems = "center";
    this.container.style.flexWrap = "wrap"; // Allow wrapping on smaller screens
    this.container.style.width = "100%"; // Optional: Make it full width
    this.container.style.padding = "10px"; // Optional: Add some padding
    this.container.style.boxSizing = "border-box"; // Include padding in width calculation
  }

  /**
   * Updates the container with two images provided as Data URIs.
   * @param {string} dataUri1 Data URI of the first image.
   * @param {string} dataUri2 Data URI of the second image.
   */
  update(dataUri1, dataUri2) {
    // Clear existing content
    this.container.innerHTML = "";

    // Create and append the first image
    const img1 = document.createElement("img");
    img1.src = dataUri1; // Directly use the Data URI
    img1.alt = "Reference Image"; // Updated alt text
    img1.style.maxWidth = "48%"; // Adjust size as needed
    img1.style.height = "auto";
    img1.style.margin = "1%";
    img1.style.borderRadius = "10px"; // Add rounded corners
    img1.style.boxShadow = "0 4px 8px rgba(0, 0, 0, 0.2)"; // Add drop shadow
    this.container.appendChild(img1);

    // Create and append the second image
    const img2 = document.createElement("img");
    img2.src = dataUri2; // Directly use the Data URI
    img2.alt = "Measurement Image"; // Updated alt text
    img2.style.maxWidth = "48%"; // Adjust size as needed
    img2.style.height = "auto";
    img2.style.margin = "1%";
    img2.style.borderRadius = "10px"; // Add rounded corners
    img2.style.boxShadow = "0 4px 8px rgba(0, 0, 0, 0.2)"; // Add drop shadow
    this.container.appendChild(img2);
  }
}

// Example Usage:
// const imagePanel = new ImagePanel(); // No container ID needed
// Assuming you have dataUri1 and dataUri2 containing 'data:image/png;base64,...' strings
// imagePanel.update(dataUri1, dataUri2);
