// Transport Mode Button Animation and Control
// This script handles the button toggle for the transport mode visualization

(function() {
  'use strict';
  
  let initialized = false;
  
  // Initialize the button when the content is loaded
  function initTransportModeButton() {
    if (initialized) return;
    
    const container = document.querySelector('[data-animate-button="true"]');
    if (!container) {
      console.log('Transport mode: container not found, will retry...');
      return;
    }
    
    const button = container.querySelector('.toggle-overlay-btn');
    const overlayImage = container.querySelector('.overlay-image');
    
    if (!button) {
      console.log('Transport mode: button not found');
      return;
    }
    if (!overlayImage) {
      console.log('Transport mode: overlay image not found');
      return;
    }
    
    initialized = true;
    console.log('Transport mode: initialized successfully');
    
    // Track overlay state
    let isOverlayVisible = false;
    
    // Add event listener for button click
    button.addEventListener('click', function(e) {
      e.preventDefault();
      isOverlayVisible = !isOverlayVisible;
      overlayImage.style.opacity = isOverlayVisible ? 1 : 0;
      button.textContent = isOverlayVisible ? 'Hide Overlay' : 'Show Overlay';
      console.log('Button clicked, overlay visible:', isOverlayVisible);
    });
    
    // Animation function to demonstrate button functionality
    function animateButton() {
      setTimeout(() => {
        isOverlayVisible = true;
        overlayImage.style.opacity = 1;
        button.textContent = 'Hide Overlay';
        
        setTimeout(() => {
          isOverlayVisible = false;
          overlayImage.style.opacity = 0;
          button.textContent = 'Show Overlay';
        }, 2000);
      }, 500);
    }
    
    // Find the parent slide element to observe (optional)
    const slideElement = container.closest('.slider-slide');
    
    if (slideElement) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            animateButton();
          }
        });
      }, {
        threshold: 0.5
      });
      
      observer.observe(slideElement);
    }
  }
  
  // Try to initialize immediately
  initTransportModeButton();
  
  // Set up a MutationObserver to watch for the button being added to the DOM
  const observer = new MutationObserver(() => {
    if (!initialized) {
      initTransportModeButton();
    }
  });
  
  // Start observing the document body for changes
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
  
  // Also try on various load events
  document.addEventListener('DOMContentLoaded', initTransportModeButton);
  window.addEventListener('load', initTransportModeButton);
  
  // Stop observing after 10 seconds to avoid memory leaks
  setTimeout(() => {
    observer.disconnect();
  }, 10000);
})();
