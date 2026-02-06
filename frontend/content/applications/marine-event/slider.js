// Marine Event Slider Animation and Control
// This script handles the opacity slider for the marine event visualization

(function() {
  'use strict';
  
  // Initialize the slider when the content is loaded
  function initMarineEventSlider() {
    // Find the marine-event slider container specifically
    const containers = document.querySelectorAll('[data-animate-slider="marine-event"]');
    
    containers.forEach(container => {
      // Skip if this container was already initialized
      if (container.dataset.sliderInitialized) return;
      container.dataset.sliderInitialized = 'true';
    
    const slider = container.querySelector('.opacity-slider');
    const overlayImage = container.querySelector('.overlay-image');
    const baseImage = container.querySelector('.base-image');
    
    if (!slider || !overlayImage || !baseImage) return;
    
    // Track if animation is currently running
    let isAnimating = false;
    
    // Add event listener for manual slider control
    slider.addEventListener('input', function() {
      const value = this.value / 100;
      overlayImage.style.opacity = value;
      // Reduce base image opacity slightly when overlay is shown (from 1.0 to 0.6)
      baseImage.style.opacity = 1 - (value * 0.4);
    });
    
    // Animation function to demonstrate slider functionality
    function animateSlider() {
      if (isAnimating) return;
      isAnimating = true;
      
      let progress = 0;
      let direction = 1; // 1 for forward, -1 for backward
      let cycles = 0;
      const maxCycles = 2; // Go forward and backward twice
      const animationSpeed = 2; // Speed of animation
      
      function animate() {
        progress += direction * animationSpeed;
        
        // Change direction at boundaries
        if (progress >= 100) {
          progress = 100;
          direction = -1;
          cycles++;
        } else if (progress <= 0) {
          progress = 0;
          direction = 1;
          cycles++;
        }
        
        // Update slider and images
        const value = progress / 100;
        slider.value = progress;
        overlayImage.style.opacity = value;
        baseImage.style.opacity = 1 - (value * 0.4);
        
        // Continue animation until we've completed the cycles
        if (cycles < maxCycles) {
          requestAnimationFrame(animate);
        } else {
          // Reset to start position after animation
          slider.value = 0;
          overlayImage.style.opacity = 0;
          baseImage.style.opacity = 1;
          isAnimating = false;
        }
      }
      
      // Start animation after a short delay
      setTimeout(() => {
        requestAnimationFrame(animate);
      }, 500);
    }
    
    // Find the parent slide element to observe
    const slideElement = container.closest('.slider-slide');
    if (!slideElement) return;
    
    // Trigger animation when this slide becomes visible
    // We'll use an IntersectionObserver to detect when the element is visible
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          // Trigger animation whenever the slide becomes visible
          animateSlider();
        }
      });
    }, {
      threshold: 0.5 // Trigger when 50% of the element is visible
    });
    
    observer.observe(slideElement);
    });
  }
  
  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMarineEventSlider);
  } else {
    initMarineEventSlider();
  }
})();
