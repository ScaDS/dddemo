# Transport Mode Recognition Application

This directory contains all assets for the Transport Mode Recognition use case demonstration.

## Structure

- `index.html` - Main content with bullet points explaining the challenge and solution
- `slider.js` - Interactive slider functionality with automatic animation
- `user1_handsVSbag_grey.png` - Base visualization image (greyscale)
- `user1_handsVSbag_noLegend.png` - Overlay visualization image (color, no legend)

## Features

### Interactive Image Overlay
The visualization uses two overlaid images with an opacity slider that allows users to transition between them:
- Base image shows the greyscale version
- Overlay image shows the colored version
- Slider controls the opacity of the overlay (0-100%)

### Automatic Animation
When the tab is first displayed or becomes visible:
- The slider automatically animates to demonstrate its functionality
- Animation cycles forward and backward twice
- After animation completes, the slider resets to the starting position
- Users can then manually control the slider

## Technical Details

The `slider.js` script:
- Uses IntersectionObserver to detect when the content becomes visible
- Animates the slider smoothly using requestAnimationFrame
- Prevents re-animation on subsequent views
- Maintains manual control after animation completes
