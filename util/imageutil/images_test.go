package imageutil

import (
	"image"
	"image/color"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// createTestImage creates a solid color test image with given dimensions.
func createTestImage(width, height int, c color.Color) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, c)
		}
	}
	return img
}

// createTestImageWithContent creates an image with a colored rectangle in the center.
// Used for testing margin cropping.
func createTestImageWithContent(width, height, contentX, contentY, contentW, contentH int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	white := color.RGBA{255, 255, 255, 255}
	black := color.RGBA{0, 0, 0, 255}

	// Fill with white (margin)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, white)
		}
	}

	// Draw black content rectangle
	for y := contentY; y < contentY+contentH; y++ {
		for x := contentX; x < contentX+contentW; x++ {
			if x >= 0 && x < width && y >= 0 && y < height {
				img.Set(x, y, black)
			}
		}
	}

	return img
}

func TestAlignLongAxisStep_LandscapeToPortrait(t *testing.T) {
	// Create a landscape image (wider than tall)
	landscape := createTestImage(200, 100, color.RGBA{255, 0, 0, 255})

	step := AlignLongAxisStep()
	result, err := step.Apply(landscape)

	require.NoError(t, err)
	assert.Equal(t, 100, result.Bounds().Dx(), "width should be 100 after rotation")
	assert.Equal(t, 200, result.Bounds().Dy(), "height should be 200 after rotation")
}

func TestAlignLongAxisStep_PortraitUnchanged(t *testing.T) {
	// Create a portrait image (taller than wide)
	portrait := createTestImage(100, 200, color.RGBA{0, 255, 0, 255})

	step := AlignLongAxisStep()
	result, err := step.Apply(portrait)

	require.NoError(t, err)
	// Portrait images should not be rotated
	assert.Equal(t, 100, result.Bounds().Dx(), "width should remain 100")
	assert.Equal(t, 200, result.Bounds().Dy(), "height should remain 200")
}

func TestAlignLongAxisStep_SquareUnchanged(t *testing.T) {
	// Create a square image
	square := createTestImage(100, 100, color.RGBA{0, 0, 255, 255})

	step := AlignLongAxisStep()
	result, err := step.Apply(square)

	require.NoError(t, err)
	// Square images should not be rotated
	assert.Equal(t, 100, result.Bounds().Dx(), "width should remain 100")
	assert.Equal(t, 100, result.Bounds().Dy(), "height should remain 100")
}

func TestPadToSizeStep_SmallerImage(t *testing.T) {
	// Create a tall image that will need horizontal padding
	// 50x200 image into 100x100 target:
	// - scaleW = 100/50 = 2.0, scaleH = 100/200 = 0.5
	// - scale = min(2.0, 0.5) = 0.5
	// - newW = 50*0.5 = 25, newH = 200*0.5 = 100
	// - horizontal padding = (100-25)/2 = 37.5 pixels each side
	tall := createTestImage(50, 200, color.RGBA{255, 0, 0, 255})

	step := PadToSizeStep(100, 100, 1.0) // White background
	result, err := step.Apply(tall)

	require.NoError(t, err)
	assert.Equal(t, 100, result.Bounds().Dx(), "width should be 100")
	assert.Equal(t, 100, result.Bounds().Dy(), "height should be 100")

	// Check that left corner is white (padding area)
	r, g, b, _ := result.At(0, 50).RGBA()
	assert.Equal(t, uint32(0xffff), r, "left side should be white (padding)")
	assert.Equal(t, uint32(0xffff), g, "left side should be white (padding)")
	assert.Equal(t, uint32(0xffff), b, "left side should be white (padding)")

	// Check that center has content (red)
	r, g, b, _ = result.At(50, 50).RGBA()
	assert.Equal(t, uint32(0xffff), r, "center should be red (content)")
	assert.Equal(t, uint32(0), g, "center should be red (content)")
	assert.Equal(t, uint32(0), b, "center should be red (content)")
}

func TestPadToSizeStep_BlackPadding(t *testing.T) {
	// Create a wide image that will need vertical padding
	// 200x50 image into 100x100 target:
	// - scaleW = 100/200 = 0.5, scaleH = 100/50 = 2.0
	// - scale = min(0.5, 2.0) = 0.5
	// - newW = 200*0.5 = 100, newH = 50*0.5 = 25
	// - vertical padding = (100-25)/2 = 37.5 pixels each side
	wide := createTestImage(200, 50, color.RGBA{255, 0, 0, 255})

	step := PadToSizeStep(100, 100, 0.0) // Black background
	result, err := step.Apply(wide)

	require.NoError(t, err)
	assert.Equal(t, 100, result.Bounds().Dx(), "width should be 100")
	assert.Equal(t, 100, result.Bounds().Dy(), "height should be 100")

	// Check that top corner is black (padding area)
	r, g, b, _ := result.At(0, 0).RGBA()
	assert.Equal(t, uint32(0), r, "top corner should be black (padding)")
	assert.Equal(t, uint32(0), g, "top corner should be black (padding)")
	assert.Equal(t, uint32(0), b, "top corner should be black (padding)")

	// Check that center has content (red)
	r, g, b, _ = result.At(50, 50).RGBA()
	assert.Equal(t, uint32(0xffff), r, "center should be red (content)")
	assert.Equal(t, uint32(0), g, "center should be red (content)")
	assert.Equal(t, uint32(0), b, "center should be red (content)")
}

func TestPadToSizeStep_PreservesAspectRatio(t *testing.T) {
	// Create a wide image (2:1 aspect ratio)
	wide := createTestImage(200, 100, color.RGBA{0, 255, 0, 255})

	step := PadToSizeStep(100, 100, 1.0)
	result, err := step.Apply(wide)

	require.NoError(t, err)
	assert.Equal(t, 100, result.Bounds().Dx())
	assert.Equal(t, 100, result.Bounds().Dy())

	// The scaled image should be 100x50 (maintaining 2:1 ratio)
	// Centered in 100x100, so content at y=25 to y=75
	// Top should be white padding
	r, g, b, _ := result.At(50, 10).RGBA()
	assert.Equal(t, uint32(0xffff), r, "top should be white padding")
	assert.Equal(t, uint32(0xffff), g, "top should be white padding")
	assert.Equal(t, uint32(0xffff), b, "top should be white padding")

	// Center should be green (the content)
	r, g, b, _ = result.At(50, 50).RGBA()
	assert.Equal(t, uint32(0), r, "center should be green content")
	assert.Equal(t, uint32(0xffff), g, "center should be green content")
	assert.Equal(t, uint32(0), b, "center should be green content")
}

func TestCropMarginStep_RemovesWhiteMargins(t *testing.T) {
	// Create image with white margins around black content
	// 100x100 image with 20x20 black square at (40,40)
	img := createTestImageWithContent(100, 100, 40, 40, 20, 20)

	step := CropMarginStep(250, 0) // High threshold, no min margin
	result, err := step.Apply(img)

	require.NoError(t, err)
	// Should crop to just the black content (20x20)
	assert.Equal(t, 20, result.Bounds().Dx(), "should crop to content width")
	assert.Equal(t, 20, result.Bounds().Dy(), "should crop to content height")
}

func TestCropMarginStep_WithMinMargin(t *testing.T) {
	// Create image with white margins around black content
	img := createTestImageWithContent(100, 100, 40, 40, 20, 20)

	step := CropMarginStep(250, 5) // Keep 5px margin
	result, err := step.Apply(img)

	require.NoError(t, err)
	// Should crop to content + margin (20 + 5*2 = 30)
	assert.Equal(t, 30, result.Bounds().Dx(), "should include margin")
	assert.Equal(t, 30, result.Bounds().Dy(), "should include margin")
}

func TestCropMarginStep_NoContent(t *testing.T) {
	// Create all-white image
	white := createTestImage(100, 100, color.RGBA{255, 255, 255, 255})

	step := CropMarginStep(250, 0)
	result, err := step.Apply(white)

	require.NoError(t, err)
	// Should return original since no content found
	assert.Equal(t, 100, result.Bounds().Dx())
	assert.Equal(t, 100, result.Bounds().Dy())
}

func TestSimpleNormalizationStep(t *testing.T) {
	step := SimpleNormalizationStep()

	// Test that it returns the expected mean/std values
	assert.Equal(t, [3]float32{0.5, 0.5, 0.5}, step.mean)
	assert.Equal(t, [3]float32{0.5, 0.5, 0.5}, step.std)
}

func TestResizeToExactStep(t *testing.T) {
	// Create a 200x100 image
	img := createTestImage(200, 100, color.RGBA{255, 0, 0, 255})

	step := ResizeToExactStep(50, 50)
	result, err := step.Apply(img)

	require.NoError(t, err)
	assert.Equal(t, 50, result.Bounds().Dx(), "should resize to exact width")
	assert.Equal(t, 50, result.Bounds().Dy(), "should resize to exact height")
}

func TestPreprocessingChain(t *testing.T) {
	// Test a realistic preprocessing chain for Donut
	// Start with a 300x200 landscape document image
	img := createTestImage(300, 200, color.RGBA{100, 100, 100, 255})

	// Apply Donut-style preprocessing chain
	steps := []PreprocessStep{
		AlignLongAxisStep(),            // Rotate to portrait
		PadToSizeStep(384, 384, 1.0),   // Pad to square
	}

	result := image.Image(img)
	var err error
	for _, step := range steps {
		result, err = step.Apply(result)
		require.NoError(t, err)
	}

	// After rotation: 200x300 (portrait)
	// After padding: 384x384 (square)
	assert.Equal(t, 384, result.Bounds().Dx())
	assert.Equal(t, 384, result.Bounds().Dy())
}
