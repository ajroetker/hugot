package imageutil

import (
	"bytes"
	"image"
	"image/color"

	_ "image/gif"  // adds gif support
	_ "image/jpeg" // adds jpeg support
	_ "image/png"  // adds png support

	"github.com/knights-analytics/hugot/util/fileutil"
	_ "golang.org/x/image/webp" // adds webp support
)

func LoadImagesFromPaths(paths []string) ([]image.Image, error) {
	images := make([]image.Image, 0, len(paths))

	for _, path := range paths {
		b, err := fileutil.ReadFileBytes(path)
		if err != nil {
			return nil, err
		}
		img, _, err := image.Decode(bytes.NewReader(b))
		if err != nil {
			return nil, err
		}
		images = append(images, img)
	}
	return images, nil
}

type PreprocessStep interface {
	Apply(img image.Image) (image.Image, error)
}

type ResizePreprocessor struct {
	targetSize int
}

func ResizeStep(targetSize int) *ResizePreprocessor {
	return &ResizePreprocessor{targetSize: targetSize}
}

func (s *ResizePreprocessor) Apply(img image.Image) (image.Image, error) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	var newW, newH int
	if w < h {
		newW = s.targetSize
		newH = int(float32(h) * float32(s.targetSize) / float32(w))
	} else {
		newH = s.targetSize
		newW = int(float32(w) * float32(s.targetSize) / float32(h))
	}
	return resizeImage(img, newW, newH), nil
}

func CenterCropStep(targetWidth, targetHeight int) *CenterCropPreprocessor {
	return &CenterCropPreprocessor{targetWidth: targetWidth, targetHeight: targetHeight}
}

type CenterCropPreprocessor struct {
	targetWidth  int
	targetHeight int
}

func (s *CenterCropPreprocessor) Apply(img image.Image) (image.Image, error) {
	bounds := img.Bounds()
	x0 := bounds.Min.X + (bounds.Dx()-s.targetWidth)/2
	y0 := bounds.Min.Y + (bounds.Dy()-s.targetHeight)/2
	rect := image.Rect(0, 0, s.targetWidth, s.targetHeight)
	dst := image.NewRGBA(rect)
	for y := 0; y < s.targetHeight; y++ {
		for x := 0; x < s.targetWidth; x++ {
			dst.Set(x, y, img.At(x0+x, y0+y))
		}
	}
	return dst, nil
}

type NormalizationStep interface {
	Apply(r, g, b float32) (float32, float32, float32)
}

type PixelNormalizationPreprocessor struct {
	mean [3]float32
	std  [3]float32
}

func (s *PixelNormalizationPreprocessor) Apply(r, g, b float32) (float32, float32, float32) {
	r = (r - s.mean[0]) / s.std[0]
	g = (g - s.mean[1]) / s.std[1]
	b = (b - s.mean[2]) / s.std[2]
	return r, g, b
}

func PixelNormalizationStep(mean, std [3]float32) *PixelNormalizationPreprocessor {
	return &PixelNormalizationPreprocessor{mean: mean, std: std}
}

func ImagenetPixelNormalizationStep() *PixelNormalizationPreprocessor {
	return &PixelNormalizationPreprocessor{
		mean: [3]float32{0.485, 0.456, 0.406},
		std:  [3]float32{0.229, 0.224, 0.225},
	}
}

// CLIPPixelNormalizationStep returns CLIP's normalization values.
// Use after RescaleStep() to normalize to 0-1 range first.
func CLIPPixelNormalizationStep() *PixelNormalizationPreprocessor {
	return &PixelNormalizationPreprocessor{
		mean: [3]float32{0.48145466, 0.4578275, 0.40821073},
		std:  [3]float32{0.26862954, 0.26130258, 0.27577711},
	}
}

type RescalePreprocessor struct{}

func (s *RescalePreprocessor) Apply(r, g, b float32) (float32, float32, float32) {
	scale := float32(1.0 / 255.0)
	return r * scale, g * scale, b * scale
}

func RescaleStep() *RescalePreprocessor {
	return &RescalePreprocessor{}
}

// ResizeToExactStep resizes an image to exact width and height dimensions.
// Unlike ResizeStep which preserves aspect ratio, this resizes to the exact size.
type ResizeToExactPreprocessor struct {
	width  int
	height int
}

func ResizeToExactStep(width, height int) *ResizeToExactPreprocessor {
	return &ResizeToExactPreprocessor{width: width, height: height}
}

func (s *ResizeToExactPreprocessor) Apply(img image.Image) (image.Image, error) {
	return resizeImage(img, s.width, s.height), nil
}

// resizeImage resizes an image to the given width and height using nearest neighbor (simple, replace with better if needed).
func resizeImage(img image.Image, newW, newH int) image.Image {
	dst := image.NewRGBA(image.Rect(0, 0, newW, newH))
	srcBounds := img.Bounds()
	for y := range newH {
		for x := range newW {
			srcX := srcBounds.Min.X + x*srcBounds.Dx()/newW
			srcY := srcBounds.Min.Y + y*srcBounds.Dy()/newH
			dst.Set(x, y, img.At(srcX, srcY))
		}
	}
	return dst
}

// AlignLongAxisPreprocessor rotates image so the long axis is vertical (portrait).
// Used by Donut for consistent document orientation.
type AlignLongAxisPreprocessor struct{}

// AlignLongAxisStep creates a preprocessor that rotates landscape images to portrait.
func AlignLongAxisStep() *AlignLongAxisPreprocessor {
	return &AlignLongAxisPreprocessor{}
}

func (s *AlignLongAxisPreprocessor) Apply(img image.Image) (image.Image, error) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	// If landscape (wider than tall), rotate 90 degrees clockwise
	if w > h {
		return rotateImage90CW(img), nil
	}
	return img, nil
}

// rotateImage90CW rotates an image 90 degrees clockwise.
func rotateImage90CW(img image.Image) image.Image {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, h, w)) // Swap dimensions

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			// Rotate 90 degrees clockwise: (x, y) -> (h-1-y, x)
			dst.Set(h-1-y, x, img.At(bounds.Min.X+x, bounds.Min.Y+y))
		}
	}
	return dst
}

// PadToSizePreprocessor pads image to target dimensions with a fill color.
// Maintains aspect ratio and centers the image.
type PadToSizePreprocessor struct {
	width    int
	height   int
	padValue float32 // 0.0 = black, 1.0 = white
}

// PadToSizeStep creates a preprocessor that pads images to target size.
// padValue: 0.0 = black background, 1.0 = white background
func PadToSizeStep(width, height int, padValue float32) *PadToSizePreprocessor {
	return &PadToSizePreprocessor{
		width:    width,
		height:   height,
		padValue: padValue,
	}
}

func (s *PadToSizePreprocessor) Apply(img image.Image) (image.Image, error) {
	bounds := img.Bounds()
	srcW, srcH := bounds.Dx(), bounds.Dy()

	// Calculate scale to fit within target (preserve aspect ratio)
	scaleW := float64(s.width) / float64(srcW)
	scaleH := float64(s.height) / float64(srcH)
	scale := scaleW
	if scaleH < scaleW {
		scale = scaleH
	}

	// New dimensions after scaling
	newW := int(float64(srcW) * scale)
	newH := int(float64(srcH) * scale)

	// Resize the image
	resized := resizeImage(img, newW, newH)

	// Create padded canvas with fill color
	padColor := uint8(s.padValue * 255)
	dst := image.NewRGBA(image.Rect(0, 0, s.width, s.height))

	// Fill with pad color
	fillColor := color.RGBA{padColor, padColor, padColor, 255}
	for y := 0; y < s.height; y++ {
		for x := 0; x < s.width; x++ {
			dst.Set(x, y, fillColor)
		}
	}

	// Center the resized image
	offsetX := (s.width - newW) / 2
	offsetY := (s.height - newH) / 2

	resizedBounds := resized.Bounds()
	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			dst.Set(offsetX+x, offsetY+y, resized.At(resizedBounds.Min.X+x, resizedBounds.Min.Y+y))
		}
	}

	return dst, nil
}

// CropMarginPreprocessor removes white margins from document images.
// Used by Nougat for cleaner document processing.
type CropMarginPreprocessor struct {
	threshold uint8 // Pixel value above which is considered margin (default: 250)
	minMargin int   // Minimum margin to keep around content (default: 0)
}

// CropMarginStep creates a preprocessor that removes white margins.
// threshold: pixel values above this are considered margin (250 = near-white)
// minMargin: minimum margin to keep around the content
func CropMarginStep(threshold uint8, minMargin int) *CropMarginPreprocessor {
	return &CropMarginPreprocessor{
		threshold: threshold,
		minMargin: minMargin,
	}
}

func (s *CropMarginPreprocessor) Apply(img image.Image) (image.Image, error) {
	bounds := img.Bounds()

	// Find content bounds (non-white regions)
	minX, minY := bounds.Max.X, bounds.Max.Y
	maxX, maxY := bounds.Min.X, bounds.Min.Y

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			// Check if pixel is below threshold (content, not margin)
			// RGBA() returns 16-bit values, shift to 8-bit
			if uint8(r>>8) < s.threshold || uint8(g>>8) < s.threshold || uint8(b>>8) < s.threshold {
				if x < minX {
					minX = x
				}
				if y < minY {
					minY = y
				}
				if x > maxX {
					maxX = x
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}

	// If no content found, return original
	if minX >= maxX || minY >= maxY {
		return img, nil
	}

	// Expand by min margin
	minX = max(bounds.Min.X, minX-s.minMargin)
	minY = max(bounds.Min.Y, minY-s.minMargin)
	maxX = min(bounds.Max.X, maxX+s.minMargin+1)
	maxY = min(bounds.Max.Y, maxY+s.minMargin+1)

	// Crop to content bounds
	newW := maxX - minX
	newH := maxY - minY

	if newW <= 0 || newH <= 0 {
		return img, nil
	}

	dst := image.NewRGBA(image.Rect(0, 0, newW, newH))
	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			dst.Set(x, y, img.At(minX+x, minY+y))
		}
	}

	return dst, nil
}

// SimpleNormalizationStep returns simple normalization with mean=0.5 and std=0.5 for all channels.
// This maps pixel values from [0, 1] to [-1, 1].
func SimpleNormalizationStep() *PixelNormalizationPreprocessor {
	return &PixelNormalizationPreprocessor{
		mean: [3]float32{0.5, 0.5, 0.5},
		std:  [3]float32{0.5, 0.5, 0.5},
	}
}
