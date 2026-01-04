package backends

import (
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"strings"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/fileutil"
)

// Vision2SeqPipelineInterface defines the interface for vision-to-sequence pipeline access.
// This is used by encoder-decoder models like TrOCR, Donut, Florence-2, and Nougat where input is an image and output is text.
type Vision2SeqPipelineInterface interface {
	GetEncoderModel() *Model
	GetDecoderModel() *Model
	GetDecoderInitModel() *Model     // Returns init decoder (nil if using merged)
	GetDecoderWithPastModel() *Model // Returns with-past decoder (nil if using merged)
	UseSplitDecoder() bool           // True if using split decoder (init + with_past)
	GetTokenizer() *Tokenizer
	GetRuntime() string
	GetMaxNewTokens() int
	GetDoSample() bool
	GetTopP() float32
	GetTemperature() float32
	GetRepetitionPenalty() float32
	GetDecoderStartTokenID() int64
	GetEosTokenIDs() map[int64]bool
	GetPadTokenID() int64
	GetVocabSize() int
	GetImageSize() int
	GetImageWidth() int  // For non-square images (Donut, Nougat)
	GetImageHeight() int // For non-square images (Donut, Nougat)
	GetNumHeads() int
	GetHeadDim() int
	GetModelType() string    // "trocr", "donut", "florence", "nougat"
	GetOutputFormat() string // "text", "json", "markdown"

	// Florence-2 specific models (separate vision encoder architecture)
	GetVisionEncoderModel() *Model  // Returns vision_encoder model (nil if not Florence-2)
	GetEmbedTokensModel() *Model    // Returns embed_tokens model (nil if not Florence-2)
	HasSeparateVisionEncoder() bool // True if using Florence-2 style separate vision encoder
}

// Vision2SeqBatchInterface defines the interface for vision2seq batch access.
type Vision2SeqBatchInterface interface {
	GetSize() int
	GetImages() []image.Image
	GetPreprocessedImages() [][][][]float32
	SetEncoderHiddenStates(states any)
	GetEncoderHiddenStates() any
	SetPastKeyValues(pkv []any)
	GetPastKeyValues() []any
	SetLogits(logits any)
	GetLogits() any
	GetGeneratedTokens() [][]int64
	SetGeneratedTokens(tokens [][]int64)
	GetFinished() []bool
	SetFinished(finished []bool)
	GetFinishedCount() int
	SetFinishedCount(count int)
	SetDestroyEncoder(fn func() error)
	SetDestroyDecoder(fn func() error)
	// Prompt support for DocVQA and other task-prompted models
	GetPromptTokenIDs() [][]int64
	SetPromptTokenIDs(tokens [][]int64)
}

// Vision2SeqConfig holds configuration for vision-to-sequence models (TrOCR, Donut, Nougat, etc.).
type Vision2SeqConfig struct {
	// Vision encoder config
	ImageSize   int
	NumChannels int

	// Extended dimensions for non-square images (Donut, Nougat)
	ImageWidth  int // For non-square images (default: ImageSize)
	ImageHeight int // For non-square images (default: ImageSize)

	// Decoder config
	DecoderStartTokenID int64
	EosTokenIDs         map[int64]bool
	PadTokenID          int64
	VocabSize           int

	// Model architecture
	HiddenSize            int
	NumDecoderLayers      int
	NumHeads              int
	HeadDim               int
	MaxPositionEmbeddings int // Max output length (Nougat: 4096, TrOCR: ~512)

	// Image preprocessing (from preprocessor_config.json)
	ImageMean []float32 // Per-channel normalization mean
	ImageStd  []float32 // Per-channel normalization std

	// Preprocessing flags (from preprocessor_config.json)
	DoAlignLongAxis bool    // Rotate to portrait orientation (Donut)
	DoPad           bool    // Pad to target size
	DoCropMargin    bool    // Remove document margins (Nougat)
	DoThumbnail     bool    // Create thumbnail (Donut)
	PadValue        float32 // Background color for padding (default: 1.0 = white)

	// Model type detection
	ModelType    string // "trocr", "donut", "florence", "nougat", "auto"
	OutputFormat string // "text", "json", "markdown"
}

// I/O names are now defined in vision2seq_models.go via GetModelProfile() and GetFlorenceIONames()

// LoadVision2SeqEncoder loads the vision encoder model for vision2seq inference.
// Looks for encoder_model.onnx or *encoder*.onnx in the model path.
func LoadVision2SeqEncoder(modelPath string, opts *options.Options) (*Model, error) {
	// First check in onnx/ subdirectory (common for HuggingFace models)
	onnxDir := fileutil.PathJoinSafe(modelPath, "onnx")
	if exists, _ := fileutil.FileExists(onnxDir); exists {
		modelPath = onnxDir
	}

	onnxFile, err := findVision2SeqOnnxFile(modelPath, "encoder")
	if err != nil {
		return nil, err
	}

	model := &Model{
		Path:         modelPath,
		OnnxFilename: onnxFile,
		Pipelines:    make(map[string]Pipeline),
	}

	if err := LoadOnnxModelBytes(model); err != nil {
		return nil, err
	}

	// Use predefined input/output names to avoid metadata extraction which
	// creates a temporary session without our session options (e.g., disabled
	// graph optimizations for model compatibility)
	profile := DefaultModelProfile()
	if opts.Backend == "ORT" {
		if err := CreateORTModelBackendWithNames(model, opts, profile.EncoderInputs, profile.EncoderOutputs); err != nil {
			return nil, err
		}
	} else {
		if err := CreateModelBackend(model, opts); err != nil {
			return nil, err
		}
	}

	model.Destroy = func() error {
		switch opts.Backend {
		case "ORT":
			if model.ORTModel != nil {
				return model.ORTModel.Destroy()
			}
		case "GO", "XLA":
			if model.GoMLXModel != nil {
				model.GoMLXModel.Destroy()
			}
		}
		return nil
	}

	return model, nil
}

// LoadVision2SeqVisionEncoder loads the separate vision encoder for Florence-2.
// Looks for vision_encoder.onnx in the model path.
func LoadVision2SeqVisionEncoder(modelPath string, opts *options.Options) (*Model, error) {
	// First check in onnx/ subdirectory (common for HuggingFace models)
	onnxDir := fileutil.PathJoinSafe(modelPath, "onnx")
	if exists, _ := fileutil.FileExists(onnxDir); exists {
		modelPath = onnxDir
	}

	onnxFile, err := findVision2SeqOnnxFile(modelPath, "vision_encoder")
	if err != nil {
		return nil, err
	}

	model := &Model{
		Path:         modelPath,
		OnnxFilename: onnxFile,
		Pipelines:    make(map[string]Pipeline),
	}

	if err := LoadOnnxModelBytes(model); err != nil {
		return nil, err
	}

	florenceIO := GetFlorenceIONames()
	if opts.Backend == "ORT" {
		if err := CreateORTModelBackendWithNames(model, opts, florenceIO.VisionEncoderInputs, florenceIO.VisionEncoderOutputs); err != nil {
			return nil, err
		}
	} else {
		if err := CreateModelBackend(model, opts); err != nil {
			return nil, err
		}
	}

	model.Destroy = func() error {
		switch opts.Backend {
		case "ORT":
			if model.ORTModel != nil {
				return model.ORTModel.Destroy()
			}
		case "GO", "XLA":
			if model.GoMLXModel != nil {
				model.GoMLXModel.Destroy()
			}
		}
		return nil
	}

	return model, nil
}

// LoadVision2SeqEmbedTokens loads the embed_tokens model for Florence-2.
// Looks for embed_tokens.onnx in the model path.
func LoadVision2SeqEmbedTokens(modelPath string, opts *options.Options) (*Model, error) {
	// First check in onnx/ subdirectory (common for HuggingFace models)
	onnxDir := fileutil.PathJoinSafe(modelPath, "onnx")
	if exists, _ := fileutil.FileExists(onnxDir); exists {
		modelPath = onnxDir
	}

	onnxFile, err := findVision2SeqOnnxFile(modelPath, "embed_tokens")
	if err != nil {
		return nil, err
	}

	model := &Model{
		Path:         modelPath,
		OnnxFilename: onnxFile,
		Pipelines:    make(map[string]Pipeline),
	}

	if err := LoadOnnxModelBytes(model); err != nil {
		return nil, err
	}

	florenceIO := GetFlorenceIONames()
	if opts.Backend == "ORT" {
		if err := CreateORTModelBackendWithNames(model, opts, florenceIO.EmbedTokensInputs, florenceIO.EmbedTokensOutputs); err != nil {
			return nil, err
		}
	} else {
		if err := CreateModelBackend(model, opts); err != nil {
			return nil, err
		}
	}

	model.Destroy = func() error {
		switch opts.Backend {
		case "ORT":
			if model.ORTModel != nil {
				return model.ORTModel.Destroy()
			}
		case "GO", "XLA":
			if model.GoMLXModel != nil {
				model.GoMLXModel.Destroy()
			}
		}
		return nil
	}

	return model, nil
}

// LoadVision2SeqFlorenceEncoder loads the text encoder for Florence-2.
// This differs from the standard encoder as it takes inputs_embeds instead of pixel_values.
func LoadVision2SeqFlorenceEncoder(modelPath string, opts *options.Options) (*Model, error) {
	// First check in onnx/ subdirectory (common for HuggingFace models)
	onnxDir := fileutil.PathJoinSafe(modelPath, "onnx")
	if exists, _ := fileutil.FileExists(onnxDir); exists {
		modelPath = onnxDir
	}

	onnxFile, err := findVision2SeqOnnxFile(modelPath, "encoder")
	if err != nil {
		return nil, err
	}

	model := &Model{
		Path:         modelPath,
		OnnxFilename: onnxFile,
		Pipelines:    make(map[string]Pipeline),
	}

	if err := LoadOnnxModelBytes(model); err != nil {
		return nil, err
	}

	florenceIO := GetFlorenceIONames()
	if opts.Backend == "ORT" {
		if err := CreateORTModelBackendWithNames(model, opts, florenceIO.EncoderInputs, florenceIO.EncoderOutputs); err != nil {
			return nil, err
		}
	} else {
		if err := CreateModelBackend(model, opts); err != nil {
			return nil, err
		}
	}

	model.Destroy = func() error {
		switch opts.Backend {
		case "ORT":
			if model.ORTModel != nil {
				return model.ORTModel.Destroy()
			}
		case "GO", "XLA":
			if model.GoMLXModel != nil {
				model.GoMLXModel.Destroy()
			}
		}
		return nil
	}

	return model, nil
}

// HasSeparateVisionEncoder checks if the model uses Florence-2 style separate vision encoder.
func HasSeparateVisionEncoder(modelPath string) bool {
	// Check in onnx/ subdirectory first
	onnxDir := fileutil.PathJoinSafe(modelPath, "onnx")
	if exists, _ := fileutil.FileExists(onnxDir); exists {
		modelPath = onnxDir
	}

	// Look for vision_encoder.onnx
	_, err := findVision2SeqOnnxFile(modelPath, "vision_encoder")
	return err == nil
}

// buildDecoderInitIONames generates input/output names for the init decoder (no past).
func buildDecoderInitIONames(numLayers int) (inputs, outputs []string) {
	// Init decoder only takes input_ids and encoder_hidden_states
	inputs = []string{"input_ids", "encoder_hidden_states"}

	// Outputs: logits + present KV for each layer
	outputs = []string{"logits"}
	for i := 0; i < numLayers; i++ {
		outputs = append(outputs,
			fmt.Sprintf("present.%d.decoder.key", i),
			fmt.Sprintf("present.%d.decoder.value", i),
			fmt.Sprintf("present.%d.encoder.key", i),
			fmt.Sprintf("present.%d.encoder.value", i),
		)
	}

	return inputs, outputs
}

// buildDecoderWithPastIONames generates input/output names for decoder with past.
// Note: The with-past decoder does NOT take encoder_hidden_states. The encoder
// cross-attention is already computed and passed through past_key_values.encoder.
func buildDecoderWithPastIONames(numLayers int) (inputs, outputs []string) {
	// Inputs: input_ids, then past_key_values (NO encoder_hidden_states)
	inputs = []string{"input_ids"}

	// Past key values for each layer (decoder self-attention + encoder cross-attention)
	for i := 0; i < numLayers; i++ {
		inputs = append(inputs,
			fmt.Sprintf("past_key_values.%d.decoder.key", i),
			fmt.Sprintf("past_key_values.%d.decoder.value", i),
			fmt.Sprintf("past_key_values.%d.encoder.key", i),
			fmt.Sprintf("past_key_values.%d.encoder.value", i),
		)
	}

	// Outputs: logits + present KV (only decoder grows, encoder passed through)
	outputs = []string{"logits"}
	for i := 0; i < numLayers; i++ {
		outputs = append(outputs,
			fmt.Sprintf("present.%d.decoder.key", i),
			fmt.Sprintf("present.%d.decoder.value", i),
			// Note: encoder PKV is not in outputs for with-past decoder
		)
	}

	return inputs, outputs
}

// buildFlorenceDecoderInitIONames generates input/output names for Florence-2 init decoder.
// Florence-2 decoder takes inputs_embeds instead of input_ids, and requires encoder_attention_mask.
func buildFlorenceDecoderInitIONames(numLayers int) (inputs, outputs []string) {
	florenceIO := GetFlorenceIONames()
	inputs = make([]string, len(florenceIO.DecoderInitInputsBase))
	copy(inputs, florenceIO.DecoderInitInputsBase)

	// Outputs: logits + present KV for each layer
	outputs = []string{"logits"}
	for i := 0; i < numLayers; i++ {
		outputs = append(outputs,
			fmt.Sprintf("present.%d.decoder.key", i),
			fmt.Sprintf("present.%d.decoder.value", i),
			fmt.Sprintf("present.%d.encoder.key", i),
			fmt.Sprintf("present.%d.encoder.value", i),
		)
	}

	return inputs, outputs
}

// buildFlorenceDecoderWithPastIONames generates input/output names for Florence-2 decoder with past.
// Florence-2 decoder takes inputs_embeds instead of input_ids.
func buildFlorenceDecoderWithPastIONames(numLayers int) (inputs, outputs []string) {
	florenceIO := GetFlorenceIONames()
	inputs = make([]string, len(florenceIO.DecoderWithPastInputsBase))
	copy(inputs, florenceIO.DecoderWithPastInputsBase)

	// Past key values for each layer (decoder self-attention + encoder cross-attention)
	for i := 0; i < numLayers; i++ {
		inputs = append(inputs,
			fmt.Sprintf("past_key_values.%d.decoder.key", i),
			fmt.Sprintf("past_key_values.%d.decoder.value", i),
			fmt.Sprintf("past_key_values.%d.encoder.key", i),
			fmt.Sprintf("past_key_values.%d.encoder.value", i),
		)
	}

	// Outputs: logits + present KV (only decoder grows, encoder passed through)
	outputs = []string{"logits"}
	for i := 0; i < numLayers; i++ {
		outputs = append(outputs,
			fmt.Sprintf("present.%d.decoder.key", i),
			fmt.Sprintf("present.%d.decoder.value", i),
			// Note: encoder PKV is not in outputs for with-past decoder
		)
	}

	return inputs, outputs
}

// buildDecoderIONames generates input/output names for a merged decoder with KV cache.
// numLayers is the number of transformer layers (e.g., 12 for TrOCR base).
func buildDecoderIONames(numLayers int) (inputs, outputs []string) {
	// Core inputs
	inputs = []string{"input_ids", "encoder_hidden_states"}

	// Past key values for each layer (decoder and encoder KV)
	for i := 0; i < numLayers; i++ {
		inputs = append(inputs,
			fmt.Sprintf("past_key_values.%d.decoder.key", i),
			fmt.Sprintf("past_key_values.%d.decoder.value", i),
			fmt.Sprintf("past_key_values.%d.encoder.key", i),
			fmt.Sprintf("past_key_values.%d.encoder.value", i),
		)
	}
	inputs = append(inputs, "use_cache_branch")

	// Outputs: logits + present KV for each layer
	outputs = []string{"logits"}
	for i := 0; i < numLayers; i++ {
		outputs = append(outputs,
			fmt.Sprintf("present.%d.decoder.key", i),
			fmt.Sprintf("present.%d.decoder.value", i),
			fmt.Sprintf("present.%d.encoder.key", i),
			fmt.Sprintf("present.%d.encoder.value", i),
		)
	}

	return inputs, outputs
}

// LoadVision2SeqDecoderInit loads the init decoder model (no past_key_values input).
func LoadVision2SeqDecoderInit(modelPath string, opts *options.Options, numDecoderLayers int) (*Model, error) {
	if numDecoderLayers <= 0 {
		return nil, errors.New("numDecoderLayers must be positive")
	}

	// First check in onnx/ subdirectory
	onnxDir := fileutil.PathJoinSafe(modelPath, "onnx")
	if exists, _ := fileutil.FileExists(onnxDir); exists {
		modelPath = onnxDir
	}

	// Look for decoder_model.onnx specifically (not merged, not with_past)
	onnxFile, err := findVision2SeqOnnxFile(modelPath, "decoder_init")
	if err != nil {
		return nil, err
	}

	model := &Model{
		Path:         modelPath,
		OnnxFilename: onnxFile,
		Pipelines:    make(map[string]Pipeline),
	}

	if err := LoadOnnxModelBytes(model); err != nil {
		return nil, err
	}

	if opts.Backend == "ORT" {
		decoderInputs, decoderOutputs := buildDecoderInitIONames(numDecoderLayers)
		if err := CreateORTModelBackendWithNames(model, opts, decoderInputs, decoderOutputs); err != nil {
			return nil, err
		}
	} else {
		if err := CreateModelBackend(model, opts); err != nil {
			return nil, err
		}
	}

	model.Destroy = func() error {
		switch opts.Backend {
		case "ORT":
			if model.ORTModel != nil {
				return model.ORTModel.Destroy()
			}
		case "GO", "XLA":
			if model.GoMLXModel != nil {
				model.GoMLXModel.Destroy()
			}
		}
		return nil
	}

	return model, nil
}

// LoadVision2SeqDecoderWithPast loads the decoder with past model (takes past_key_values).
func LoadVision2SeqDecoderWithPast(modelPath string, opts *options.Options, numDecoderLayers int) (*Model, error) {
	if numDecoderLayers <= 0 {
		return nil, errors.New("numDecoderLayers must be positive")
	}

	// First check in onnx/ subdirectory
	onnxDir := fileutil.PathJoinSafe(modelPath, "onnx")
	if exists, _ := fileutil.FileExists(onnxDir); exists {
		modelPath = onnxDir
	}

	onnxFile, err := findVision2SeqOnnxFile(modelPath, "decoder_with_past")
	if err != nil {
		return nil, err
	}

	model := &Model{
		Path:         modelPath,
		OnnxFilename: onnxFile,
		Pipelines:    make(map[string]Pipeline),
	}

	if err := LoadOnnxModelBytes(model); err != nil {
		return nil, err
	}

	if opts.Backend == "ORT" {
		decoderInputs, decoderOutputs := buildDecoderWithPastIONames(numDecoderLayers)
		if err := CreateORTModelBackendWithNames(model, opts, decoderInputs, decoderOutputs); err != nil {
			return nil, err
		}
	} else {
		if err := CreateModelBackend(model, opts); err != nil {
			return nil, err
		}
	}

	model.Destroy = func() error {
		switch opts.Backend {
		case "ORT":
			if model.ORTModel != nil {
				return model.ORTModel.Destroy()
			}
		case "GO", "XLA":
			if model.GoMLXModel != nil {
				model.GoMLXModel.Destroy()
			}
		}
		return nil
	}

	return model, nil
}

// LoadVision2SeqFlorenceDecoderInit loads the Florence-2 init decoder model.
// Florence-2 decoder takes inputs_embeds instead of input_ids.
func LoadVision2SeqFlorenceDecoderInit(modelPath string, opts *options.Options, numDecoderLayers int) (*Model, error) {
	if numDecoderLayers <= 0 {
		return nil, errors.New("numDecoderLayers must be positive")
	}

	// First check in onnx/ subdirectory
	onnxDir := fileutil.PathJoinSafe(modelPath, "onnx")
	if exists, _ := fileutil.FileExists(onnxDir); exists {
		modelPath = onnxDir
	}

	// Look for decoder_model.onnx specifically (not merged, not with_past)
	onnxFile, err := findVision2SeqOnnxFile(modelPath, "decoder_init")
	if err != nil {
		return nil, err
	}

	model := &Model{
		Path:         modelPath,
		OnnxFilename: onnxFile,
		Pipelines:    make(map[string]Pipeline),
	}

	if err := LoadOnnxModelBytes(model); err != nil {
		return nil, err
	}

	if opts.Backend == "ORT" {
		decoderInputs, decoderOutputs := buildFlorenceDecoderInitIONames(numDecoderLayers)
		if err := CreateORTModelBackendWithNames(model, opts, decoderInputs, decoderOutputs); err != nil {
			return nil, err
		}
	} else {
		if err := CreateModelBackend(model, opts); err != nil {
			return nil, err
		}
	}

	model.Destroy = func() error {
		switch opts.Backend {
		case "ORT":
			if model.ORTModel != nil {
				return model.ORTModel.Destroy()
			}
		case "GO", "XLA":
			if model.GoMLXModel != nil {
				model.GoMLXModel.Destroy()
			}
		}
		return nil
	}

	return model, nil
}

// LoadVision2SeqFlorenceDecoderWithPast loads the Florence-2 decoder with past model.
// Florence-2 decoder takes inputs_embeds instead of input_ids.
func LoadVision2SeqFlorenceDecoderWithPast(modelPath string, opts *options.Options, numDecoderLayers int) (*Model, error) {
	if numDecoderLayers <= 0 {
		return nil, errors.New("numDecoderLayers must be positive")
	}

	// First check in onnx/ subdirectory
	onnxDir := fileutil.PathJoinSafe(modelPath, "onnx")
	if exists, _ := fileutil.FileExists(onnxDir); exists {
		modelPath = onnxDir
	}

	onnxFile, err := findVision2SeqOnnxFile(modelPath, "decoder_with_past")
	if err != nil {
		return nil, err
	}

	model := &Model{
		Path:         modelPath,
		OnnxFilename: onnxFile,
		Pipelines:    make(map[string]Pipeline),
	}

	if err := LoadOnnxModelBytes(model); err != nil {
		return nil, err
	}

	if opts.Backend == "ORT" {
		decoderInputs, decoderOutputs := buildFlorenceDecoderWithPastIONames(numDecoderLayers)
		if err := CreateORTModelBackendWithNames(model, opts, decoderInputs, decoderOutputs); err != nil {
			return nil, err
		}
	} else {
		if err := CreateModelBackend(model, opts); err != nil {
			return nil, err
		}
	}

	model.Destroy = func() error {
		switch opts.Backend {
		case "ORT":
			if model.ORTModel != nil {
				return model.ORTModel.Destroy()
			}
		case "GO", "XLA":
			if model.GoMLXModel != nil {
				model.GoMLXModel.Destroy()
			}
		}
		return nil
	}

	return model, nil
}

// buildFlorenceMergedDecoderIONames generates input/output names for Florence-2 merged decoder.
// Florence uses inputs_embeds instead of input_ids, plus encoder_attention_mask.
func buildFlorenceMergedDecoderIONames(numLayers int) (inputs, outputs []string) {
	// Core inputs from Florence I/O registry
	florenceIO := GetFlorenceIONames()
	inputs = append(inputs, florenceIO.MergedDecoderInputsBase...)

	// Past key values for each layer (decoder and encoder KV)
	for i := 0; i < numLayers; i++ {
		inputs = append(inputs,
			fmt.Sprintf("past_key_values.%d.decoder.key", i),
			fmt.Sprintf("past_key_values.%d.decoder.value", i),
			fmt.Sprintf("past_key_values.%d.encoder.key", i),
			fmt.Sprintf("past_key_values.%d.encoder.value", i),
		)
	}
	inputs = append(inputs, florenceIO.UseCacheBranch)

	// Outputs: logits + present KV for each layer
	outputs = []string{"logits"}
	for i := 0; i < numLayers; i++ {
		outputs = append(outputs,
			fmt.Sprintf("present.%d.decoder.key", i),
			fmt.Sprintf("present.%d.decoder.value", i),
			fmt.Sprintf("present.%d.encoder.key", i),
			fmt.Sprintf("present.%d.encoder.value", i),
		)
	}

	return inputs, outputs
}

// LoadVision2SeqFlorenceMergedDecoder loads the merged decoder model for Florence-2.
// This handles both initial and subsequent steps in a single model with use_cache_branch.
func LoadVision2SeqFlorenceMergedDecoder(modelPath string, opts *options.Options, numDecoderLayers int) (*Model, error) {
	if numDecoderLayers <= 0 {
		return nil, errors.New("numDecoderLayers must be positive")
	}

	// First check in onnx/ subdirectory
	onnxDir := fileutil.PathJoinSafe(modelPath, "onnx")
	if exists, _ := fileutil.FileExists(onnxDir); exists {
		modelPath = onnxDir
	}

	onnxFile, err := findVision2SeqOnnxFile(modelPath, "decoder_merged")
	if err != nil {
		return nil, err
	}

	model := &Model{
		Path:         modelPath,
		OnnxFilename: onnxFile,
		Pipelines:    make(map[string]Pipeline),
	}

	if err := LoadOnnxModelBytes(model); err != nil {
		return nil, err
	}

	// Use CreateModelBackend to get actual input/output metadata from the ONNX model.
	// This provides proper dimension info for PKV shape inference.
	if err := CreateModelBackend(model, opts); err != nil {
		return nil, err
	}

	model.Destroy = func() error {
		switch opts.Backend {
		case "ORT":
			if model.ORTModel != nil {
				return model.ORTModel.Destroy()
			}
		case "GO", "XLA":
			if model.GoMLXModel != nil {
				model.GoMLXModel.Destroy()
			}
		}
		return nil
	}

	return model, nil
}

// LoadVision2SeqDecoder loads the decoder model for vision2seq inference.
// Looks for decoder_model_merged.onnx or decoder*.onnx in the model path.
// The "merged" decoder handles both initial step (no past_key_values) and
// subsequent steps (with past_key_values) in a single model.
// numDecoderLayers must be provided from the model config.
func LoadVision2SeqDecoder(modelPath string, opts *options.Options, numDecoderLayers int) (*Model, error) {
	if numDecoderLayers <= 0 {
		return nil, errors.New("numDecoderLayers must be positive")
	}

	// First check in onnx/ subdirectory
	onnxDir := fileutil.PathJoinSafe(modelPath, "onnx")
	if exists, _ := fileutil.FileExists(onnxDir); exists {
		modelPath = onnxDir
	}

	onnxFile, err := findVision2SeqOnnxFile(modelPath, "decoder")
	if err != nil {
		return nil, err
	}

	model := &Model{
		Path:         modelPath,
		OnnxFilename: onnxFile,
		Pipelines:    make(map[string]Pipeline),
	}

	if err := LoadOnnxModelBytes(model); err != nil {
		return nil, err
	}

	// Use predefined input/output names for ORT to avoid metadata extraction
	// which creates a temporary session without our session options
	if opts.Backend == "ORT" {
		decoderInputs, decoderOutputs := buildDecoderIONames(numDecoderLayers)
		if err := CreateORTModelBackendWithNames(model, opts, decoderInputs, decoderOutputs); err != nil {
			return nil, err
		}
	} else {
		if err := CreateModelBackend(model, opts); err != nil {
			return nil, err
		}
	}

	model.Destroy = func() error {
		switch opts.Backend {
		case "ORT":
			if model.ORTModel != nil {
				return model.ORTModel.Destroy()
			}
		case "GO", "XLA":
			if model.GoMLXModel != nil {
				model.GoMLXModel.Destroy()
			}
		}
		return nil
	}

	return model, nil
}

// LoadVision2SeqTokenizer loads the tokenizer for vision2seq models.
func LoadVision2SeqTokenizer(modelPath string, opts *options.Options) (*Tokenizer, error) {
	tempModel := &Model{
		Path: modelPath,
	}

	if err := LoadTokenizer(tempModel, opts); err != nil {
		return nil, err
	}

	return tempModel.Tokenizer, nil
}

// LoadVision2SeqConfig loads the model configuration from config.json.
// TrOCR models have a nested structure with vision_config and text_config.
// Also loads generation_config.json for token IDs if not found in config.json.
// Supports TrOCR, Donut, Florence-2, and Nougat model types.
func LoadVision2SeqConfig(modelPath string) (*Vision2SeqConfig, error) {
	configPath := fileutil.PathJoinSafe(modelPath, "config.json")

	exists, err := fileutil.FileExists(configPath)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, fmt.Errorf("config.json not found at %s", modelPath)
	}

	configBytes, err := fileutil.ReadFileBytes(configPath)
	if err != nil {
		return nil, err
	}

	var configMap map[string]any
	if err := json.Unmarshal(configBytes, &configMap); err != nil {
		return nil, err
	}

	config := &Vision2SeqConfig{
		EosTokenIDs:  make(map[int64]bool),
		ImageSize:    384, // Default for TrOCR
		NumChannels:  3,
		PadValue:     1.0,    // White padding by default
		ModelType:    "auto", // Auto-detect
		OutputFormat: "text", // Plain text by default
	}

	// Check top-level model_type for Florence-2 and similar models
	if modelType, ok := configMap["model_type"].(string); ok {
		switch modelType {
		case "florence2":
			config.ModelType = "florence"
			config.OutputFormat = "text"
		case "vision-encoder-decoder":
			// Fall through to encoder config parsing
		}
	}

	// Parse encoder config for image dimensions and detect model type
	if encoderConfig, ok := configMap["encoder"].(map[string]any); ok {
		parseEncoderConfig(encoderConfig, config, modelPath)
	} else if visionConfig, ok := configMap["vision_config"].(map[string]any); ok {
		parseEncoderConfig(visionConfig, config, modelPath)
	} else if v, ok := configMap["image_size"].(float64); ok {
		// Flat config format
		config.ImageSize = int(v)
	}

	// Decoder start token
	if v, ok := configMap["decoder_start_token_id"].(float64); ok {
		config.DecoderStartTokenID = int64(v)
	}

	// EOS token(s)
	if eosRaw, exists := configMap["eos_token_id"]; exists {
		switch v := eosRaw.(type) {
		case []any:
			for _, item := range v {
				if num, ok := item.(float64); ok {
					config.EosTokenIDs[int64(num)] = true
				}
			}
		case float64:
			config.EosTokenIDs[int64(v)] = true
		}
	}

	// Pad token
	if v, ok := configMap["pad_token_id"].(float64); ok {
		config.PadTokenID = int64(v)
	}

	// Vocab size
	if v, ok := configMap["vocab_size"].(float64); ok {
		config.VocabSize = int(v)
	}

	// Parse decoder/text_config for hidden dimensions and decoder-specific tokens
	if decoderConfig, ok := configMap["decoder"].(map[string]any); ok {
		parseDecoderConfig(decoderConfig, config)
		// Also check decoder config for tokens if not set at top level
		parseTokenConfig(decoderConfig, config)
	} else if textConfig, ok := configMap["text_config"].(map[string]any); ok {
		parseDecoderConfig(textConfig, config)
		parseTokenConfig(textConfig, config)
	} else {
		// Flat config
		parseDecoderConfig(configMap, config)
	}

	// Try loading generation_config.json for token IDs if not found in config.json
	// This is common with Optimum-exported models
	if len(config.EosTokenIDs) == 0 || config.DecoderStartTokenID == 0 {
		genConfigPath := fileutil.PathJoinSafe(modelPath, "generation_config.json")
		if exists, _ := fileutil.FileExists(genConfigPath); exists {
			genBytes, err := fileutil.ReadFileBytes(genConfigPath)
			if err == nil {
				var genConfig map[string]any
				if json.Unmarshal(genBytes, &genConfig) == nil {
					parseTokenConfig(genConfig, config)
				}
			}
		}
	}

	// Load preprocessor_config.json for image normalization parameters and preprocessing flags
	preprocPath := fileutil.PathJoinSafe(modelPath, "preprocessor_config.json")
	if exists, _ := fileutil.FileExists(preprocPath); exists {
		preprocBytes, err := fileutil.ReadFileBytes(preprocPath)
		if err == nil {
			var preprocConfig map[string]any
			if json.Unmarshal(preprocBytes, &preprocConfig) == nil {
				// Parse image_mean
				if meanRaw, ok := preprocConfig["image_mean"].([]any); ok {
					config.ImageMean = make([]float32, len(meanRaw))
					for i, v := range meanRaw {
						if f, ok := v.(float64); ok {
							config.ImageMean[i] = float32(f)
						}
					}
				}
				// Parse image_std
				if stdRaw, ok := preprocConfig["image_std"].([]any); ok {
					config.ImageStd = make([]float32, len(stdRaw))
					for i, v := range stdRaw {
						if f, ok := v.(float64); ok {
							config.ImageStd[i] = float32(f)
						}
					}
				}
				// Override image size if specified
				if sizeRaw, ok := preprocConfig["size"].(map[string]any); ok {
					if h, ok := sizeRaw["height"].(float64); ok {
						config.ImageHeight = int(h)
						if config.ImageHeight > config.ImageSize {
							config.ImageSize = config.ImageHeight
						}
					}
					if w, ok := sizeRaw["width"].(float64); ok {
						config.ImageWidth = int(w)
						if config.ImageWidth > config.ImageSize {
							config.ImageSize = config.ImageWidth
						}
					}
				} else if sizeArr, ok := preprocConfig["size"].([]any); ok {
					// [height, width] array format
					if len(sizeArr) >= 2 {
						if h, ok := sizeArr[0].(float64); ok {
							config.ImageHeight = int(h)
						}
						if w, ok := sizeArr[1].(float64); ok {
							config.ImageWidth = int(w)
						}
						if config.ImageHeight > config.ImageWidth {
							config.ImageSize = config.ImageHeight
						} else {
							config.ImageSize = config.ImageWidth
						}
					}
				}

				// Parse preprocessing flags
				if v, ok := preprocConfig["do_align_long_axis"].(bool); ok {
					config.DoAlignLongAxis = v
				}
				if v, ok := preprocConfig["do_pad"].(bool); ok {
					config.DoPad = v
				}
				if v, ok := preprocConfig["do_crop_margin"].(bool); ok {
					config.DoCropMargin = v
				}
				if v, ok := preprocConfig["do_thumbnail"].(bool); ok {
					config.DoThumbnail = v
				}
			}
		}
	}

	// Set sensible defaults for image dimensions if not set
	if config.ImageWidth == 0 {
		config.ImageWidth = config.ImageSize
	}
	if config.ImageHeight == 0 {
		config.ImageHeight = config.ImageSize
	}

	return config, nil
}

// parseTokenConfig extracts token IDs and vocab size from a config section.
// Called after top-level parsing to check decoder/text_config as fallback.
func parseTokenConfig(cfg map[string]any, config *Vision2SeqConfig) {
	// Vocab size (only if not already set)
	if config.VocabSize == 0 {
		if v, ok := cfg["vocab_size"].(float64); ok {
			config.VocabSize = int(v)
		}
	}

	// EOS token(s) (only if not already set)
	if len(config.EosTokenIDs) == 0 {
		if eosRaw, exists := cfg["eos_token_id"]; exists {
			switch v := eosRaw.(type) {
			case []any:
				for _, item := range v {
					if num, ok := item.(float64); ok {
						config.EosTokenIDs[int64(num)] = true
					}
				}
			case float64:
				config.EosTokenIDs[int64(v)] = true
			}
		}
	}

	// Pad token (only if not already set)
	if config.PadTokenID == 0 {
		if v, ok := cfg["pad_token_id"].(float64); ok {
			config.PadTokenID = int64(v)
		}
	}

	// Decoder start token (only if not already set)
	if config.DecoderStartTokenID == 0 {
		if v, ok := cfg["decoder_start_token_id"].(float64); ok {
			config.DecoderStartTokenID = int64(v)
		}
	}
}

func parseDecoderConfig(cfg map[string]any, config *Vision2SeqConfig) {
	if v, ok := cfg["hidden_size"].(float64); ok {
		config.HiddenSize = int(v)
	} else if v, ok := cfg["d_model"].(float64); ok {
		config.HiddenSize = int(v)
	}

	if v, ok := cfg["decoder_layers"].(float64); ok {
		config.NumDecoderLayers = int(v)
	} else if v, ok := cfg["num_decoder_layers"].(float64); ok {
		config.NumDecoderLayers = int(v)
	} else if v, ok := cfg["num_layers"].(float64); ok {
		config.NumDecoderLayers = int(v)
	}

	if v, ok := cfg["decoder_attention_heads"].(float64); ok {
		config.NumHeads = int(v)
	} else if v, ok := cfg["num_attention_heads"].(float64); ok {
		config.NumHeads = int(v)
	} else if v, ok := cfg["num_heads"].(float64); ok {
		config.NumHeads = int(v)
	}

	// Max position embeddings (important for Nougat's long sequences)
	if v, ok := cfg["max_position_embeddings"].(float64); ok {
		config.MaxPositionEmbeddings = int(v)
	}

	if config.HiddenSize > 0 && config.NumHeads > 0 {
		config.HeadDim = config.HiddenSize / config.NumHeads
	}
}

// parseEncoderConfig extracts encoder configuration and detects model type.
func parseEncoderConfig(cfg map[string]any, config *Vision2SeqConfig, modelPath string) {
	// Extract image size (can be int or [height, width] array)
	if v, ok := cfg["image_size"].(float64); ok {
		config.ImageSize = int(v)
		config.ImageWidth = int(v)
		config.ImageHeight = int(v)
	} else if sizeArr, ok := cfg["image_size"].([]any); ok {
		// [height, width] format used by Donut/Nougat
		if len(sizeArr) >= 2 {
			if h, ok := sizeArr[0].(float64); ok {
				config.ImageHeight = int(h)
			}
			if w, ok := sizeArr[1].(float64); ok {
				config.ImageWidth = int(w)
			}
			// Set ImageSize to max dimension for compatibility
			if config.ImageHeight > config.ImageWidth {
				config.ImageSize = config.ImageHeight
			} else {
				config.ImageSize = config.ImageWidth
			}
		}
	}

	if v, ok := cfg["num_channels"].(float64); ok {
		config.NumChannels = int(v)
	}

	// Detect model type from encoder model_type using the registry
	var detectedType Vision2SeqModelType
	if encoderModelType, ok := cfg["model_type"].(string); ok {
		detectedType = DetectModelTypeFromEncoder(encoderModelType)

		// Special case: swin/swinv2 could be Nougat - check path
		if (encoderModelType == "swin" || encoderModelType == "swinv2") &&
			DetectModelTypeFromPath(modelPath) == ModelTypeNougat {
			detectedType = ModelTypeNougat
		}
	}

	// Fall back to path-based detection if encoder type didn't match
	if detectedType == ModelTypeUnknown {
		detectedType = DetectModelTypeFromPath(modelPath)
	}

	// Apply the detected model profile
	if detectedType != ModelTypeUnknown {
		applyModelProfile(config, detectedType)
	}
}

// applyModelProfile applies a model profile's settings to the config.
// Only applies preprocessing flags; other config values come from model files.
func applyModelProfile(config *Vision2SeqConfig, modelType Vision2SeqModelType) {
	profile := GetModelProfile(modelType)
	config.ModelType = string(profile.ModelType)
	config.OutputFormat = profile.OutputFormat
	config.DoAlignLongAxis = profile.DoAlignLongAxis
	config.DoPad = profile.DoPad
	config.DoCropMargin = profile.DoCropMargin
	config.DoThumbnail = profile.DoThumbnail
	// Note: PadValue defaults are already set in config initialization
}

// findVision2SeqOnnxFile finds an ONNX file for vision2seq models.
func findVision2SeqOnnxFile(modelPath string, modelType string) (string, error) {
	onnxFiles, err := getOnnxFiles(modelPath)
	if err != nil {
		return "", err
	}

	// Priority order for finding model files
	var patterns []string
	var excludePatterns []string
	switch modelType {
	case "encoder":
		patterns = []string{"encoder_model", "encoder"}
		excludePatterns = []string{"vision_encoder", "quantized", "fp16", "int8", "uint8", "bnb4", "q4", "q4f16"}
	case "vision_encoder":
		// Florence-2 vision encoder: vision_encoder.onnx
		patterns = []string{"vision_encoder"}
		excludePatterns = []string{"quantized", "fp16", "int8", "uint8", "bnb4", "q4", "q4f16"}
	case "embed_tokens":
		// Florence-2 embed tokens: embed_tokens.onnx
		patterns = []string{"embed_tokens"}
		excludePatterns = []string{"quantized", "fp16", "int8", "uint8", "bnb4", "q4", "q4f16"}
	case "decoder":
		// Prefer merged decoder (single model for both init and subsequent steps)
		patterns = []string{"decoder_model_merged", "decoder_model", "decoder"}
		excludePatterns = []string{"quantized", "fp16", "int8", "uint8", "bnb4", "q4", "q4f16"}
	case "decoder_merged":
		// Merged decoder specifically: decoder_model_merged.onnx
		patterns = []string{"decoder_model_merged"}
		excludePatterns = []string{"quantized", "fp16", "int8", "uint8", "bnb4", "q4", "q4f16"}
	case "decoder_init":
		// Init decoder: decoder_model.onnx (not merged, not with_past)
		patterns = []string{"decoder_model"}
		excludePatterns = []string{"merged", "with_past", "quantized", "fp16", "int8", "uint8", "bnb4", "q4", "q4f16"}
	case "decoder_with_past":
		// Decoder with past: decoder_with_past_model.onnx
		patterns = []string{"decoder_with_past"}
		excludePatterns = []string{"quantized", "fp16", "int8", "uint8", "bnb4", "q4", "q4f16"}
	}

	for _, pattern := range patterns {
		for _, file := range onnxFiles {
			filename := strings.ToLower(file[1])
			if strings.Contains(filename, pattern) {
				// Check exclusions
				excluded := false
				for _, ex := range excludePatterns {
					if strings.Contains(filename, ex) {
						excluded = true
						break
					}
				}
				if !excluded {
					return file[1], nil
				}
			}
		}
	}

	return "", fmt.Errorf("no ONNX %s file found in %s", modelType, modelPath)
}

// Vision2SeqBatch holds the intermediate state during vision2seq generation.
type Vision2SeqBatch struct {
	// Input
	Images           []image.Image
	PreprocessedData [][][][]float32
	Size             int

	// Encoder output (cached for decoder)
	EncoderHiddenStates any // Backend-specific tensor

	// Decoder state
	PastKeyValues []any // List of KV cache tensors
	Logits        any

	// Prompt token IDs for task-prompted models (DocVQA, etc.)
	// PromptTokenIDs[i] contains the prompt tokens for image i
	PromptTokenIDs [][]int64

	// Generation tracking
	GeneratedTokens [][]int64
	Finished        []bool
	FinishedCount   int

	// Cleanup functions
	DestroyEncoder func() error
	DestroyDecoder func() error
}

// NewVision2SeqBatch creates a new batch for vision2seq inference.
func NewVision2SeqBatch(size int) *Vision2SeqBatch {
	return &Vision2SeqBatch{
		Size:            size,
		GeneratedTokens: make([][]int64, size),
		Finished:        make([]bool, size),
		DestroyEncoder:  func() error { return nil },
		DestroyDecoder:  func() error { return nil },
	}
}

// Destroy cleans up batch resources.
func (b *Vision2SeqBatch) Destroy() error {
	return errors.Join(b.DestroyEncoder(), b.DestroyDecoder())
}

// Interface implementations for Vision2SeqBatchInterface

func (b *Vision2SeqBatch) GetSize() int                           { return b.Size }
func (b *Vision2SeqBatch) GetImages() []image.Image               { return b.Images }
func (b *Vision2SeqBatch) GetPreprocessedImages() [][][][]float32 { return b.PreprocessedData }
func (b *Vision2SeqBatch) SetEncoderHiddenStates(states any)      { b.EncoderHiddenStates = states }
func (b *Vision2SeqBatch) GetEncoderHiddenStates() any            { return b.EncoderHiddenStates }
func (b *Vision2SeqBatch) SetPastKeyValues(pkv []any)             { b.PastKeyValues = pkv }
func (b *Vision2SeqBatch) GetPastKeyValues() []any                { return b.PastKeyValues }
func (b *Vision2SeqBatch) SetLogits(logits any)                   { b.Logits = logits }
func (b *Vision2SeqBatch) GetLogits() any                         { return b.Logits }
func (b *Vision2SeqBatch) GetGeneratedTokens() [][]int64          { return b.GeneratedTokens }
func (b *Vision2SeqBatch) SetGeneratedTokens(tokens [][]int64)    { b.GeneratedTokens = tokens }
func (b *Vision2SeqBatch) GetFinished() []bool                    { return b.Finished }
func (b *Vision2SeqBatch) SetFinished(finished []bool)            { b.Finished = finished }
func (b *Vision2SeqBatch) GetFinishedCount() int                  { return b.FinishedCount }
func (b *Vision2SeqBatch) SetFinishedCount(count int)             { b.FinishedCount = count }
func (b *Vision2SeqBatch) SetDestroyEncoder(fn func() error)      { b.DestroyEncoder = fn }
func (b *Vision2SeqBatch) SetDestroyDecoder(fn func() error)      { b.DestroyDecoder = fn }
func (b *Vision2SeqBatch) GetPromptTokenIDs() [][]int64           { return b.PromptTokenIDs }
func (b *Vision2SeqBatch) SetPromptTokenIDs(tokens [][]int64)     { b.PromptTokenIDs = tokens }
