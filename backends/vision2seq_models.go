package backends

// Vision2SeqModelType defines supported vision-to-sequence model types.
type Vision2SeqModelType string

const (
	ModelTypeUnknown  Vision2SeqModelType = ""
	ModelTypeAuto     Vision2SeqModelType = "auto"
	ModelTypeTrOCR    Vision2SeqModelType = "trocr"
	ModelTypeDonut    Vision2SeqModelType = "donut"
	ModelTypeNougat   Vision2SeqModelType = "nougat"
	ModelTypeFlorence Vision2SeqModelType = "florence"
)

// Vision2SeqArchitecture defines the encoder architecture type.
type Vision2SeqArchitecture string

const (
	// ArchitectureStandard is the standard vision-encoder-decoder (TrOCR, Donut, Nougat).
	// Uses encoder_model.onnx + decoder_model.onnx
	ArchitectureStandard Vision2SeqArchitecture = "standard"

	// ArchitectureFlorence is the Florence-2 architecture with separate vision encoder.
	// Uses vision_encoder.onnx + embed_tokens.onnx + encoder_model.onnx + decoder_model.onnx
	ArchitectureFlorence Vision2SeqArchitecture = "florence"
)

// Vision2SeqModelProfile contains all model-specific configuration.
// This eliminates hardcoded switch statements and makes adding new models easy.
type Vision2SeqModelProfile struct {
	// Model identification
	ModelType    Vision2SeqModelType
	Architecture Vision2SeqArchitecture

	// Output format
	OutputFormat string // "text", "json", "markdown"

	// I/O tensor names for standard architecture
	EncoderInputs  []string
	EncoderOutputs []string

	// Preprocessing configuration
	DoAlignLongAxis bool    // Rotate landscape to portrait
	DoPad           bool    // Pad to target size (preserves aspect ratio)
	DoCropMargin    bool    // Remove white margins
	DoThumbnail     bool    // Create thumbnail
	PadValue        float32 // Background color (0.0=black, 1.0=white)
	CropThreshold   uint8   // Pixel threshold for margin detection (default: 250)
	CropMinMargin   int     // Minimum margin to keep around content

	// Default generation parameters
	MaxNewTokens      int
	DoSample          bool
	TopP              float32
	Temperature       float32
	RepetitionPenalty float32

	// Default image size (can be overridden by model config)
	DefaultImageSize int
}

// DefaultModelProfile returns the default profile for unknown models.
func DefaultModelProfile() Vision2SeqModelProfile {
	return Vision2SeqModelProfile{
		ModelType:         ModelTypeTrOCR,
		Architecture:      ArchitectureStandard,
		OutputFormat:      "text",
		EncoderInputs:     []string{"pixel_values"},
		EncoderOutputs:    []string{"last_hidden_state"},
		PadValue:          1.0, // White background
		MaxNewTokens:      128,
		DoSample:          false,
		TopP:              0.9,
		Temperature:       1.0,
		RepetitionPenalty: 1.0,
		DefaultImageSize:  384,
	}
}

// modelProfiles contains configurations for all supported model types.
var modelProfiles = map[Vision2SeqModelType]Vision2SeqModelProfile{
	ModelTypeTrOCR: {
		ModelType:         ModelTypeTrOCR,
		Architecture:      ArchitectureStandard,
		OutputFormat:      "text",
		EncoderInputs:     []string{"pixel_values"},
		EncoderOutputs:    []string{"last_hidden_state"},
		PadValue:          1.0,
		MaxNewTokens:      128,
		DoSample:          false,
		TopP:              0.9,
		Temperature:       1.0,
		RepetitionPenalty: 1.0,
		DefaultImageSize:  384,
	},
	ModelTypeDonut: {
		ModelType:         ModelTypeDonut,
		Architecture:      ArchitectureStandard,
		OutputFormat:      "json",
		EncoderInputs:     []string{"pixel_values"},
		EncoderOutputs:    []string{"last_hidden_state"},
		DoAlignLongAxis:   true,
		DoPad:             true,
		PadValue:          1.0, // White background
		MaxNewTokens:      512,
		DoSample:          false,
		TopP:              0.9,
		Temperature:       1.0,
		RepetitionPenalty: 1.0,
		DefaultImageSize:  1280,
	},
	ModelTypeNougat: {
		ModelType:         ModelTypeNougat,
		Architecture:      ArchitectureStandard,
		OutputFormat:      "markdown",
		EncoderInputs:     []string{"pixel_values"},
		EncoderOutputs:    []string{"last_hidden_state"},
		DoCropMargin:      true,
		DoPad:             true,
		CropThreshold:     250,
		CropMinMargin:     0,
		PadValue:          1.0, // White background
		MaxNewTokens:      4096,
		DoSample:          false,
		TopP:              0.9,
		Temperature:       1.0,
		RepetitionPenalty: 1.0,
		DefaultImageSize:  896,
	},
	ModelTypeFlorence: {
		ModelType:         ModelTypeFlorence,
		Architecture:      ArchitectureFlorence,
		OutputFormat:      "text",
		EncoderInputs:     []string{"pixel_values"},
		EncoderOutputs:    []string{"last_hidden_state"},
		PadValue:          0.0, // Black background for Florence
		MaxNewTokens:      1024,
		DoSample:          false,
		TopP:              0.9,
		Temperature:       1.0,
		RepetitionPenalty: 1.0,
		DefaultImageSize:  768,
	},
}

// FlorenceIONames contains I/O names for Florence-2 architecture components.
type FlorenceIONames struct {
	VisionEncoderInputs  []string
	VisionEncoderOutputs []string
	EmbedTokensInputs    []string
	EmbedTokensOutputs   []string
	EncoderInputs        []string
	EncoderOutputs       []string
	// Decoder init takes: encoder_attention_mask, encoder_hidden_states, inputs_embeds
	// Unlike standard decoders that take input_ids
	DecoderInitInputsBase []string
	// Decoder with past takes: inputs_embeds + past_key_values
	DecoderWithPastInputsBase []string
	// Merged decoder takes: encoder_attention_mask, encoder_hidden_states, inputs_embeds + past_key_values + use_cache_branch
	MergedDecoderInputsBase []string
	// UseCacheBranch is the control input for merged decoder (Florence-specific)
	UseCacheBranch string
}

// florenceIONames contains I/O names for Florence-2 architecture components.
var florenceIONames = FlorenceIONames{
	VisionEncoderInputs:       []string{"pixel_values"},
	VisionEncoderOutputs:      []string{"image_features"},
	EmbedTokensInputs:         []string{"input_ids"},
	EmbedTokensOutputs:        []string{"inputs_embeds"},
	EncoderInputs:             []string{"inputs_embeds", "attention_mask"},
	EncoderOutputs:            []string{"last_hidden_state"},
	DecoderInitInputsBase:     []string{"encoder_attention_mask", "encoder_hidden_states", "inputs_embeds"},
	DecoderWithPastInputsBase: []string{"inputs_embeds"},
	MergedDecoderInputsBase:   []string{"encoder_attention_mask", "encoder_hidden_states", "inputs_embeds"},
	UseCacheBranch:            "use_cache_branch",
}

// GetModelProfile returns the profile for a given model type.
// Returns the default profile if the type is unknown.
func GetModelProfile(modelType Vision2SeqModelType) Vision2SeqModelProfile {
	if profile, ok := modelProfiles[modelType]; ok {
		return profile
	}
	return DefaultModelProfile()
}

// GetFlorenceIONames returns the I/O names for Florence-2 architecture.
func GetFlorenceIONames() FlorenceIONames {
	return florenceIONames
}

// encoderModelTypeMapping maps encoder model_type values to Vision2SeqModelType.
var encoderModelTypeMapping = map[string]Vision2SeqModelType{
	"donut-swin": ModelTypeDonut,
	"davit":      ModelTypeFlorence,
	"swin":       ModelTypeDonut, // Could be Nougat, needs path check
	"swinv2":     ModelTypeDonut, // Could be Nougat, needs path check
	"vit":        ModelTypeTrOCR,
	"deit":       ModelTypeTrOCR,
	"beit":       ModelTypeTrOCR,
}

// DetectModelTypeFromEncoder detects model type from encoder config model_type field.
func DetectModelTypeFromEncoder(encoderModelType string) Vision2SeqModelType {
	if modelType, ok := encoderModelTypeMapping[encoderModelType]; ok {
		return modelType
	}
	return ModelTypeUnknown
}

// DetectModelTypeFromPath attempts to detect model type from the model path.
// This is a fallback when encoder model_type doesn't provide a match.
func DetectModelTypeFromPath(modelPath string) Vision2SeqModelType {
	pathLower := toLower(modelPath)
	switch {
	case contains(pathLower, "trocr"):
		return ModelTypeTrOCR
	case contains(pathLower, "donut"):
		return ModelTypeDonut
	case contains(pathLower, "nougat"):
		return ModelTypeNougat
	case contains(pathLower, "florence"):
		return ModelTypeFlorence
	default:
		return ModelTypeUnknown
	}
}

// GetArchitecture returns the architecture type for a model.
func GetArchitecture(modelType Vision2SeqModelType) Vision2SeqArchitecture {
	profile := GetModelProfile(modelType)
	return profile.Architecture
}

// IsFlorence returns true if the model uses Florence-2 architecture.
func IsFlorence(modelType Vision2SeqModelType) bool {
	return GetArchitecture(modelType) == ArchitectureFlorence
}

// GetProfileByModelTypeString returns a profile from a string model type.
// This is useful when working with config values that are strings.
func GetProfileByModelTypeString(modelType string) Vision2SeqModelProfile {
	return GetModelProfile(Vision2SeqModelType(modelType))
}

// Helper functions to avoid importing strings package
func toLower(s string) string {
	b := make([]byte, len(s))
	for i := range b {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		b[i] = c
	}
	return string(b)
}

func contains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// IONameMatcher provides centralized I/O name matching for vision2seq models.
// It checks registry names first (exact match), then falls back to pattern matching
// for flexibility with different ONNX export variations.
type IONameMatcher struct {
	florenceIO FlorenceIONames
}

// NewIONameMatcher creates a new matcher initialized with the Florence I/O registry.
func NewIONameMatcher() *IONameMatcher {
	return &IONameMatcher{
		florenceIO: GetFlorenceIONames(),
	}
}

// Global matcher instance for convenience
var ioMatcher = NewIONameMatcher()

// GetIONameMatcher returns the global IONameMatcher instance.
func GetIONameMatcher() *IONameMatcher {
	return ioMatcher
}

// --- Encoder Output Matchers ---

// IsImageFeatures checks if the name is an image features output (Florence vision encoder).
func (m *IONameMatcher) IsImageFeatures(name string) bool {
	// Exact match from registry
	if len(m.florenceIO.VisionEncoderOutputs) > 0 && name == m.florenceIO.VisionEncoderOutputs[0] {
		return true
	}
	// Fallback patterns for ONNX variations
	return contains(name, "image_features") || contains(name, "features")
}

// IsHiddenStates checks if the name is a hidden states output (encoder output).
func (m *IONameMatcher) IsHiddenStates(name string) bool {
	// Exact match from registry
	if len(m.florenceIO.EncoderOutputs) > 0 && name == m.florenceIO.EncoderOutputs[0] {
		return true
	}
	// Fallback patterns
	return contains(name, "hidden_states") || name == "last_hidden_state"
}

// --- Embed Tokens Matchers ---

// IsInputsEmbeds checks if the name is an inputs_embeds tensor.
func (m *IONameMatcher) IsInputsEmbeds(name string) bool {
	// Exact match from registry
	if len(m.florenceIO.EmbedTokensOutputs) > 0 && name == m.florenceIO.EmbedTokensOutputs[0] {
		return true
	}
	// Also check decoder inputs
	if len(m.florenceIO.DecoderInitInputsBase) >= 3 && name == m.florenceIO.DecoderInitInputsBase[2] {
		return true
	}
	// Fallback pattern
	return contains(name, "inputs_embeds") || contains(name, "embeds")
}

// --- Decoder Input Matchers ---

// IsEncoderHiddenStates checks if the name is encoder_hidden_states input.
func (m *IONameMatcher) IsEncoderHiddenStates(name string) bool {
	// Exact match from registry
	if len(m.florenceIO.DecoderInitInputsBase) >= 2 && name == m.florenceIO.DecoderInitInputsBase[1] {
		return true
	}
	// Fallback pattern
	return contains(name, "encoder_hidden_states")
}

// IsEncoderAttentionMask checks if the name is encoder_attention_mask input.
func (m *IONameMatcher) IsEncoderAttentionMask(name string) bool {
	// Exact match from registry
	if len(m.florenceIO.DecoderInitInputsBase) >= 1 && name == m.florenceIO.DecoderInitInputsBase[0] {
		return true
	}
	// Fallback pattern
	return contains(name, "encoder_attention_mask")
}

// IsUseCacheBranch checks if the name is use_cache_branch input (Florence merged decoder).
func (m *IONameMatcher) IsUseCacheBranch(name string) bool {
	// Exact match from registry
	if m.florenceIO.UseCacheBranch != "" && name == m.florenceIO.UseCacheBranch {
		return true
	}
	// Fallback pattern
	return contains(name, "use_cache_branch")
}

// IsPastKeyValues checks if the name is a past_key_values input.
func (m *IONameMatcher) IsPastKeyValues(name string) bool {
	return contains(name, "past_key_values") || contains(name, "past")
}

// --- Standard Decoder Matchers ---

// IsInputIds checks if the name is input_ids (standard decoder input).
func (m *IONameMatcher) IsInputIds(name string) bool {
	return name == "input_ids" || contains(name, "input_ids")
}

// IsLogits checks if the name is logits output.
func (m *IONameMatcher) IsLogits(name string) bool {
	return name == "logits" || contains(name, "logits")
}

// IsAttentionMask checks if the name is attention_mask (not encoder_attention_mask).
func (m *IONameMatcher) IsAttentionMask(name string) bool {
	// Exact match from registry (Florence encoder)
	if len(m.florenceIO.EncoderInputs) >= 2 && name == m.florenceIO.EncoderInputs[1] {
		return true
	}
	// Match attention_mask but not encoder_attention_mask
	return name == "attention_mask" || (contains(name, "attention_mask") && !contains(name, "encoder_attention_mask"))
}
