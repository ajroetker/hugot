package pipelines

import (
	"errors"
	"fmt"
	"image"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/imageutil"
	"github.com/knights-analytics/hugot/util/safeconv"
)

// Vision2SeqPipeline enables vision-to-text inference for models like TrOCR.
// It takes images as input and generates text sequences.
//
// The pipeline expects models exported in the Optimum format with:
//   - encoder_model.onnx (vision encoder, takes pixel_values)
//   - decoder_model_merged.onnx (decoder with optional past_key_values)
//
// Example usage:
//
//	session, err := hugot.NewORTSession()
//	check(err)
//	defer session.Destroy()
//
//	config := backends.PipelineConfig[*Vision2SeqPipeline]{
//		ModelPath: "./models/trocr-base-printed",
//		Name:      "trocr",
//		Options: []backends.PipelineOption[*Vision2SeqPipeline]{
//			WithVision2SeqMaxTokens(64),
//		},
//	}
//
//	pipeline, err := NewVision2SeqPipeline(config, session.Options, nil)
//	check(err)
//
//	output, err := pipeline.RunWithImages([]image.Image{img})
//	// output.GeneratedTexts[0] = "recognized text from image"
type Vision2SeqPipeline struct {
	// Models
	EncoderModel         *backends.Model
	DecoderModel         *backends.Model // For merged decoder
	DecoderInitModel     *backends.Model // For split decoder (init, no past)
	DecoderWithPastModel *backends.Model // For split decoder (with past)
	useSplitDecoder      bool            // True if using split decoder

	// Tokenizer (for decoding output tokens)
	Tokenizer *backends.Tokenizer

	// Pipeline configuration
	PipelineName    string
	PipelineTimings *vision2seqTimings
	Runtime         string

	// Generation parameters
	MaxNewTokens      int
	DoSample          bool
	TopP              float32
	Temperature       float32
	RepetitionPenalty float32

	// Special token IDs
	DecoderStartTokenID int64
	EosTokenIDs         map[int64]bool
	PadTokenID          int64

	// Model dimensions
	VocabSize        int
	ImageSize        int
	NumHeads         int
	HeadDim          int
	NumDecoderLayers int

	// Image preprocessing
	imageFormat        string
	preprocessSteps    []imageutil.PreprocessStep
	normalizationSteps []imageutil.NormalizationStep
}

// vision2seqTimings tracks timing statistics for vision2seq pipeline components.
type vision2seqTimings struct {
	PreprocessNumCalls uint64
	PreprocessTotalNS  uint64
	EncoderNumCalls    uint64
	EncoderTotalNS     uint64
	DecoderNumCalls    uint64
	DecoderTotalNS     uint64
}

// Vision2SeqOutput contains the generated text from the pipeline.
type Vision2SeqOutput struct {
	// GeneratedTexts[i] is the generated text for the i-th input image
	GeneratedTexts []string
	// GeneratedTokens[i] is the token IDs for GeneratedTexts[i]
	GeneratedTokens [][]uint32
}

// GetOutput implements backends.PipelineBatchOutput interface.
func (o *Vision2SeqOutput) GetOutput() []any {
	out := make([]any, len(o.GeneratedTexts))
	for i, text := range o.GeneratedTexts {
		out[i] = text
	}
	return out
}

// Pipeline options

// WithVision2SeqMaxTokens sets the maximum number of tokens to generate.
func WithVision2SeqMaxTokens(maxTokens int) backends.PipelineOption[*Vision2SeqPipeline] {
	return func(p *Vision2SeqPipeline) error {
		if maxTokens <= 0 {
			return errors.New("maxTokens must be positive")
		}
		p.MaxNewTokens = maxTokens
		return nil
	}
}

// WithVision2SeqSampling enables top-p (nucleus) sampling with the given temperature.
func WithVision2SeqSampling(topP, temperature float32) backends.PipelineOption[*Vision2SeqPipeline] {
	return func(p *Vision2SeqPipeline) error {
		if topP <= 0 || topP > 1 {
			return errors.New("topP must be in (0, 1]")
		}
		if temperature <= 0 {
			return errors.New("temperature must be positive")
		}
		p.DoSample = true
		p.TopP = topP
		p.Temperature = temperature
		return nil
	}
}

// WithVision2SeqImageSize overrides the image size for preprocessing.
func WithVision2SeqImageSize(size int) backends.PipelineOption[*Vision2SeqPipeline] {
	return func(p *Vision2SeqPipeline) error {
		if size <= 0 {
			return errors.New("image size must be positive")
		}
		p.ImageSize = size
		return nil
	}
}

// NewVision2SeqPipeline creates a new vision-to-sequence pipeline.
func NewVision2SeqPipeline(
	config backends.PipelineConfig[*Vision2SeqPipeline],
	opts *options.Options,
) (*Vision2SeqPipeline, error) {
	pipeline := &Vision2SeqPipeline{
		PipelineName:    config.Name,
		PipelineTimings: &vision2seqTimings{},
		Runtime:         opts.Backend,

		// Defaults
		MaxNewTokens:      128,
		DoSample:          false,
		TopP:              0.9,
		Temperature:       1.0,
		RepetitionPenalty: 1.0,
		ImageSize:         384, // TrOCR default
		imageFormat:       "NCHW",
	}

	// Apply user options
	for _, opt := range config.Options {
		if err := opt(pipeline); err != nil {
			return nil, fmt.Errorf("applying option: %w", err)
		}
	}

	// Load models
	if err := pipeline.loadModels(config.ModelPath, opts); err != nil {
		return nil, fmt.Errorf("loading models: %w", err)
	}

	// Set default preprocessing steps.
	// ResizeToExactStep is used instead of ResizeStep because vision encoders
	// (like TrOCR's ViT) require exact input dimensions. ResizeStep preserves
	// aspect ratio which would produce non-square images.
	if len(pipeline.preprocessSteps) == 0 {
		pipeline.preprocessSteps = []imageutil.PreprocessStep{
			imageutil.ResizeToExactStep(pipeline.ImageSize, pipeline.ImageSize),
		}
	}
	if len(pipeline.normalizationSteps) == 0 {
		pipeline.normalizationSteps = []imageutil.NormalizationStep{
			imageutil.RescaleStep(),
			imageutil.ImagenetPixelNormalizationStep(),
		}
	}

	// Validate pipeline
	if err := pipeline.Validate(); err != nil {
		return nil, fmt.Errorf("validation: %w", err)
	}

	return pipeline, nil
}

// loadModels loads the encoder, decoder, and tokenizer.
func (p *Vision2SeqPipeline) loadModels(modelPath string, opts *options.Options) error {
	var err error

	// Load model config FIRST - we need NumDecoderLayers for decoder loading
	if err := p.loadConfig(modelPath); err != nil {
		return fmt.Errorf("loading config: %w", err)
	}

	// Load vision encoder
	p.EncoderModel, err = backends.LoadVision2SeqEncoder(modelPath, opts)
	if err != nil {
		return fmt.Errorf("loading encoder: %w", err)
	}

	// Try to load split decoders first (init + with_past)
	// This is more reliable than merged decoders
	initDecoder, initErr := backends.LoadVision2SeqDecoderInit(modelPath, opts, p.NumDecoderLayers)
	withPastDecoder, withPastErr := backends.LoadVision2SeqDecoderWithPast(modelPath, opts, p.NumDecoderLayers)

	if initErr == nil && withPastErr == nil {
		// Successfully loaded split decoders
		p.DecoderInitModel = initDecoder
		p.DecoderWithPastModel = withPastDecoder
		p.useSplitDecoder = true
	} else {
		// Fall back to merged decoder
		p.DecoderModel, err = backends.LoadVision2SeqDecoder(modelPath, opts, p.NumDecoderLayers)
		if err != nil {
			// If merged also fails, return the original split decoder errors
			if initErr != nil {
				return fmt.Errorf("loading init decoder: %w", initErr)
			}
			return fmt.Errorf("loading with-past decoder: %w", withPastErr)
		}
		p.useSplitDecoder = false
	}

	// Load tokenizer
	p.Tokenizer, err = backends.LoadVision2SeqTokenizer(modelPath, opts)
	if err != nil {
		return fmt.Errorf("loading tokenizer: %w", err)
	}

	return nil
}

// loadConfig loads model configuration from config.json.
func (p *Vision2SeqPipeline) loadConfig(modelPath string) error {
	config, err := backends.LoadVision2SeqConfig(modelPath)
	if err != nil {
		return err
	}

	p.DecoderStartTokenID = config.DecoderStartTokenID
	p.EosTokenIDs = config.EosTokenIDs
	p.PadTokenID = config.PadTokenID
	p.VocabSize = config.VocabSize
	p.NumHeads = config.NumHeads
	p.HeadDim = config.HeadDim
	p.NumDecoderLayers = config.NumDecoderLayers

	// Override image size if not set by user
	if config.ImageSize > 0 && p.ImageSize == 384 {
		p.ImageSize = config.ImageSize
	}

	// Set normalization from preprocessor config if available
	if len(config.ImageMean) == 3 && len(config.ImageStd) == 3 {
		p.normalizationSteps = []imageutil.NormalizationStep{
			imageutil.RescaleStep(),
			imageutil.PixelNormalizationStep(
				[3]float32{config.ImageMean[0], config.ImageMean[1], config.ImageMean[2]},
				[3]float32{config.ImageStd[0], config.ImageStd[1], config.ImageStd[2]},
			),
		}
	}

	return nil
}

// Validate checks that the pipeline is correctly configured.
func (p *Vision2SeqPipeline) Validate() error {
	var errs []error

	if p.EncoderModel == nil {
		errs = append(errs, errors.New("encoder model not loaded"))
	}
	// Check decoder models based on split/merged mode
	if p.useSplitDecoder {
		if p.DecoderInitModel == nil {
			errs = append(errs, errors.New("init decoder model not loaded"))
		}
		if p.DecoderWithPastModel == nil {
			errs = append(errs, errors.New("with-past decoder model not loaded"))
		}
	} else {
		if p.DecoderModel == nil {
			errs = append(errs, errors.New("decoder model not loaded"))
		}
	}
	if p.Tokenizer == nil {
		errs = append(errs, errors.New("tokenizer not loaded"))
	}
	if len(p.EosTokenIDs) == 0 {
		errs = append(errs, errors.New("no EOS token IDs configured"))
	}
	if p.VocabSize == 0 {
		errs = append(errs, errors.New("vocab size is 0"))
	}
	if p.MaxNewTokens <= 0 {
		errs = append(errs, errors.New("maxNewTokens must be positive"))
	}
	if p.NumHeads <= 0 {
		errs = append(errs, errors.New("num_heads must be positive (check model config)"))
	}
	if p.HeadDim <= 0 {
		errs = append(errs, errors.New("head_dim must be positive (check model config)"))
	}
	if p.NumDecoderLayers <= 0 {
		errs = append(errs, errors.New("num_decoder_layers must be positive (check model config)"))
	}

	return errors.Join(errs...)
}

// GetModel returns the encoder model (primary model for the pipeline).
func (p *Vision2SeqPipeline) GetModel() *backends.Model {
	return p.EncoderModel
}

// GetMetadata returns pipeline metadata.
func (p *Vision2SeqPipeline) GetMetadata() backends.PipelineMetadata {
	return backends.PipelineMetadata{}
}

// GetStatistics returns runtime statistics for the pipeline.
func (p *Vision2SeqPipeline) GetStatistics() backends.PipelineStatistics {
	stats := backends.PipelineStatistics{}

	// Combined timing stats
	totalOnnxNS := p.PipelineTimings.EncoderTotalNS + p.PipelineTimings.DecoderTotalNS
	totalOnnxCalls := p.PipelineTimings.EncoderNumCalls + p.PipelineTimings.DecoderNumCalls
	stats.OnnxTotalTime = safeconv.U64ToDuration(totalOnnxNS)
	stats.OnnxExecutionCount = totalOnnxCalls

	return stats
}

// IsGenerative returns true as Vision2Seq is a generative model.
func (p *Vision2SeqPipeline) IsGenerative() bool {
	return true
}

// Run is not implemented for Vision2Seq - use RunWithImages instead.
func (p *Vision2SeqPipeline) Run(inputs []string) (backends.PipelineBatchOutput, error) {
	return nil, errors.New("Vision2SeqPipeline does not support text input; use RunWithImages")
}

// RunWithImages runs the pipeline on a batch of images.
func (p *Vision2SeqPipeline) RunWithImages(images []image.Image) (*Vision2SeqOutput, error) {
	batch := backends.NewVision2SeqBatch(len(images))
	batch.Images = images
	defer batch.Destroy()

	// 1. Preprocess images
	if err := p.Preprocess(batch); err != nil {
		return nil, fmt.Errorf("preprocess: %w", err)
	}

	// 2. Run encoder
	if err := p.Encode(batch); err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}

	// 3. Generate sequences
	if err := p.Generate(batch); err != nil {
		return nil, fmt.Errorf("generate: %w", err)
	}

	// 4. Decode tokens to text
	output, err := p.Postprocess(batch)
	if err != nil {
		return nil, fmt.Errorf("postprocess: %w", err)
	}

	return output, nil
}

// RunWithImagePaths loads images from file paths and runs the pipeline.
func (p *Vision2SeqPipeline) RunWithImagePaths(paths []string) (*Vision2SeqOutput, error) {
	images, err := imageutil.LoadImagesFromPaths(paths)
	if err != nil {
		return nil, fmt.Errorf("failed to load images: %w", err)
	}
	return p.RunWithImages(images)
}

// Preprocess converts images to tensors.
func (p *Vision2SeqPipeline) Preprocess(batch *backends.Vision2SeqBatch) error {
	start := time.Now()

	preprocessed, err := backends.PreprocessImages(
		p.imageFormat,
		batch.Images,
		p.preprocessSteps,
		p.normalizationSteps,
	)
	if err != nil {
		return err
	}

	batch.PreprocessedData = preprocessed

	atomic.AddUint64(&p.PipelineTimings.PreprocessNumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.PreprocessTotalNS, safeconv.DurationToU64(time.Since(start)))

	return nil
}

// Encode runs the vision encoder.
func (p *Vision2SeqPipeline) Encode(batch *backends.Vision2SeqBatch) error {
	start := time.Now()

	err := backends.RunVision2SeqEncoder(batch, p.EncoderModel, p.Runtime)
	if err != nil {
		return err
	}

	atomic.AddUint64(&p.PipelineTimings.EncoderNumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.EncoderTotalNS, safeconv.DurationToU64(time.Since(start)))

	return nil
}

// Generate performs autoregressive decoding.
func (p *Vision2SeqPipeline) Generate(batch *backends.Vision2SeqBatch) error {
	start := time.Now()

	var err error
	if p.DoSample {
		err = backends.RunVision2SeqGenerationSampling(batch, p)
	} else {
		err = backends.RunVision2SeqGenerationGreedy(batch, p)
	}

	atomic.AddUint64(&p.PipelineTimings.DecoderNumCalls, uint64(p.MaxNewTokens))
	atomic.AddUint64(&p.PipelineTimings.DecoderTotalNS, safeconv.DurationToU64(time.Since(start)))

	return err
}

// Postprocess decodes the generated token IDs back to text.
func (p *Vision2SeqPipeline) Postprocess(batch *backends.Vision2SeqBatch) (*Vision2SeqOutput, error) {
	output := &Vision2SeqOutput{
		GeneratedTexts:  make([]string, batch.Size),
		GeneratedTokens: make([][]uint32, batch.Size),
	}

	for i := 0; i < batch.Size; i++ {
		tokens := batch.GeneratedTokens[i]
		convertedTokens := make([]uint32, len(tokens))
		for j, tok := range tokens {
			convertedTokens[j] = safeconv.Int64ToUint32(tok)
		}

		text, err := backends.Decode(convertedTokens, p.Tokenizer, true)
		if err != nil {
			return nil, fmt.Errorf("decoding output %d: %w", i, err)
		}

		output.GeneratedTexts[i] = text
		output.GeneratedTokens[i] = convertedTokens
	}

	return output, nil
}

// Destroy cleans up pipeline resources.
func (p *Vision2SeqPipeline) Destroy() error {
	var errs []error

	if p.EncoderModel != nil && p.EncoderModel.Destroy != nil {
		errs = append(errs, p.EncoderModel.Destroy())
	}
	// Clean up decoder models (either merged or split)
	if p.DecoderModel != nil && p.DecoderModel.Destroy != nil {
		errs = append(errs, p.DecoderModel.Destroy())
	}
	if p.DecoderInitModel != nil && p.DecoderInitModel.Destroy != nil {
		errs = append(errs, p.DecoderInitModel.Destroy())
	}
	if p.DecoderWithPastModel != nil && p.DecoderWithPastModel.Destroy != nil {
		errs = append(errs, p.DecoderWithPastModel.Destroy())
	}
	if p.Tokenizer != nil {
		errs = append(errs, p.Tokenizer.Destroy())
	}

	return errors.Join(errs...)
}

// Interface implementations for backends.Vision2SeqPipelineInterface

func (p *Vision2SeqPipeline) GetEncoderModel() *backends.Model  { return p.EncoderModel }
func (p *Vision2SeqPipeline) GetDecoderModel() *backends.Model  { return p.DecoderModel }
func (p *Vision2SeqPipeline) GetTokenizer() *backends.Tokenizer { return p.Tokenizer }
func (p *Vision2SeqPipeline) GetRuntime() string                { return p.Runtime }
func (p *Vision2SeqPipeline) GetMaxNewTokens() int              { return p.MaxNewTokens }
func (p *Vision2SeqPipeline) GetDoSample() bool                 { return p.DoSample }
func (p *Vision2SeqPipeline) GetTopP() float32                  { return p.TopP }
func (p *Vision2SeqPipeline) GetTemperature() float32           { return p.Temperature }
func (p *Vision2SeqPipeline) GetRepetitionPenalty() float32     { return p.RepetitionPenalty }
func (p *Vision2SeqPipeline) GetDecoderStartTokenID() int64     { return p.DecoderStartTokenID }
func (p *Vision2SeqPipeline) GetEosTokenIDs() map[int64]bool    { return p.EosTokenIDs }
func (p *Vision2SeqPipeline) GetPadTokenID() int64              { return p.PadTokenID }
func (p *Vision2SeqPipeline) GetVocabSize() int                 { return p.VocabSize }
func (p *Vision2SeqPipeline) GetImageSize() int                 { return p.ImageSize }
func (p *Vision2SeqPipeline) GetNumHeads() int                  { return p.NumHeads }
func (p *Vision2SeqPipeline) GetHeadDim() int                   { return p.HeadDim }

// Split decoder methods
func (p *Vision2SeqPipeline) GetDecoderInitModel() *backends.Model     { return p.DecoderInitModel }
func (p *Vision2SeqPipeline) GetDecoderWithPastModel() *backends.Model { return p.DecoderWithPastModel }
func (p *Vision2SeqPipeline) UseSplitDecoder() bool                    { return p.useSplitDecoder }
