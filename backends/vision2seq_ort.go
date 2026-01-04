//go:build ORT || ALL

package backends

import (
	"errors"
	"fmt"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

// RunVision2SeqEncoderORT runs the vision encoder on preprocessed images using ORT backend.
func RunVision2SeqEncoder(batch Vision2SeqBatchInterface, model *Model, runtime string) error {
	if runtime != "ORT" {
		return fmt.Errorf("unsupported runtime for vision2seq encoder: %s", runtime)
	}

	return runVision2SeqEncoderORT(batch, model)
}

func runVision2SeqEncoderORT(batch Vision2SeqBatchInterface, model *Model) error {
	preprocessed := batch.GetPreprocessedImages()
	if len(preprocessed) == 0 {
		return errors.New("no preprocessed images provided")
	}

	batchSize := batch.GetSize()
	c := len(preprocessed[0])
	h := len(preprocessed[0][0])
	w := len(preprocessed[0][0][0])

	// Flatten image tensor to 1D array
	flatData := make([]float32, batchSize*c*h*w)
	idx := 0
	for i := 0; i < batchSize; i++ {
		for ch := 0; ch < c; ch++ {
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					flatData[idx] = preprocessed[i][ch][y][x]
					idx++
				}
			}
		}
	}

	// Create pixel_values tensor
	pixelValuesTensor, err := ort.NewTensor(
		ort.NewShape(int64(batchSize), int64(c), int64(h), int64(w)),
		flatData,
	)
	if err != nil {
		return fmt.Errorf("creating pixel_values tensor: %w", err)
	}

	// Create output tensor slice - DynamicAdvancedSession will allocate tensors automatically.
	// This avoids needing to pre-compute output dimensions which vary by encoder architecture.
	inputs := []ort.Value{pixelValuesTensor}
	outputs := make([]ort.Value, len(model.OutputsMeta))

	// Run encoder - ORT allocates output tensors dynamically based on actual model output shape
	if err := model.ORTModel.Session.Run(inputs, outputs); err != nil {
		pixelValuesTensor.Destroy()
		for _, o := range outputs {
			if o != nil {
				o.Destroy()
			}
		}
		return fmt.Errorf("running vision encoder: %w", err)
	}

	// Find and validate the encoder hidden states output
	var outputTensor ort.Value
	for i, meta := range model.OutputsMeta {
		if strings.Contains(meta.Name, "hidden_states") || meta.Name == "last_hidden_state" {
			outputTensor = outputs[i]
			break
		}
	}
	if outputTensor == nil && len(outputs) > 0 {
		// Fallback to first output if no named match
		outputTensor = outputs[0]
	}
	if outputTensor == nil {
		pixelValuesTensor.Destroy()
		return errors.New("no encoder output tensor produced")
	}

	// Store encoder outputs in batch (keep tensors alive for decoder)
	batch.SetEncoderHiddenStates(outputTensor)

	// Set cleanup function - destroy input tensor and any extra outputs
	batch.SetDestroyEncoder(func() error {
		var errs []error
		errs = append(errs, pixelValuesTensor.Destroy())
		// Destroy any outputs that aren't the main hidden states (in case there are multiple)
		for _, o := range outputs {
			if o != nil && o != outputTensor {
				errs = append(errs, o.Destroy())
			}
		}
		// outputTensor is destroyed after generation
		return errors.Join(errs...)
	})

	return nil
}

// RunVision2SeqGenerationGreedy performs greedy decoding for vision2seq.
func RunVision2SeqGenerationGreedy(batch Vision2SeqBatchInterface, pipeline Vision2SeqPipelineInterface) error {
	if pipeline.GetRuntime() != "ORT" {
		return fmt.Errorf("unsupported runtime: %s", pipeline.GetRuntime())
	}

	// Greedy selection: argmax over vocabulary
	selector := func(logits []float32, batchSize, vocabSize int) []int64 {
		return argmaxVision2Seq(logits, batchSize, vocabSize)
	}

	return runVision2SeqGenerationORT(batch, pipeline, selector)
}

// RunVision2SeqGenerationSampling performs top-p sampling for vision2seq.
func RunVision2SeqGenerationSampling(batch Vision2SeqBatchInterface, pipeline Vision2SeqPipelineInterface) error {
	if pipeline.GetRuntime() != "ORT" {
		return fmt.Errorf("unsupported runtime: %s", pipeline.GetRuntime())
	}

	topP := pipeline.GetTopP()
	temperature := pipeline.GetTemperature()

	selector := func(logits []float32, batchSize, vocabSize int) []int64 {
		return sampleTopP(logits, batchSize, vocabSize, topP, temperature)
	}

	return runVision2SeqGenerationORT(batch, pipeline, selector)
}

// runVision2SeqGenerationORT is the unified generation loop for ORT backend.
func runVision2SeqGenerationORT(batch Vision2SeqBatchInterface, pipeline Vision2SeqPipelineInterface, selectTokens tokenSelector) error {
	batchSize := batch.GetSize()
	maxNewTokens := pipeline.GetMaxNewTokens()
	eosTokenIDs := pipeline.GetEosTokenIDs()
	decoderStartToken := pipeline.GetDecoderStartTokenID()
	vocabSize := pipeline.GetVocabSize()
	numHeads := pipeline.GetNumHeads()
	headDim := pipeline.GetHeadDim()

	if numHeads <= 0 || headDim <= 0 {
		return fmt.Errorf("invalid model config: numHeads=%d, headDim=%d", numHeads, headDim)
	}

	// Initialize generation state
	generatedTokens := make([][]int64, batchSize)
	finished := make([]bool, batchSize)
	finishedCount := 0
	for i := range generatedTokens {
		generatedTokens[i] = []int64{}
	}

	// Get encoder outputs
	encoderHiddenStates := batch.GetEncoderHiddenStates().(ort.Value)

	// Decoder input starts with decoder_start_token_id
	decoderInputIDs := make([]int64, batchSize)
	for i := range decoderInputIDs {
		decoderInputIDs[i] = decoderStartToken
	}

	// Check if using split decoders (separate init and with-past models)
	useSplitDecoder := pipeline.UseSplitDecoder()
	var decoderInitModel, decoderWithPastModel, mergedDecoderModel *Model
	if useSplitDecoder {
		decoderInitModel = pipeline.GetDecoderInitModel()
		decoderWithPastModel = pipeline.GetDecoderWithPastModel()
	} else {
		mergedDecoderModel = pipeline.GetDecoderModel()
	}

	var pastKeyValues []ort.Value

	// For split decoders, encoder PKV is computed once and reused
	var encoderPKV []ort.Value // Kept separately for split decoder

	// Prompt prefill phase: process prompt tokens if present
	// For DocVQA, the prompt is like: <s_docvqa><s_question>What is...?</s_question><s_answer>
	promptTokenIDs := batch.GetPromptTokenIDs()
	promptLength := 0
	if len(promptTokenIDs) > 0 && len(promptTokenIDs[0]) > 0 {
		promptLength = len(promptTokenIDs[0])

		// Feed prompt tokens through decoder to build up past key values
		for promptIdx := 0; promptIdx < promptLength; promptIdx++ {
			// Get current prompt token for each batch item
			for i := 0; i < batchSize; i++ {
				if promptIdx < len(promptTokenIDs[i]) {
					decoderInputIDs[i] = promptTokenIDs[i][promptIdx]
				}
			}

			var logits []float32
			var newPKV []ort.Value
			var err error

			if promptIdx == 0 {
				// First prompt token: no past_key_values
				if useSplitDecoder {
					logits, newPKV, err = runVision2SeqDecoderInitSplitORT(
						decoderInputIDs, encoderHiddenStates,
						decoderInitModel, batchSize, vocabSize, numHeads, headDim,
					)
					if err == nil {
						pastKeyValues = make([]ort.Value, 0, len(newPKV)/2)
						encoderPKV = make([]ort.Value, 0, len(newPKV)/2)
						for i := 0; i < len(newPKV); i += 4 {
							pastKeyValues = append(pastKeyValues, newPKV[i], newPKV[i+1])
							encoderPKV = append(encoderPKV, newPKV[i+2], newPKV[i+3])
						}
					}
				} else {
					logits, newPKV, err = runVision2SeqDecoderInitORT(
						decoderInputIDs, encoderHiddenStates,
						mergedDecoderModel, batchSize, vocabSize, numHeads, headDim,
					)
					pastKeyValues = newPKV
				}
			} else {
				// Subsequent prompt tokens: with past_key_values
				if useSplitDecoder {
					fullPKV := make([]ort.Value, 0, len(pastKeyValues)+len(encoderPKV))
					for i := 0; i < len(pastKeyValues); i += 2 {
						fullPKV = append(fullPKV, pastKeyValues[i], pastKeyValues[i+1])
						fullPKV = append(fullPKV, encoderPKV[i], encoderPKV[i+1])
					}

					var decoderPKV []ort.Value
					logits, decoderPKV, err = runVision2SeqDecoderStepSplitORT(
						decoderInputIDs, encoderHiddenStates, fullPKV,
						decoderWithPastModel, batchSize, vocabSize, promptIdx, numHeads, headDim,
					)

					for _, pkv := range pastKeyValues {
						pkv.Destroy()
					}
					pastKeyValues = decoderPKV
				} else {
					logits, newPKV, err = runVision2SeqDecoderStepORT(
						decoderInputIDs, encoderHiddenStates, pastKeyValues,
						mergedDecoderModel, batchSize, vocabSize, promptIdx, numHeads, headDim,
					)

					for _, pkv := range pastKeyValues {
						pkv.Destroy()
					}
					pastKeyValues = newPKV
				}
			}

			if err != nil {
				for _, pkv := range pastKeyValues {
					pkv.Destroy()
				}
				for _, pkv := range encoderPKV {
					pkv.Destroy()
				}
				return fmt.Errorf("prompt prefill step %d: %w", promptIdx, err)
			}

			// Discard logits during prefill (we don't need them)
			_ = logits
		}
	}

	// Main generation loop (after any prompt prefill)
	for step := 0; step < maxNewTokens; step++ {
		if finishedCount == batchSize {
			break
		}

		// effectiveStep accounts for prompt tokens already processed
		effectiveStep := step + promptLength

		var logits []float32
		var newPKV []ort.Value
		var err error

		// Use init decoder only on first step with no prior prefill
		if step == 0 && promptLength == 0 {
			// First step: no past_key_values
			if useSplitDecoder {
				logits, newPKV, err = runVision2SeqDecoderInitSplitORT(
					decoderInputIDs, encoderHiddenStates,
					decoderInitModel, batchSize, vocabSize, numHeads, headDim,
				)
				if err == nil {
					// Split init PKV into decoder and encoder
					// Layout: [dec0.key, dec0.val, enc0.key, enc0.val, ...]
					pastKeyValues = make([]ort.Value, 0, len(newPKV)/2)
					encoderPKV = make([]ort.Value, 0, len(newPKV)/2)
					for i := 0; i < len(newPKV); i += 4 {
						pastKeyValues = append(pastKeyValues, newPKV[i], newPKV[i+1]) // decoder
						encoderPKV = append(encoderPKV, newPKV[i+2], newPKV[i+3])    // encoder
					}
				}
			} else {
				logits, newPKV, err = runVision2SeqDecoderInitORT(
					decoderInputIDs, encoderHiddenStates,
					mergedDecoderModel, batchSize, vocabSize, numHeads, headDim,
				)
				pastKeyValues = newPKV
			}
		} else {
			// Subsequent steps or steps after prompt prefill: with past_key_values
			if useSplitDecoder {
				// Build full PKV by interleaving decoder and encoder
				fullPKV := make([]ort.Value, 0, len(pastKeyValues)+len(encoderPKV))
				for i := 0; i < len(pastKeyValues); i += 2 {
					fullPKV = append(fullPKV, pastKeyValues[i], pastKeyValues[i+1]) // decoder
					fullPKV = append(fullPKV, encoderPKV[i], encoderPKV[i+1])       // encoder
				}

				var decoderPKV []ort.Value
				logits, decoderPKV, err = runVision2SeqDecoderStepSplitORT(
					decoderInputIDs, encoderHiddenStates, fullPKV,
					decoderWithPastModel, batchSize, vocabSize, effectiveStep, numHeads, headDim,
				)

				// Destroy old decoder PKV only (encoder PKV is reused)
				for _, pkv := range pastKeyValues {
					pkv.Destroy()
				}
				pastKeyValues = decoderPKV
			} else {
				logits, newPKV, err = runVision2SeqDecoderStepORT(
					decoderInputIDs, encoderHiddenStates, pastKeyValues,
					mergedDecoderModel, batchSize, vocabSize, effectiveStep, numHeads, headDim,
				)

				// Destroy old PKV
				for _, pkv := range pastKeyValues {
					pkv.Destroy()
				}
				pastKeyValues = newPKV
			}
		}

		if err != nil {
			// Cleanup on error
			for _, pkv := range pastKeyValues {
				pkv.Destroy()
			}
			for _, pkv := range encoderPKV {
				pkv.Destroy()
			}
			return err
		}

		// Select next tokens
		nextTokens := selectTokens(logits, batchSize, vocabSize)

		// Update decoder input for next step
		decoderInputIDs = nextTokens

		// Append tokens and check for EOS
		for i := 0; i < batchSize; i++ {
			if finished[i] {
				continue
			}

			tok := nextTokens[i]
			generatedTokens[i] = append(generatedTokens[i], tok)

			if eosTokenIDs[tok] {
				finished[i] = true
				finishedCount++
			}
		}
	}

	// Store results
	batch.SetGeneratedTokens(generatedTokens)
	batch.SetFinished(finished)
	batch.SetFinishedCount(finishedCount)

	// Set cleanup for decoder resources
	// Capture encoderPKV in closure for split decoder cleanup
	capturedEncoderPKV := encoderPKV
	batch.SetDestroyDecoder(func() error {
		var errs []error
		for _, pkv := range pastKeyValues {
			errs = append(errs, pkv.Destroy())
		}
		// Also destroy encoder PKV for split decoders
		for _, pkv := range capturedEncoderPKV {
			errs = append(errs, pkv.Destroy())
		}
		// Also destroy encoder outputs now
		if hs, ok := batch.GetEncoderHiddenStates().(ort.Value); ok {
			errs = append(errs, hs.Destroy())
		}
		return errors.Join(errs...)
	})

	return nil
}

// runVision2SeqDecoderInitORT runs the initial decoder step (no past_key_values).
func runVision2SeqDecoderInitORT(
	decoderInputIDs []int64,
	encoderHiddenStates ort.Value,
	model *Model,
	batchSize, vocabSize, numHeads, headDim int,
) ([]float32, []ort.Value, error) {
	// Get encoder sequence length from encoder hidden states (required, no default)
	encoderSeqLen := 0
	if hs, ok := encoderHiddenStates.(*ort.Tensor[float32]); ok {
		shape := hs.GetShape()
		if len(shape) >= 2 {
			encoderSeqLen = int(shape[1])
		}
	}
	if encoderSeqLen == 0 {
		return nil, nil, errors.New("cannot determine encoder sequence length from hidden states")
	}

	// Create decoder input tensor
	inputTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), 1), decoderInputIDs)
	if err != nil {
		return nil, nil, fmt.Errorf("creating decoder input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Build inputs based on model's expected inputs
	inputs, inputDestroys, err := buildVision2SeqDecoderInputs(model, inputTensor, encoderHiddenStates, nil, batchSize, true, numHeads, headDim)
	if err != nil {
		return nil, nil, err
	}
	defer func() {
		for _, d := range inputDestroys {
			d()
		}
	}()

	// Create output tensors
	numOutputs := len(model.OutputsMeta)
	outputs := make([]ort.Value, numOutputs)

	// Logits output (typically first)
	logitsTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(batchSize), 1, int64(vocabSize)))
	if err != nil {
		return nil, nil, fmt.Errorf("creating logits tensor: %w", err)
	}
	outputs[0] = logitsTensor

	// Past key values outputs
	// Encoder-decoder models have two types per layer:
	// - decoder self-attention: present.X.decoder.key/value (seq_len = 1 for init)
	// - encoder cross-attention: present.X.encoder.key/value (seq_len = encoder seq len)
	for i := 1; i < numOutputs; i++ {
		meta := model.OutputsMeta[i]
		name := strings.ToLower(meta.Name)
		isEncoderPKV := strings.Contains(name, "encoder")

		var seqLen int
		if isEncoderPKV {
			seqLen = encoderSeqLen
		} else {
			seqLen = 1 // decoder self-attention starts with seq_len=1
		}

		shape := inferVision2SeqPKVShape(meta.Dimensions, batchSize, seqLen, numHeads, headDim)
		pkvTensor, err := ort.NewEmptyTensor[float32](shape)
		if err != nil {
			for j := 0; j < i; j++ {
				outputs[j].Destroy()
			}
			return nil, nil, fmt.Errorf("creating pkv tensor %d (%s): %w", i, meta.Name, err)
		}
		outputs[i] = pkvTensor
	}

	// Run decoder
	if err := model.ORTModel.Session.Run(inputs, outputs); err != nil {
		for _, o := range outputs {
			o.Destroy()
		}
		return nil, nil, fmt.Errorf("running decoder-init: %w", err)
	}

	// Extract logits
	logits := outputs[0].(*ort.Tensor[float32]).GetData()

	// Keep past_key_values
	pastKeyValues := outputs[1:]

	// Destroy logits tensor (we copied the data)
	outputs[0].Destroy()

	return logits, pastKeyValues, nil
}

// runVision2SeqDecoderStepORT runs a decoder step with past_key_values.
func runVision2SeqDecoderStepORT(
	decoderInputIDs []int64,
	encoderHiddenStates ort.Value,
	pastKeyValues []ort.Value,
	model *Model,
	batchSize, vocabSize, pastSeqLen, numHeads, headDim int,
) ([]float32, []ort.Value, error) {
	// Get encoder sequence length from encoder hidden states (required, no default)
	encoderSeqLen := 0
	if hs, ok := encoderHiddenStates.(*ort.Tensor[float32]); ok {
		shape := hs.GetShape()
		if len(shape) >= 2 {
			encoderSeqLen = int(shape[1])
		}
	}
	if encoderSeqLen == 0 {
		return nil, nil, errors.New("cannot determine encoder sequence length from hidden states")
	}

	// Create decoder input tensor
	inputTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), 1), decoderInputIDs)
	if err != nil {
		return nil, nil, fmt.Errorf("creating decoder input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Build inputs with past_key_values
	inputs, inputDestroys, err := buildVision2SeqDecoderInputs(model, inputTensor, encoderHiddenStates, pastKeyValues, batchSize, false, numHeads, headDim)
	if err != nil {
		return nil, nil, err
	}
	defer func() {
		for _, d := range inputDestroys {
			d()
		}
	}()

	// Create output tensors
	numOutputs := len(model.OutputsMeta)
	outputs := make([]ort.Value, numOutputs)

	// Logits output
	logitsTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(batchSize), 1, int64(vocabSize)))
	if err != nil {
		return nil, nil, fmt.Errorf("creating logits tensor: %w", err)
	}
	outputs[0] = logitsTensor

	// Present key values outputs
	// - Decoder self-attention: seq_len grows (pastSeqLen + 1)
	// - Encoder cross-attention: seq_len stays constant (encoderSeqLen)
	newDecoderSeqLen := pastSeqLen + 1
	for i := 1; i < numOutputs; i++ {
		meta := model.OutputsMeta[i]
		name := strings.ToLower(meta.Name)
		isEncoderPKV := strings.Contains(name, "encoder")

		var seqLen int
		if isEncoderPKV {
			seqLen = encoderSeqLen
		} else {
			seqLen = newDecoderSeqLen
		}

		shape := inferVision2SeqPKVShape(meta.Dimensions, batchSize, seqLen, numHeads, headDim)
		pkvTensor, err := ort.NewEmptyTensor[float32](shape)
		if err != nil {
			for j := 0; j < i; j++ {
				outputs[j].Destroy()
			}
			return nil, nil, fmt.Errorf("creating present tensor %d: %w", i, err)
		}
		outputs[i] = pkvTensor
	}

	// Run decoder
	if err := model.ORTModel.Session.Run(inputs, outputs); err != nil {
		for _, o := range outputs {
			o.Destroy()
		}
		return nil, nil, fmt.Errorf("running decoder: %w", err)
	}

	// Extract logits
	logits := outputs[0].(*ort.Tensor[float32]).GetData()

	// Keep present key values
	presentKeyValues := outputs[1:]

	outputs[0].Destroy()

	return logits, presentKeyValues, nil
}

// buildVision2SeqDecoderInputs builds the input slice for the decoder.
// The merged decoder can take different inputs depending on whether we have past_key_values.
func buildVision2SeqDecoderInputs(
	model *Model,
	inputIDsTensor, encoderHiddenStates ort.Value,
	pastKeyValues []ort.Value,
	batchSize int,
	isInit bool,
	numHeads, headDim int,
) ([]ort.Value, []func(), error) {
	inputs := make([]ort.Value, len(model.InputsMeta))
	var destroys []func()

	pkvIndex := 0

	// Get encoder sequence length from encoder hidden states (required, no default)
	encoderSeqLen := 0
	if hs, ok := encoderHiddenStates.(*ort.Tensor[float32]); ok {
		shape := hs.GetShape()
		if len(shape) >= 2 {
			encoderSeqLen = int(shape[1])
		}
	}
	if encoderSeqLen == 0 {
		return nil, nil, errors.New("cannot determine encoder sequence length from hidden states")
	}

	for i, meta := range model.InputsMeta {
		name := strings.ToLower(meta.Name)

		switch {
		case strings.Contains(name, "input_ids"):
			inputs[i] = inputIDsTensor

		case strings.Contains(name, "encoder_hidden_states"):
			inputs[i] = encoderHiddenStates

		case strings.Contains(name, "encoder_attention_mask"):
			// Create all-ones mask for encoder
			maskData := make([]int64, batchSize*encoderSeqLen)
			for j := range maskData {
				maskData[j] = 1
			}
			maskTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), int64(encoderSeqLen)), maskData)
			if err != nil {
				return nil, destroys, fmt.Errorf("creating encoder attention mask: %w", err)
			}
			inputs[i] = maskTensor
			destroys = append(destroys, func() { maskTensor.Destroy() })

		case strings.Contains(name, "use_cache_branch"):
			// Boolean tensor indicating whether to use cache
			// On init: false (no cache), subsequent: true (use cache)
			useCacheData := make([]bool, 1)
			useCacheData[0] = !isInit
			useCacheTensor, err := ort.NewTensor(ort.NewShape(1), useCacheData)
			if err != nil {
				return nil, destroys, fmt.Errorf("creating use_cache_branch: %w", err)
			}
			inputs[i] = useCacheTensor
			destroys = append(destroys, func() { useCacheTensor.Destroy() })

		case strings.Contains(name, "past_key_values") || strings.Contains(name, "past"):
			// Past key values input
			// Encoder-decoder models have two types of PKV per layer:
			// - decoder self-attention: past_key_values.X.decoder.key/value (seq_len = decoder steps)
			// - encoder cross-attention: past_key_values.X.encoder.key/value (seq_len = encoder seq len)
			isEncoderPKV := strings.Contains(name, "encoder")

			if isInit || pkvIndex >= len(pastKeyValues) {
				// Create empty/dummy tensor for init step
				var seqLen int
				if isEncoderPKV {
					// Encoder cross-attention PKV uses encoder sequence length
					seqLen = encoderSeqLen
				} else {
					// Decoder self-attention PKV starts at 0
					seqLen = 0
				}
				shape := inferVision2SeqPKVShape(meta.Dimensions, batchSize, seqLen, numHeads, headDim)
				emptyTensor, err := ort.NewEmptyTensor[float32](shape)
				if err != nil {
					return nil, destroys, fmt.Errorf("creating empty pkv tensor: %w", err)
				}
				inputs[i] = emptyTensor
				destroys = append(destroys, func() { emptyTensor.Destroy() })
			} else {
				inputs[i] = pastKeyValues[pkvIndex]
				pkvIndex++
			}

		default:
			// Unknown input - try to create a reasonable default
			return nil, destroys, fmt.Errorf("unknown decoder input: %s", meta.Name)
		}
	}

	return inputs, destroys, nil
}

// inferVision2SeqPKVShape infers the shape for past/present key values tensors.
// PKV tensors have shape [batch, num_heads, seq_len, head_dim].
func inferVision2SeqPKVShape(dims Shape, batchSize, seqLen, numHeads, headDim int) ort.Shape {
	// If no dimension info available, use config values
	if len(dims) == 0 {
		return ort.NewShape(int64(batchSize), int64(numHeads), int64(seqLen), int64(headDim))
	}

	shape := make([]int64, len(dims))
	for i, d := range dims {
		// Both -1 and 0 indicate dynamic dimensions in ONNX
		if d == -1 || d == 0 {
			// Dynamic dimension - fill in based on position
			if i == 0 {
				shape[i] = int64(batchSize)
			} else if i == 2 {
				// Sequence length dimension (typically [batch, heads, seq, head_dim])
				shape[i] = int64(seqLen)
			} else if i == 1 {
				// num_heads dimension
				shape[i] = int64(numHeads)
			} else if i == 3 {
				// head_dim dimension
				shape[i] = int64(headDim)
			} else {
				// Unknown dynamic dim - use 1 as fallback
				shape[i] = 1
			}
		} else {
			shape[i] = d
		}
	}
	return ort.NewShape(shape...)
}

// argmaxVision2Seq performs argmax over the last dimension of logits.
func argmaxVision2Seq(logits []float32, batchSize, vocabSize int) []int64 {
	tokens := make([]int64, batchSize)

	for b := 0; b < batchSize; b++ {
		offset := b * vocabSize
		maxIdx := 0
		maxVal := logits[offset]

		for v := 1; v < vocabSize; v++ {
			if logits[offset+v] > maxVal {
				maxVal = logits[offset+v]
				maxIdx = v
			}
		}
		tokens[b] = int64(maxIdx)
	}

	return tokens
}

// runVision2SeqDecoderInitSplitORT runs the init decoder for split decoder models.
// The init decoder takes input_ids and encoder_hidden_states (no past_key_values).
func runVision2SeqDecoderInitSplitORT(
	decoderInputIDs []int64,
	encoderHiddenStates ort.Value,
	model *Model,
	batchSize, vocabSize, numHeads, headDim int,
) ([]float32, []ort.Value, error) {
	// Get encoder sequence length from encoder hidden states
	encoderSeqLen := 0
	if hs, ok := encoderHiddenStates.(*ort.Tensor[float32]); ok {
		shape := hs.GetShape()
		if len(shape) >= 2 {
			encoderSeqLen = int(shape[1])
		}
	}
	if encoderSeqLen == 0 {
		return nil, nil, errors.New("cannot determine encoder sequence length from hidden states")
	}

	// Create decoder input tensor
	inputTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), 1), decoderInputIDs)
	if err != nil {
		return nil, nil, fmt.Errorf("creating decoder input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Build inputs: input_ids, encoder_hidden_states (no past_key_values for init)
	inputs := []ort.Value{inputTensor, encoderHiddenStates}

	// Create output tensors
	numOutputs := len(model.OutputsMeta)
	outputs := make([]ort.Value, numOutputs)

	// Logits output (first)
	logitsTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(batchSize), 1, int64(vocabSize)))
	if err != nil {
		return nil, nil, fmt.Errorf("creating logits tensor: %w", err)
	}
	outputs[0] = logitsTensor

	// Present key values outputs
	for i := 1; i < numOutputs; i++ {
		meta := model.OutputsMeta[i]
		name := strings.ToLower(meta.Name)
		isEncoderPKV := strings.Contains(name, "encoder")

		var seqLen int
		if isEncoderPKV {
			seqLen = encoderSeqLen
		} else {
			seqLen = 1 // decoder self-attention starts with seq_len=1
		}

		shape := inferVision2SeqPKVShape(meta.Dimensions, batchSize, seqLen, numHeads, headDim)
		pkvTensor, err := ort.NewEmptyTensor[float32](shape)
		if err != nil {
			for j := 0; j < i; j++ {
				outputs[j].Destroy()
			}
			return nil, nil, fmt.Errorf("creating pkv tensor %d (%s): %w", i, meta.Name, err)
		}
		outputs[i] = pkvTensor
	}

	// Run decoder
	if err := model.ORTModel.Session.Run(inputs, outputs); err != nil {
		for _, o := range outputs {
			o.Destroy()
		}
		return nil, nil, fmt.Errorf("running split decoder-init: %w", err)
	}

	// Extract logits
	logits := outputs[0].(*ort.Tensor[float32]).GetData()

	// Keep past_key_values
	pastKeyValues := outputs[1:]

	// Destroy logits tensor (we copied the data)
	outputs[0].Destroy()

	return logits, pastKeyValues, nil
}

// runVision2SeqDecoderStepSplitORT runs subsequent decoder steps with past_key_values.
// The with-past decoder takes input_ids and past_key_values (NO encoder_hidden_states).
// The encoder cross-attention is already computed and stored in past_key_values.encoder.
func runVision2SeqDecoderStepSplitORT(
	decoderInputIDs []int64,
	encoderHiddenStates ort.Value, // Not passed to model, but needed for shape info
	pastKeyValues []ort.Value,
	model *Model,
	batchSize, vocabSize, pastSeqLen, numHeads, headDim int,
) ([]float32, []ort.Value, error) {
	// Get encoder sequence length for encoder PKV outputs
	encoderSeqLen := 0
	if hs, ok := encoderHiddenStates.(*ort.Tensor[float32]); ok {
		shape := hs.GetShape()
		if len(shape) >= 2 {
			encoderSeqLen = int(shape[1])
		}
	}
	if encoderSeqLen == 0 {
		return nil, nil, errors.New("cannot determine encoder sequence length from hidden states")
	}

	// Create decoder input tensor
	inputTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), 1), decoderInputIDs)
	if err != nil {
		return nil, nil, fmt.Errorf("creating decoder input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Build inputs: input_ids, then past_key_values (NO encoder_hidden_states)
	inputs := make([]ort.Value, 1+len(pastKeyValues))
	inputs[0] = inputTensor
	for i, pkv := range pastKeyValues {
		inputs[1+i] = pkv
	}

	// Create output tensors
	// With-past decoder outputs: logits + present PKV (both decoder and encoder)
	numOutputs := len(model.OutputsMeta)
	outputs := make([]ort.Value, numOutputs)

	// Logits output
	logitsTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(batchSize), 1, int64(vocabSize)))
	if err != nil {
		return nil, nil, fmt.Errorf("creating logits tensor: %w", err)
	}
	outputs[0] = logitsTensor

	// Present key values outputs
	// - Decoder PKV: seq_len grows (pastSeqLen + 1)
	// - Encoder PKV: seq_len stays constant (encoderSeqLen)
	newDecoderSeqLen := pastSeqLen + 1
	for i := 1; i < numOutputs; i++ {
		meta := model.OutputsMeta[i]
		name := strings.ToLower(meta.Name)
		isEncoderPKV := strings.Contains(name, "encoder")

		var seqLen int
		if isEncoderPKV {
			seqLen = encoderSeqLen // Encoder PKV keeps encoder sequence length
		} else {
			seqLen = newDecoderSeqLen // Decoder PKV grows with each step
		}

		shape := inferVision2SeqPKVShape(meta.Dimensions, batchSize, seqLen, numHeads, headDim)
		pkvTensor, err := ort.NewEmptyTensor[float32](shape)
		if err != nil {
			for j := 0; j < i; j++ {
				outputs[j].Destroy()
			}
			return nil, nil, fmt.Errorf("creating present tensor %d: %w", i, err)
		}
		outputs[i] = pkvTensor
	}

	// Run decoder
	if err := model.ORTModel.Session.Run(inputs, outputs); err != nil {
		for _, o := range outputs {
			o.Destroy()
		}
		return nil, nil, fmt.Errorf("running split decoder-step: %w", err)
	}

	// Extract logits
	logits := outputs[0].(*ort.Tensor[float32]).GetData()

	// Separate decoder PKV from encoder PKV outputs
	// We only return decoder PKV; encoder PKV outputs are discarded
	// (caller reuses encoder PKV from init step)
	var decoderPKV []ort.Value
	for i := 1; i < numOutputs; i++ {
		meta := model.OutputsMeta[i]
		name := strings.ToLower(meta.Name)
		if strings.Contains(name, "encoder") {
			// Encoder PKV output - destroy it (we reuse from init)
			outputs[i].Destroy()
		} else {
			// Decoder PKV output - keep it
			decoderPKV = append(decoderPKV, outputs[i])
		}
	}

	// Destroy logits tensor (we copied the data)
	outputs[0].Destroy()

	return logits, decoderPKV, nil
}
