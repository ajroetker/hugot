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
	matcher := GetIONameMatcher()
	var outputTensor ort.Value
	for i, meta := range model.OutputsMeta {
		if matcher.IsHiddenStates(meta.Name) {
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

// RunFlorenceEncoder runs the Florence-2 multi-stage encoder on preprocessed images.
// Florence-2 uses: vision_encoder(pixel_values) -> image_features,
// then concatenates with prompt embeddings and runs through the text encoder.
func RunFlorenceEncoder(batch Vision2SeqBatchInterface, pipeline Vision2SeqPipelineInterface, runtime string) error {
	if runtime != "ORT" {
		return fmt.Errorf("unsupported runtime for Florence encoder: %s", runtime)
	}

	return runFlorenceEncoderORT(batch, pipeline)
}

func runFlorenceEncoderORT(batch Vision2SeqBatchInterface, pipeline Vision2SeqPipelineInterface) error {
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

	// Step 1: Run vision encoder
	visionModel := pipeline.GetVisionEncoderModel()
	if visionModel == nil || visionModel.ORTModel == nil {
		return errors.New("vision encoder model not loaded")
	}

	pixelValuesTensor, err := ort.NewTensor(
		ort.NewShape(int64(batchSize), int64(c), int64(h), int64(w)),
		flatData,
	)
	if err != nil {
		return fmt.Errorf("creating pixel_values tensor: %w", err)
	}
	defer pixelValuesTensor.Destroy()

	visionInputs := []ort.Value{pixelValuesTensor}
	visionOutputs := make([]ort.Value, len(visionModel.OutputsMeta))

	if err := visionModel.ORTModel.Session.Run(visionInputs, visionOutputs); err != nil {
		return fmt.Errorf("running vision encoder: %w", err)
	}

	// Get image_features output
	matcher := GetIONameMatcher()
	var imageFeatures ort.Value
	for i, meta := range visionModel.OutputsMeta {
		if matcher.IsImageFeatures(meta.Name) {
			imageFeatures = visionOutputs[i]
			break
		}
	}
	if imageFeatures == nil && len(visionOutputs) > 0 {
		imageFeatures = visionOutputs[0]
	}
	if imageFeatures == nil {
		return errors.New("no image features output from vision encoder")
	}
	defer func() {
		for _, o := range visionOutputs {
			if o != nil {
				o.Destroy()
			}
		}
	}()

	// Get image features shape and data
	imageFeaturesShape := imageFeatures.GetShape()
	if len(imageFeaturesShape) != 3 {
		return fmt.Errorf("unexpected image features shape: %v (expected 3D)", imageFeaturesShape)
	}
	imageSeqLen := imageFeaturesShape[1]
	hiddenSize := imageFeaturesShape[2]
	// Step 2: Get prompt tokens and run embed_tokens
	promptTokenIDs := batch.GetPromptTokenIDs()
	if len(promptTokenIDs) == 0 || len(promptTokenIDs[0]) == 0 {
		// No prompt tokens - use just image features as inputs_embeds
		// This shouldn't happen for Florence-2 but handle gracefully
		// Store image features as encoder hidden states
		batch.SetEncoderHiddenStates(imageFeatures)
		return nil
	}

	embedModel := pipeline.GetEmbedTokensModel()
	if embedModel == nil || embedModel.ORTModel == nil {
		return errors.New("embed_tokens model not loaded")
	}

	// Assume all prompts have the same length (padded)
	promptLen := int64(len(promptTokenIDs[0]))

	// Create input_ids tensor for embed_tokens
	flatPromptTokens := make([]int64, batchSize*int(promptLen))
	for i := 0; i < batchSize; i++ {
		for j := 0; j < int(promptLen); j++ {
			if j < len(promptTokenIDs[i]) {
				flatPromptTokens[i*int(promptLen)+j] = promptTokenIDs[i][j]
			}
		}
	}

	inputIdsTensor, err := ort.NewTensor(
		ort.NewShape(int64(batchSize), promptLen),
		flatPromptTokens,
	)
	if err != nil {
		return fmt.Errorf("creating input_ids tensor: %w", err)
	}
	defer inputIdsTensor.Destroy()

	embedInputs := []ort.Value{inputIdsTensor}
	embedOutputs := make([]ort.Value, len(embedModel.OutputsMeta))

	if err := embedModel.ORTModel.Session.Run(embedInputs, embedOutputs); err != nil {
		return fmt.Errorf("running embed_tokens: %w", err)
	}

	var promptEmbeds ort.Value
	for i, meta := range embedModel.OutputsMeta {
		if matcher.IsInputsEmbeds(meta.Name) {
			promptEmbeds = embedOutputs[i]
			break
		}
	}
	if promptEmbeds == nil && len(embedOutputs) > 0 {
		promptEmbeds = embedOutputs[0]
	}
	if promptEmbeds == nil {
		return errors.New("no embeddings output from embed_tokens")
	}
	defer func() {
		for _, o := range embedOutputs {
			if o != nil {
				o.Destroy()
			}
		}
	}()

	// Step 3: Concatenate image_features and prompt_embeds
	// Total sequence length = image_seq_len + prompt_len
	totalSeqLen := imageSeqLen + promptLen

	// Get raw data from both tensors using type assertion
	imageFeaturesTyped, ok := imageFeatures.(*ort.Tensor[float32])
	if !ok {
		return fmt.Errorf("image features is not a float32 tensor")
	}
	imageFeaturesFloat := imageFeaturesTyped.GetData()

	promptEmbedsTyped, ok := promptEmbeds.(*ort.Tensor[float32])
	if !ok {
		return fmt.Errorf("prompt embeds is not a float32 tensor")
	}
	promptEmbedsFloat := promptEmbedsTyped.GetData()

	// Concatenate: [image_features | prompt_embeds]
	inputsEmbeds := make([]float32, batchSize*int(totalSeqLen)*int(hiddenSize))
	for b := 0; b < batchSize; b++ {
		// Copy image features
		for s := int64(0); s < imageSeqLen; s++ {
			srcIdx := int64(b)*imageSeqLen*hiddenSize + s*hiddenSize
			dstIdx := int64(b)*totalSeqLen*hiddenSize + s*hiddenSize
			copy(inputsEmbeds[dstIdx:dstIdx+hiddenSize], imageFeaturesFloat[srcIdx:srcIdx+hiddenSize])
		}
		// Copy prompt embeds
		for s := int64(0); s < promptLen; s++ {
			srcIdx := int64(b)*promptLen*hiddenSize + s*hiddenSize
			dstIdx := int64(b)*totalSeqLen*hiddenSize + (imageSeqLen+s)*hiddenSize
			copy(inputsEmbeds[dstIdx:dstIdx+hiddenSize], promptEmbedsFloat[srcIdx:srcIdx+hiddenSize])
		}
	}

	inputsEmbedsTensor, err := ort.NewTensor(
		ort.NewShape(int64(batchSize), totalSeqLen, hiddenSize),
		inputsEmbeds,
	)
	if err != nil {
		return fmt.Errorf("creating inputs_embeds tensor: %w", err)
	}
	defer inputsEmbedsTensor.Destroy()

	// Step 4: Create attention mask (all 1s)
	attentionMask := make([]int64, batchSize*int(totalSeqLen))
	for i := range attentionMask {
		attentionMask[i] = 1
	}
	attentionMaskTensor, err := ort.NewTensor(
		ort.NewShape(int64(batchSize), totalSeqLen),
		attentionMask,
	)
	if err != nil {
		return fmt.Errorf("creating attention_mask tensor: %w", err)
	}
	defer attentionMaskTensor.Destroy()

	// Step 5: Run encoder model
	encoderModel := pipeline.GetEncoderModel()
	if encoderModel == nil || encoderModel.ORTModel == nil {
		return errors.New("encoder model not loaded")
	}

	// Florence-2 encoder expects: inputs_embeds, attention_mask (in that order based on florenceEncoderInputs)
	encoderInputs := []ort.Value{inputsEmbedsTensor, attentionMaskTensor}
	encoderOutputs := make([]ort.Value, len(encoderModel.OutputsMeta))

	if err := encoderModel.ORTModel.Session.Run(encoderInputs, encoderOutputs); err != nil {
		return fmt.Errorf("running encoder: %w", err)
	}

	// Find and store the encoder hidden states output
	var outputTensor ort.Value
	for i, meta := range encoderModel.OutputsMeta {
		if matcher.IsHiddenStates(meta.Name) {
			outputTensor = encoderOutputs[i]
			break
		}
	}
	if outputTensor == nil && len(encoderOutputs) > 0 {
		outputTensor = encoderOutputs[0]
	}
	if outputTensor == nil {
		return errors.New("no encoder output tensor produced")
	}

	// Store encoder outputs in batch
	batch.SetEncoderHiddenStates(outputTensor)

	// Set cleanup function
	batch.SetDestroyEncoder(func() error {
		var errs []error
		for _, o := range encoderOutputs {
			if o != nil && o != outputTensor {
				errs = append(errs, o.Destroy())
			}
		}
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

	// For Florence-2 merged decoder, encoder PKV is computed once and reused
	var florenceEncoderPKV []ort.Value

	// Prompt prefill phase: process prompt tokens if present
	// For DocVQA, the prompt is like: <s_docvqa><s_question>What is...?</s_question><s_answer>
	// NOTE: For Florence-2, prompt tokens are ONLY processed in the encoder (concatenated
	// with image features). The decoder should NOT receive prompt tokens - it starts fresh
	// from decoder_start_token and generates based on encoder hidden states.
	promptTokenIDs := batch.GetPromptTokenIDs()
	promptLength := 0
	isFlorence := pipeline.HasSeparateVisionEncoder()

	// Skip prompt prefill for Florence-2 - prompts are already in the encoder
	// For standard models (Donut, TrOCR), prompts are fed to the decoder
	if !isFlorence && len(promptTokenIDs) > 0 && len(promptTokenIDs[0]) > 0 {
		promptLength = len(promptTokenIDs[0])

		// Standard models: process prompt tokens one at a time
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
						pipeline,
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
						pipeline, // Pass pipeline for Florence-2 inputs_embeds
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
						pipeline,
					)

					for _, pkv := range pastKeyValues {
						pkv.Destroy()
					}
					pastKeyValues = decoderPKV
				} else {
					logits, newPKV, err = runVision2SeqDecoderStepORT(
						decoderInputIDs, encoderHiddenStates, pastKeyValues,
						mergedDecoderModel, batchSize, vocabSize, promptIdx, numHeads, headDim,
						pipeline, // Pass pipeline for Florence-2 inputs_embeds
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
					pipeline,
				)
				if err == nil {
					// Split init PKV into decoder and encoder
					// Layout: [dec0.key, dec0.val, enc0.key, enc0.val, ...]
					pastKeyValues = make([]ort.Value, 0, len(newPKV)/2)
					encoderPKV = make([]ort.Value, 0, len(newPKV)/2)
					for i := 0; i < len(newPKV); i += 4 {
						pastKeyValues = append(pastKeyValues, newPKV[i], newPKV[i+1]) // decoder
						encoderPKV = append(encoderPKV, newPKV[i+2], newPKV[i+3])     // encoder
					}
				}
			} else {
				logits, newPKV, err = runVision2SeqDecoderInitORT(
					decoderInputIDs, encoderHiddenStates,
					mergedDecoderModel, batchSize, vocabSize, numHeads, headDim,
					pipeline, // Pass pipeline for Florence-2 inputs_embeds
				)
				if isFlorence && err == nil {
					// Florence-2 merged decoder: separate decoder and encoder PKVs
					// Layout: layer0.decoder.key, layer0.decoder.value, layer0.encoder.key, layer0.encoder.value, ...
					pastKeyValues = make([]ort.Value, 0, len(newPKV)/2)
					florenceEncoderPKV = make([]ort.Value, 0, len(newPKV)/2)
					for i := 0; i < len(newPKV); i += 4 {
						pastKeyValues = append(pastKeyValues, newPKV[i], newPKV[i+1])
						florenceEncoderPKV = append(florenceEncoderPKV, newPKV[i+2], newPKV[i+3])
					}
				} else {
					pastKeyValues = newPKV
				}
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
					pipeline,
				)

				// Destroy old decoder PKV only (encoder PKV is reused)
				for _, pkv := range pastKeyValues {
					pkv.Destroy()
				}
				pastKeyValues = decoderPKV
			} else {
				if isFlorence {
					// Florence-2 merged decoder with cache
					// pastKeyValues contains only decoder PKVs, florenceEncoderPKV is passed separately
					var decoderPKV []ort.Value
					logits, decoderPKV, err = runFlorenceMergedDecoderStepORT(
						decoderInputIDs, encoderHiddenStates, pastKeyValues, florenceEncoderPKV,
						mergedDecoderModel, batchSize, vocabSize, effectiveStep, numHeads, headDim,
						pipeline,
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
						pipeline, // Pass pipeline for Florence-2 inputs_embeds
					)

					// Destroy old PKV
					for _, pkv := range pastKeyValues {
						pkv.Destroy()
					}
					pastKeyValues = newPKV
				}
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
			for _, pkv := range florenceEncoderPKV {
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
	capturedFlorenceEncoderPKV := florenceEncoderPKV
	batch.SetDestroyDecoder(func() error {
		var errs []error
		for _, pkv := range pastKeyValues {
			errs = append(errs, pkv.Destroy())
		}
		// Also destroy encoder PKV for split decoders
		for _, pkv := range capturedEncoderPKV {
			errs = append(errs, pkv.Destroy())
		}
		// Also destroy encoder PKV for Florence-2 merged decoder
		for _, pkv := range capturedFlorenceEncoderPKV {
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
	pipeline ...Vision2SeqPipelineInterface, // Optional pipeline for embed_tokens access
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
	inputs, inputDestroys, err := buildVision2SeqDecoderInputs(model, inputTensor, encoderHiddenStates, nil, batchSize, true, numHeads, headDim, pipeline...)
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
	pipeline ...Vision2SeqPipelineInterface, // Optional pipeline for embed_tokens access
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
	inputs, inputDestroys, err := buildVision2SeqDecoderInputs(model, inputTensor, encoderHiddenStates, pastKeyValues, batchSize, false, numHeads, headDim, pipeline...)
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
	pipeline ...Vision2SeqPipelineInterface, // Optional pipeline for embed_tokens access
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

	matcher := GetIONameMatcher()
	for i, meta := range model.InputsMeta {
		name := meta.Name

		switch {
		case matcher.IsInputIds(name):
			inputs[i] = inputIDsTensor

		case matcher.IsInputsEmbeds(name):
			// Florence-2 style: need to embed tokens first
			if len(pipeline) == 0 || pipeline[0] == nil {
				return nil, destroys, errors.New("inputs_embeds required but no pipeline provided for embed_tokens")
			}
			embedModel := pipeline[0].GetEmbedTokensModel()
			if embedModel == nil || embedModel.ORTModel == nil {
				return nil, destroys, errors.New("embed_tokens model not available for inputs_embeds")
			}

			// Run embed_tokens on decoder input IDs
			embedInputs := []ort.Value{inputIDsTensor}
			embedOutputs := make([]ort.Value, len(embedModel.OutputsMeta))

			if err := embedModel.ORTModel.Session.Run(embedInputs, embedOutputs); err != nil {
				return nil, destroys, fmt.Errorf("running embed_tokens for decoder: %w", err)
			}

			// Get embeddings output
			var inputsEmbeds ort.Value
			for j, m := range embedModel.OutputsMeta {
				if matcher.IsInputsEmbeds(m.Name) {
					inputsEmbeds = embedOutputs[j]
					break
				}
			}
			if inputsEmbeds == nil && len(embedOutputs) > 0 {
				inputsEmbeds = embedOutputs[0]
			}
			if inputsEmbeds == nil {
				return nil, destroys, errors.New("no embeddings output from embed_tokens")
			}

			inputs[i] = inputsEmbeds
			// Clean up other outputs but keep inputsEmbeds
			for j, o := range embedOutputs {
				if o != nil && o != inputsEmbeds {
					embedOutputs[j].Destroy()
				}
			}
			destroys = append(destroys, func() { inputsEmbeds.Destroy() })

		case matcher.IsEncoderHiddenStates(name):
			inputs[i] = encoderHiddenStates

		case matcher.IsEncoderAttentionMask(name):
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

		case matcher.IsUseCacheBranch(name):
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

		case matcher.IsPastKeyValues(name):
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
// The init decoder takes inputs based on the model's expected inputs (which vary by architecture).
// Standard models: input_ids, encoder_hidden_states
// Florence-2: encoder_attention_mask, encoder_hidden_states, inputs_embeds
func runVision2SeqDecoderInitSplitORT(
	decoderInputIDs []int64,
	encoderHiddenStates ort.Value,
	model *Model,
	batchSize, vocabSize, numHeads, headDim int,
	pipeline ...Vision2SeqPipelineInterface, // Optional pipeline for embed_tokens access
) ([]float32, []ort.Value, error) {
	// Get encoder sequence length from encoder hidden states
	encoderSeqLen := 0
	hiddenSize := 0
	if hs, ok := encoderHiddenStates.(*ort.Tensor[float32]); ok {
		shape := hs.GetShape()
		if len(shape) >= 2 {
			encoderSeqLen = int(shape[1])
		}
		if len(shape) >= 3 {
			hiddenSize = int(shape[2])
		}
	}
	if encoderSeqLen == 0 {
		return nil, nil, errors.New("cannot determine encoder sequence length from hidden states")
	}

	// Build inputs dynamically based on model's expected inputs
	inputs := make([]ort.Value, len(model.InputsMeta))
	var destroys []func()
	defer func() {
		for _, d := range destroys {
			d()
		}
	}()

	matcher := GetIONameMatcher()
	for i, meta := range model.InputsMeta {
		name := meta.Name

		switch {
		case matcher.IsInputIds(name):
			// Standard decoder uses input_ids
			inputTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), 1), decoderInputIDs)
			if err != nil {
				return nil, nil, fmt.Errorf("creating input_ids tensor: %w", err)
			}
			inputs[i] = inputTensor
			destroys = append(destroys, func() { inputTensor.Destroy() })

		case matcher.IsInputsEmbeds(name):
			// Florence-2 style: need to embed tokens first
			if len(pipeline) == 0 || pipeline[0] == nil {
				return nil, nil, errors.New("inputs_embeds required but no pipeline provided for embed_tokens")
			}
			embedModel := pipeline[0].GetEmbedTokensModel()
			if embedModel == nil || embedModel.ORTModel == nil {
				return nil, nil, errors.New("embed_tokens model not available for inputs_embeds")
			}

			// Run embed_tokens on decoder input IDs
			inputIdsTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), 1), decoderInputIDs)
			if err != nil {
				return nil, nil, fmt.Errorf("creating input_ids for embedding: %w", err)
			}
			defer inputIdsTensor.Destroy()

			embedInputs := []ort.Value{inputIdsTensor}
			embedOutputs := make([]ort.Value, len(embedModel.OutputsMeta))

			if err := embedModel.ORTModel.Session.Run(embedInputs, embedOutputs); err != nil {
				return nil, nil, fmt.Errorf("running embed_tokens for decoder: %w", err)
			}

			// Get embeddings output
			var inputsEmbeds ort.Value
			for j, m := range embedModel.OutputsMeta {
				if matcher.IsInputsEmbeds(m.Name) {
					inputsEmbeds = embedOutputs[j]
					break
				}
			}
			if inputsEmbeds == nil && len(embedOutputs) > 0 {
				inputsEmbeds = embedOutputs[0]
			}
			if inputsEmbeds == nil {
				return nil, nil, errors.New("no embeddings output from embed_tokens")
			}

			inputs[i] = inputsEmbeds
			// Clean up other outputs but keep inputsEmbeds
			for j, o := range embedOutputs {
				if o != nil && o != inputsEmbeds {
					embedOutputs[j].Destroy()
				}
			}
			destroys = append(destroys, func() { inputsEmbeds.Destroy() })

		case matcher.IsEncoderHiddenStates(name):
			inputs[i] = encoderHiddenStates

		case matcher.IsEncoderAttentionMask(name):
			// Create all-ones mask for encoder
			maskData := make([]int64, batchSize*encoderSeqLen)
			for j := range maskData {
				maskData[j] = 1
			}
			maskTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), int64(encoderSeqLen)), maskData)
			if err != nil {
				return nil, nil, fmt.Errorf("creating encoder attention mask: %w", err)
			}
			inputs[i] = maskTensor
			destroys = append(destroys, func() { maskTensor.Destroy() })

		default:
			return nil, nil, fmt.Errorf("unknown init decoder input: %s", meta.Name)
		}
	}

	_ = hiddenSize // May be used for validation in future

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
// The with-past decoder takes inputs based on model's expected inputs.
// Standard: input_ids, past_key_values
// Florence-2: encoder_attention_mask, past_key_values, inputs_embeds
func runVision2SeqDecoderStepSplitORT(
	decoderInputIDs []int64,
	encoderHiddenStates ort.Value, // Not passed to model, but needed for shape info
	pastKeyValues []ort.Value,
	model *Model,
	batchSize, vocabSize, pastSeqLen, numHeads, headDim int,
	pipeline ...Vision2SeqPipelineInterface, // Optional pipeline for embed_tokens access
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

	// Build inputs dynamically based on model's expected inputs
	inputs := make([]ort.Value, len(model.InputsMeta))
	var destroys []func()
	defer func() {
		for _, d := range destroys {
			d()
		}
	}()

	matcher := GetIONameMatcher()
	pkvIndex := 0
	for i, meta := range model.InputsMeta {
		name := meta.Name

		switch {
		case matcher.IsInputIds(name):
			// Standard decoder uses input_ids
			inputTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), 1), decoderInputIDs)
			if err != nil {
				return nil, nil, fmt.Errorf("creating input_ids tensor: %w", err)
			}
			inputs[i] = inputTensor
			destroys = append(destroys, func() { inputTensor.Destroy() })

		case matcher.IsInputsEmbeds(name):
			// Florence-2 style: need to embed tokens first
			if len(pipeline) == 0 || pipeline[0] == nil {
				return nil, nil, errors.New("inputs_embeds required but no pipeline provided for embed_tokens")
			}
			embedModel := pipeline[0].GetEmbedTokensModel()
			if embedModel == nil || embedModel.ORTModel == nil {
				return nil, nil, errors.New("embed_tokens model not available for inputs_embeds")
			}

			// Run embed_tokens on decoder input IDs
			inputIdsTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), 1), decoderInputIDs)
			if err != nil {
				return nil, nil, fmt.Errorf("creating input_ids for embedding: %w", err)
			}
			defer inputIdsTensor.Destroy()

			embedInputs := []ort.Value{inputIdsTensor}
			embedOutputs := make([]ort.Value, len(embedModel.OutputsMeta))

			if err := embedModel.ORTModel.Session.Run(embedInputs, embedOutputs); err != nil {
				return nil, nil, fmt.Errorf("running embed_tokens for decoder: %w", err)
			}

			// Get embeddings output
			var inputsEmbeds ort.Value
			for j, m := range embedModel.OutputsMeta {
				if matcher.IsInputsEmbeds(m.Name) {
					inputsEmbeds = embedOutputs[j]
					break
				}
			}
			if inputsEmbeds == nil && len(embedOutputs) > 0 {
				inputsEmbeds = embedOutputs[0]
			}
			if inputsEmbeds == nil {
				return nil, nil, errors.New("no embeddings output from embed_tokens")
			}

			inputs[i] = inputsEmbeds
			// Clean up other outputs but keep inputsEmbeds
			for j, o := range embedOutputs {
				if o != nil && o != inputsEmbeds {
					embedOutputs[j].Destroy()
				}
			}
			destroys = append(destroys, func() { inputsEmbeds.Destroy() })

		case matcher.IsEncoderAttentionMask(name):
			// Create all-ones mask for encoder
			maskData := make([]int64, batchSize*encoderSeqLen)
			for j := range maskData {
				maskData[j] = 1
			}
			maskTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), int64(encoderSeqLen)), maskData)
			if err != nil {
				return nil, nil, fmt.Errorf("creating encoder attention mask: %w", err)
			}
			inputs[i] = maskTensor
			destroys = append(destroys, func() { maskTensor.Destroy() })

		case matcher.IsPastKeyValues(name):
			if pkvIndex < len(pastKeyValues) {
				inputs[i] = pastKeyValues[pkvIndex]
				pkvIndex++
			} else {
				return nil, nil, fmt.Errorf("not enough past_key_values provided for input %s", meta.Name)
			}

		default:
			return nil, nil, fmt.Errorf("unknown step decoder input: %s", meta.Name)
		}
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

// runFlorenceDecoderInitWithPromptORT runs the Florence-2 init decoder with a full prompt sequence.
// Unlike standard decoders which process tokens one at a time, Florence-2's decoder_with_past
// expects the same sequence length as the init decoder, so we process all prompt tokens at once.
func runFlorenceDecoderInitWithPromptORT(
	promptTokenIDs []int64, // Flat array: [batch * promptLength]
	promptLength int,
	encoderHiddenStates ort.Value,
	model *Model,
	batchSize, vocabSize, numHeads, headDim int,
	pipeline Vision2SeqPipelineInterface,
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

	// Build inputs dynamically based on model's expected inputs
	inputs := make([]ort.Value, len(model.InputsMeta))
	var destroys []func()
	defer func() {
		for _, d := range destroys {
			d()
		}
	}()

	matcher := GetIONameMatcher()
	for i, meta := range model.InputsMeta {
		name := meta.Name

		switch {
		case matcher.IsInputsEmbeds(name):
			// Florence-2: need to embed all prompt tokens
			embedModel := pipeline.GetEmbedTokensModel()
			if embedModel == nil || embedModel.ORTModel == nil {
				return nil, nil, errors.New("embed_tokens model not available for inputs_embeds")
			}

			// Run embed_tokens on all prompt tokens
			inputIdsTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), int64(promptLength)), promptTokenIDs)
			if err != nil {
				return nil, nil, fmt.Errorf("creating input_ids for embedding: %w", err)
			}
			defer inputIdsTensor.Destroy()

			embedInputs := []ort.Value{inputIdsTensor}
			embedOutputs := make([]ort.Value, len(embedModel.OutputsMeta))

			if err := embedModel.ORTModel.Session.Run(embedInputs, embedOutputs); err != nil {
				return nil, nil, fmt.Errorf("running embed_tokens for decoder: %w", err)
			}

			// Get embeddings output
			var inputsEmbeds ort.Value
			for j, m := range embedModel.OutputsMeta {
				if matcher.IsInputsEmbeds(m.Name) {
					inputsEmbeds = embedOutputs[j]
					break
				}
			}
			if inputsEmbeds == nil && len(embedOutputs) > 0 {
				inputsEmbeds = embedOutputs[0]
			}
			if inputsEmbeds == nil {
				return nil, nil, errors.New("no embeddings output from embed_tokens")
			}

			inputs[i] = inputsEmbeds
			// Clean up other outputs but keep inputsEmbeds
			for j, o := range embedOutputs {
				if o != nil && o != inputsEmbeds {
					embedOutputs[j].Destroy()
				}
			}
			destroys = append(destroys, func() { inputsEmbeds.Destroy() })

		case matcher.IsEncoderHiddenStates(name):
			inputs[i] = encoderHiddenStates

		case matcher.IsEncoderAttentionMask(name):
			// Create all-ones mask for encoder
			maskData := make([]int64, batchSize*encoderSeqLen)
			for j := range maskData {
				maskData[j] = 1
			}
			maskTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), int64(encoderSeqLen)), maskData)
			if err != nil {
				return nil, nil, fmt.Errorf("creating encoder attention mask: %w", err)
			}
			inputs[i] = maskTensor
			destroys = append(destroys, func() { maskTensor.Destroy() })

		default:
			return nil, nil, fmt.Errorf("unknown florence init decoder input: %s", meta.Name)
		}
	}

	// Create output tensors
	numOutputs := len(model.OutputsMeta)
	outputs := make([]ort.Value, numOutputs)

	// Logits output (first) - shape: [batch, promptLength, vocabSize]
	logitsTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(batchSize), int64(promptLength), int64(vocabSize)))
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
			seqLen = promptLength // decoder self-attention for full prompt
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
		return nil, nil, fmt.Errorf("running florence init decoder with prompt: %w", err)
	}

	// Extract logits
	logits := outputs[0].(*ort.Tensor[float32]).GetData()

	// Keep past_key_values
	pastKeyValues := outputs[1:]

	// Destroy logits tensor (we copied the data)
	outputs[0].Destroy()

	return logits, pastKeyValues, nil
}

// runFlorenceMergedDecoderWithPromptORT runs the Florence-2 merged decoder with a full prompt sequence.
// The merged decoder has an optimum::if node that branches based on use_cache_branch.
// For prompt processing, we set use_cache_branch=False and provide empty PKV tensors.
func runFlorenceMergedDecoderWithPromptORT(
	promptTokenIDs []int64, // Flat array: [batch * promptLength]
	promptLength int,
	encoderHiddenStates ort.Value,
	model *Model,
	batchSize, vocabSize, numHeads, headDim int,
	pipeline Vision2SeqPipelineInterface,
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

	// Build inputs dynamically based on model's expected inputs
	inputs := make([]ort.Value, len(model.InputsMeta))
	var destroys []func()
	defer func() {
		for _, d := range destroys {
			d()
		}
	}()

	matcher := GetIONameMatcher()
	for i, meta := range model.InputsMeta {
		name := meta.Name

		switch {
		case matcher.IsInputsEmbeds(name):
			// Florence-2: need to embed all prompt tokens
			embedModel := pipeline.GetEmbedTokensModel()
			if embedModel == nil || embedModel.ORTModel == nil {
				return nil, nil, errors.New("embed_tokens model not available for inputs_embeds")
			}

			// Run embed_tokens on all prompt tokens
			inputIdsTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), int64(promptLength)), promptTokenIDs)
			if err != nil {
				return nil, nil, fmt.Errorf("creating input_ids for embedding: %w", err)
			}
			defer inputIdsTensor.Destroy()

			embedInputs := []ort.Value{inputIdsTensor}
			embedOutputs := make([]ort.Value, len(embedModel.OutputsMeta))

			if err := embedModel.ORTModel.Session.Run(embedInputs, embedOutputs); err != nil {
				return nil, nil, fmt.Errorf("running embed_tokens for decoder: %w", err)
			}

			// Get embeddings output
			var inputsEmbeds ort.Value
			for j, m := range embedModel.OutputsMeta {
				if matcher.IsInputsEmbeds(m.Name) {
					inputsEmbeds = embedOutputs[j]
					break
				}
			}
			if inputsEmbeds == nil && len(embedOutputs) > 0 {
				inputsEmbeds = embedOutputs[0]
			}
			if inputsEmbeds == nil {
				return nil, nil, errors.New("no embeddings output from embed_tokens")
			}

			inputs[i] = inputsEmbeds
			// Clean up other outputs but keep inputsEmbeds
			for j, o := range embedOutputs {
				if o != nil && o != inputsEmbeds {
					embedOutputs[j].Destroy()
				}
			}
			destroys = append(destroys, func() { inputsEmbeds.Destroy() })

		case matcher.IsEncoderHiddenStates(name):
			inputs[i] = encoderHiddenStates

		case matcher.IsEncoderAttentionMask(name):
			// Create all-ones mask for encoder
			maskData := make([]int64, batchSize*encoderSeqLen)
			for j := range maskData {
				maskData[j] = 1
			}
			maskTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), int64(encoderSeqLen)), maskData)
			if err != nil {
				return nil, nil, fmt.Errorf("creating encoder attention mask: %w", err)
			}
			inputs[i] = maskTensor
			destroys = append(destroys, func() { maskTensor.Destroy() })

		case matcher.IsUseCacheBranch(name):
			// For init/prompt processing, use_cache_branch = False
			useCacheData := make([]bool, 1)
			useCacheData[0] = false
			useCacheTensor, err := ort.NewTensor(ort.NewShape(1), useCacheData)
			if err != nil {
				return nil, nil, fmt.Errorf("creating use_cache_branch: %w", err)
			}
			inputs[i] = useCacheTensor
			destroys = append(destroys, func() { useCacheTensor.Destroy() })

		case matcher.IsPastKeyValues(name):
			// For init step, create empty PKV tensors
			isEncoderPKV := strings.Contains(name, "encoder")
			var seqLen int
			if isEncoderPKV {
				seqLen = encoderSeqLen
			} else {
				seqLen = 0 // decoder self-attention starts at 0 for init
			}
			shape := inferVision2SeqPKVShape(meta.Dimensions, batchSize, seqLen, numHeads, headDim)
			emptyTensor, err := ort.NewEmptyTensor[float32](shape)
			if err != nil {
				return nil, nil, fmt.Errorf("creating empty pkv tensor: %w", err)
			}
			inputs[i] = emptyTensor
			destroys = append(destroys, func() { emptyTensor.Destroy() })

		default:
			return nil, nil, fmt.Errorf("unknown florence merged decoder input: %s", meta.Name)
		}
	}

	// Create output tensors - let ORT allocate dynamically
	numOutputs := len(model.OutputsMeta)
	outputs := make([]ort.Value, numOutputs)

	// Run decoder
	if err := model.ORTModel.Session.Run(inputs, outputs); err != nil {
		for _, o := range outputs {
			if o != nil {
				o.Destroy()
			}
		}
		return nil, nil, fmt.Errorf("running florence merged decoder with prompt: %w", err)
	}

	// Verify logits output
	if outputs[0] == nil {
		for _, o := range outputs[1:] {
			if o != nil {
				o.Destroy()
			}
		}
		return nil, nil, errors.New("logits output not allocated by ORT")
	}

	// Extract logits
	logits := outputs[0].(*ort.Tensor[float32]).GetData()

	// Keep past_key_values
	pastKeyValues := outputs[1:]

	// Destroy logits tensor (we copied the data)
	outputs[0].Destroy()

	return logits, pastKeyValues, nil
}

// runFlorenceMergedDecoderStepORT runs subsequent generation steps with Florence-2 merged decoder.
// Sets use_cache_branch=True and passes existing PKV for incremental generation.
// decoderPKV contains only decoder PKVs (2 per layer: key, value)
// encoderPKV contains only encoder PKVs (2 per layer: key, value) - these are reused, not destroyed
// Returns only decoder PKVs (encoder PKVs don't change during decoding)
func runFlorenceMergedDecoderStepORT(
	decoderInputIDs []int64, // Single token per batch item
	encoderHiddenStates ort.Value,
	decoderPKV []ort.Value, // Decoder PKVs only (12 tensors for 6 layers)
	encoderPKV []ort.Value, // Encoder PKVs only (12 tensors for 6 layers) - reused, not destroyed
	model *Model,
	batchSize, vocabSize, pastSeqLen, numHeads, headDim int,
	pipeline Vision2SeqPipelineInterface,
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

	// Interleave decoder and encoder PKVs for model input
	// Model expects: layer0.decoder.key, layer0.decoder.value, layer0.encoder.key, layer0.encoder.value, ...
	numLayers := len(decoderPKV) / 2
	fullPKV := make([]ort.Value, 0, len(decoderPKV)+len(encoderPKV))
	for layer := 0; layer < numLayers; layer++ {
		// Decoder key, value
		fullPKV = append(fullPKV, decoderPKV[layer*2], decoderPKV[layer*2+1])
		// Encoder key, value
		fullPKV = append(fullPKV, encoderPKV[layer*2], encoderPKV[layer*2+1])
	}

	// Build inputs dynamically based on model's expected inputs
	inputs := make([]ort.Value, len(model.InputsMeta))
	var destroys []func()
	defer func() {
		for _, d := range destroys {
			d()
		}
	}()

	matcher := GetIONameMatcher()
	pkvIndex := 0
	for i, meta := range model.InputsMeta {
		name := meta.Name

		switch {
		case matcher.IsInputsEmbeds(name):
			// Florence-2: need to embed single token
			embedModel := pipeline.GetEmbedTokensModel()
			if embedModel == nil || embedModel.ORTModel == nil {
				return nil, nil, errors.New("embed_tokens model not available for inputs_embeds")
			}

			// Run embed_tokens on decoder input IDs (single token)
			inputIdsTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), 1), decoderInputIDs)
			if err != nil {
				return nil, nil, fmt.Errorf("creating input_ids for embedding: %w", err)
			}
			defer inputIdsTensor.Destroy()

			embedInputs := []ort.Value{inputIdsTensor}
			embedOutputs := make([]ort.Value, len(embedModel.OutputsMeta))

			if err := embedModel.ORTModel.Session.Run(embedInputs, embedOutputs); err != nil {
				return nil, nil, fmt.Errorf("running embed_tokens for decoder: %w", err)
			}

			// Get embeddings output
			var inputsEmbeds ort.Value
			for j, m := range embedModel.OutputsMeta {
				if matcher.IsInputsEmbeds(m.Name) {
					inputsEmbeds = embedOutputs[j]
					break
				}
			}
			if inputsEmbeds == nil && len(embedOutputs) > 0 {
				inputsEmbeds = embedOutputs[0]
			}
			if inputsEmbeds == nil {
				return nil, nil, errors.New("no embeddings output from embed_tokens")
			}

			inputs[i] = inputsEmbeds
			// Clean up other outputs but keep inputsEmbeds
			for j, o := range embedOutputs {
				if o != nil && o != inputsEmbeds {
					embedOutputs[j].Destroy()
				}
			}
			destroys = append(destroys, func() { inputsEmbeds.Destroy() })

		case matcher.IsEncoderHiddenStates(name):
			inputs[i] = encoderHiddenStates

		case matcher.IsEncoderAttentionMask(name):
			// Create all-ones mask for encoder
			maskData := make([]int64, batchSize*encoderSeqLen)
			for j := range maskData {
				maskData[j] = 1
			}
			maskTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), int64(encoderSeqLen)), maskData)
			if err != nil {
				return nil, nil, fmt.Errorf("creating encoder attention mask: %w", err)
			}
			inputs[i] = maskTensor
			destroys = append(destroys, func() { maskTensor.Destroy() })

		case matcher.IsUseCacheBranch(name):
			// For subsequent steps, use_cache_branch = True
			useCacheData := make([]bool, 1)
			useCacheData[0] = true
			useCacheTensor, err := ort.NewTensor(ort.NewShape(1), useCacheData)
			if err != nil {
				return nil, nil, fmt.Errorf("creating use_cache_branch: %w", err)
			}
			inputs[i] = useCacheTensor
			destroys = append(destroys, func() { useCacheTensor.Destroy() })

		case matcher.IsPastKeyValues(name):
			// Use interleaved PKV (decoder + encoder)
			if pkvIndex < len(fullPKV) {
				inputs[i] = fullPKV[pkvIndex]
				pkvIndex++
			} else {
				return nil, nil, fmt.Errorf("not enough past_key_values provided for input %s (have %d, need %d)", meta.Name, len(fullPKV), pkvIndex+1)
			}

		default:
			return nil, nil, fmt.Errorf("unknown florence merged decoder step input: %s", meta.Name)
		}
	}

	// Create output tensors - let ORT allocate dynamically for the If node
	numOutputs := len(model.OutputsMeta)
	outputs := make([]ort.Value, numOutputs)

	// Run decoder
	if err := model.ORTModel.Session.Run(inputs, outputs); err != nil {
		for _, o := range outputs {
			if o != nil {
				o.Destroy()
			}
		}
		return nil, nil, fmt.Errorf("running florence merged decoder step: %w", err)
	}

	// Extract logits from first output
	logitsTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		for _, o := range outputs {
			if o != nil {
				o.Destroy()
			}
		}
		return nil, nil, errors.New("logits output is not a float32 tensor")
	}
	logits := logitsTensor.GetData()

	// Extract only decoder PKVs from outputs
	// Output layout per layer: decoder.key, decoder.value, encoder.key, encoder.value
	// We only keep decoder PKVs (indices 0,1 within each group of 4)
	// Encoder PKV outputs are empty placeholders when use_cache_branch=True
	allPKV := outputs[1:]
	newDecoderPKV := make([]ort.Value, 0, numLayers*2)

	for layer := 0; layer < numLayers; layer++ {
		base := layer * 4
		// Keep decoder key and value
		newDecoderPKV = append(newDecoderPKV, allPKV[base], allPKV[base+1])
		// Destroy encoder key and value outputs (they're placeholders)
		if base+2 < len(allPKV) && allPKV[base+2] != nil {
			allPKV[base+2].Destroy()
		}
		if base+3 < len(allPKV) && allPKV[base+3] != nil {
			allPKV[base+3].Destroy()
		}
	}

	// Destroy logits tensor (we copied the data)
	outputs[0].Destroy()

	return logits, newDecoderPKV, nil
}
