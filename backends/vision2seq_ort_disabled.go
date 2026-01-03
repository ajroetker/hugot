//go:build !ORT && !ALL

package backends

import "errors"

// RunVision2SeqEncoder is a stub when ORT is disabled.
func RunVision2SeqEncoder(batch Vision2SeqBatchInterface, model *Model, runtime string) error {
	return errors.New("ORT is not enabled - vision2seq requires ORT backend")
}

// RunVision2SeqGenerationGreedy is a stub when ORT is disabled.
func RunVision2SeqGenerationGreedy(batch Vision2SeqBatchInterface, pipeline Vision2SeqPipelineInterface) error {
	return errors.New("ORT is not enabled - vision2seq requires ORT backend")
}

// RunVision2SeqGenerationSampling is a stub when ORT is disabled.
func RunVision2SeqGenerationSampling(batch Vision2SeqBatchInterface, pipeline Vision2SeqPipelineInterface) error {
	return errors.New("ORT is not enabled - vision2seq requires ORT backend")
}
