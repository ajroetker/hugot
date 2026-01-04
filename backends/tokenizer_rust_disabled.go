//go:build !ORT && !XLA && !ALL

package backends

import "errors"

type RustTokenizer struct{}

func loadRustTokenizer(_ []byte, _ *Model) error {
	return errors.New("rust Tokenizer is not enabled")
}

func tokenizeInputsRust(_ *PipelineBatch, _ *Tokenizer, _ []string) {}

func decodeRust(_ []uint32, _ *Tokenizer, _ bool) string {
	return ""
}

func allInputTokensRust(_ *BasePipeline) {}

// EncodePromptRust is a stub when Rust tokenizer is disabled.
func EncodePromptRust(_ *Tokenizer, _ string) ([]int64, error) {
	return nil, errors.New("Rust tokenizer is not enabled")
}
