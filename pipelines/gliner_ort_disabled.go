//go:build !ORT && !XLA && !ALL

package pipelines

import (
	"github.com/knights-analytics/hugot/backends"
)

// ORT stubs - not available when using pure Go runtime
func createGLiNERTensorsORT(batch *GLiNERBatch, model *backends.Model) error {
	return nil
}

func runGLiNERSessionOnBatchORT(batch *GLiNERBatch, p *backends.BasePipeline) error {
	return nil
}

// GoMLX stubs - not available when using pure Go runtime
func createGLiNERTensorsGoMLX(batch *GLiNERBatch, model *backends.Model) error {
	return nil
}

func runGLiNERSessionOnBatchGoMLX(batch *GLiNERBatch, p *backends.BasePipeline) error {
	return nil
}
