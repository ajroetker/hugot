package pipelines

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVision2SeqOutput_GetOutput(t *testing.T) {
	output := &Vision2SeqOutput{
		GeneratedTexts: []string{"text1", "text2", "text3"},
	}

	result := output.GetOutput()

	require.Len(t, result, 3)
	assert.Equal(t, "text1", result[0])
	assert.Equal(t, "text2", result[1])
	assert.Equal(t, "text3", result[2])
}
