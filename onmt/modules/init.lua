onmt = onmt or {}

require('onmt.modules.Sequencer')
require('onmt.modules.Encoder')
require('onmt.modules.BiEncoder')
require('onmt.modules.Decoder')

require('onmt.modules.Network')

require('onmt.modules.GRU')
require('onmt.modules.GRUNode')
require('onmt.modules.LSTM')
require('onmt.modules.LSTMCell')

require('onmt.modules.MaskedSoftmax')
require('onmt.modules.WordEmbedding')
require('onmt.modules.FeaturesEmbedding')
require('onmt.modules.VDropout')

-- Attention modules
require('onmt.modules.GlobalAttention')
require('onmt.modules.ContextGateAttention')


require('onmt.modules.Generator')
require('onmt.modules.FeaturesGenerator')

require('onmt.modules.ParallelClassNLLCriterion')


-- Coverage modules
require('onmt.modules.Coverage.ContextCoverage')
require('onmt.modules.Coverage.CoverageAttention')

-- Other utility modules
require('onmt.modules.Utils.SequenceLinear')
require('onmt.modules.Utils.SequenceModule')
require('onmt.modules.Utils.Replicator')
require('onmt.modules.Utils.LayerNormalization')
require('onmt.modules.Utils.ExpandAs')
require('onmt.modules.Utils.JoinReplicateTable')
require('onmt.modules.Utils.Reverse')
require('onmt.modules.Utils.ZipTable')
require('onmt.modules.Utils.NilModule')
require('onmt.modules.Utils.CompactBilinearPooling')

return onmt
