-- Applying a linear layer to a sequence --

local SequenceModule, parent = torch.class('onmt.SequenceModule','onmt.Network')

function SequenceModule:__init(module)
	
	-- assume that the input is a 3D sequence
	self.inputViewer = nn.View(1,1, -1):setNumInputDims(3)
	-- we reshape it to have the size (batch_size * seq_length) x hidden

	self.outputViewer = nn.View(1, -1):setNumInputDims(2)
	
	local myModule = nn.Sequential():add(self.inputViewer):add(module):add(self.outputViewer)
	
	parent.__init(self, myModule)
end

function SequenceModule:updateOutput(input)

	local batchSize = input:size(1)
	local seqLength = input:size(2)
	local inputDim = input:size(3)
	
	self.inputViewer:resetSize(batchSize * seqLength, -1)
	self.outputViewer:resetSize(batchSize, seqLength, -1)
	
	self.output = self.net:updateOutput(input)
	
	return self.output
	
	
end


function SequenceModule:setBatchSizeBeforeForward(bsz)
	self.modules[1].modules[2]:setBatchSize(bsz)
end


-- updateGradInput should be the same
