-- Applying a linear layer to a sequence --

local SequenceModule, parent = torch.class('onmt.SequenceModule','onmt.Network')

function SequenceModule:__init(module, nInputs, nOutputs)

	nInputs = nInputs or 1
	nOutputs = nOutputs or 1
	
	local inParaTable 
	
	if nInputs > 1 then
		inParaTable = nn.ParallelTable()
	end
	
	self.inputViewer = {}
	
	for i = 1, nInputs do
		self.inputViewer[i] = nn.View(1,1, -1):setNumInputDims(3)
		
		if inParaTable then
			local seq = nn.Sequential():add(nn.Contiguous())
			inParaTable:add(seq:add(self.inputViewer[i]))
		else
			inParaTable = self.inputViewer[i]
		end
	end
	
	self.outputViewer = {}
	
	local outParaTable 
	
	if nOutputs > 1 then
		outParaTable = nn.ParallelTable()
	end
	
	for i = 1, nOutputs do
		self.outputViewer[i] = nn.View(1, -1):setNumInputDims(2)
		
		if outParaTable then
			local seq = nn.Sequential():add(nn.Contiguous())
			outParaTable:add(seq:add(self.outputViewer[i]))
		else
			outParaTable = self.outputViewer[i]
		end
	end
	
	-- assume that the input is a 3D sequence
	--~ self.inputViewer = nn.View(1,1, -1):setNumInputDims(3)
	-- we reshape it to have the size (batch_size * seq_length) x hidden

	--~ self.outputViewer = nn.View(1, -1):setNumInputDims(2)
	
	
	
	
	
	local myModule = nn.Sequential()
		
	myModule:add(inParaTable):add(module):add(outParaTable)
	
	parent.__init(self, myModule)
end

function SequenceModule:updateOutput(input)

	-- input can be a table
	local batchSize, seqlength
	if torch.type(input) == 'table' then
		batchSize = input[1]:size(1)
		seqLength = input[1]:size(2)
	else
		batchSize = input:size(1)
		seqLength = input:size(2)
	end
	
	
	for i = 1, #self.inputViewer do
		self.inputViewer[i]:resetSize(batchSize * seqLength, -1)
	end
			
	for i = 1, #self.outputViewer do
		self.outputViewer[i]:resetSize(batchSize, seqLength, -1)
	end
	--~ self.inputViewer:resetSize(batchSize * seqLength, -1)
	--~ self.outputViewer:resetSize(batchSize, seqLength, -1)
	
	self.output = self.net:updateOutput(input)
	
	return self.output
	
	
end


function SequenceModule:setBatchSizeBeforeForward(bsz)
	if self.modules[1].modules[2].setBatchSize then
		self.modules[1].modules[2]:setBatchSize(bsz)
	end
end


-- updateGradInput should be the same
