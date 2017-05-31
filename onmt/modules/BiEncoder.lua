local function reverseInput(batch)
  batch.sourceInput, batch.sourceInputRev = batch.sourceInputRev, batch.sourceInput
  batch.sourceInputFeatures, batch.sourceInputRevFeatures = batch.sourceInputRevFeatures, batch.sourceInputFeatures
  batch.sourceInputPadLeft, batch.sourceInputRevPadLeft = batch.sourceInputRevPadLeft, batch.sourceInputPadLeft
end

--[[ BiEncoder is a bidirectional Sequencer used for the source language.


 `netFwd`

    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

 `netBwd`

    h_1 <= h_2 <= h_3 <= ... <= h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 <= h_2 <= h_3 <= ... <= h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

Inherits from [onmt.Sequencer](onmt+modules+Sequencer).

--]]
local BiEncoder, parent = torch.class('onmt.BiEncoder', 'nn.Container')

--[[ Create a bi-encoder.

Parameters:

  * `input` - input neural network.
  * `rnn` - recurrent template module.
  * `merge` - fwd/bwd merge operation {"concat", "sum"}
]]
function BiEncoder:__init(input, rnn, merge, bridge, nDecLayers)
  parent.__init(self)

  self.fwd = onmt.Encoder.new(input, rnn)
  self.bwd = onmt.Encoder.new(input:clone('weight', 'bias', 'gradWeight', 'gradBias'), rnn:clone())
  self.wordEmb = input:clone('weight', 'bias', 'gradWeight', 'gradBias')

  self.args = {}
  self.args.merge = merge

  self.args.rnnSize = rnn.outputSize
  self.args.numEffectiveLayers = rnn.numEffectiveLayers
  self.args.layers = rnn.layers
  self.args.bridge = bridge
  
  self.args.nDecLayers = nDecLayers or self.args.numEffectiveLayers 

  if self.args.merge == 'concat' then
    self.args.hiddenSize = self.args.rnnSize * 2
  else
    self.args.hiddenSize = self.args.rnnSize
  end

  self:add(self.fwd)
  self:add(self.bwd)
  self:add(self.wordEmb)
  
  self.contextMerger = self:_buildContextMerger()
  self.stateMerger = self:_buildStateMerger()
  self.bridge = self:_buildBridge()

	self:add(self.contextMerger)
	self:add(self.stateMerger)
	self:add(self.bridge)
  self:resetPreallocation()
end

--[[ Return a new BiEncoder using the serialized data `pretrained`. ]]
function BiEncoder.load(pretrained)
  local self = torch.factory('onmt.BiEncoder')()

  parent.__init(self)
  
  -- For backward compatibility
  if #pretrained.modules == 6 and torch.typename(pretrained.modules[5]) ~= torch.typename(pretrained.modules[6]) then
		self.fwd = onmt.Encoder.load(pretrained.modules[1])
		self.bwd = onmt.Encoder.load(pretrained.modules[2])
		self.wordEmb = pretrained.modules[3]
		self.contextMerger = pretrained.modules[4]
		self.stateMerger = pretrained.modules[5]
		self.bridge      = pretrained.modules[6]
  else
		self.fwd = onmt.Encoder.load(pretrained.modules[1])
		self.bwd = onmt.Encoder.load(pretrained.modules[2])
		self.contextMerger = pretrained.modules[3]
		self.stateMerger = pretrained.modules[4]
		self.bridge      = pretrained.modules[5]
  end
	

  self.args = pretrained.args
  
  
  
  -- backward compatibility with old models
  self.args.nDecLayers = self.args.nDecLayers or self.args.numEffectiveLayers 
  
  if self.contextMerger == nil then
		self.contextMerger = self:_buildContextMerger()
  end
  
  if self.stateMerger == nil then
		self.stateMerger = self:_buildStateMerger()
  end
  
  if self.bridge == nil then
		self.args.bridge = 'copy'
		self.bridge = self:_buildBridge()
  end
  
  self:add(self.fwd)
  self:add(self.bwd)
  
  if self.wordEmb then
		self:add(self.wordEmb)
  end
  
  self:add(self.contextMerger)
  self:add(self.stateMerger)
  self:add(self.bridge)

  self:resetPreallocation()

  return self
end

--[[ Return data to serialize. ]]
function BiEncoder:serialize()
  local modulesData = {}
  for i = 1, #self.modules do
		if self.modules[i].serialize then
			table.insert(modulesData, self.modules[i]:serialize())
		else
			table.insert(modulesData, self.modules[i])
		end
  end

  return {
    name = 'BiEncoder',
    modules = modulesData,
    args = self.args
  }
end

function BiEncoder:resetPreallocation()
  -- Prototype for preallocated full context vector.
  self.contextProto = torch.Tensor()

  -- Prototype for preallocated full hidden states tensors.
  self.stateProto = torch.Tensor()

  -- Prototype for preallocated gradient of the backward context
  self.gradContextBwdProto = torch.Tensor()
end

-- we will build a merging module for the fwd and bwd 
function BiEncoder:_buildContextMerger()

	
	local paraTable = nn.ParallelTable()
	paraTable:add(nn.Identity()) -- for the fwdContext
	
	paraTable:add(nn.Reverse(2, true)) -- reverse the backward context, so the states are 'aligned'
	
	local contextMerger = nn.Sequential():add(paraTable)
	local mergeModule
	if self.args.merge == 'concat' then
		_G.logger:info(' * Merging the forward and backward encoder contexts by concatenation')
		mergeModule = nn.JoinTable(2, 2)
	else
		_G.logger:info(' * Merging the forward and backward encoder contexts by sum')
		mergeModule = nn.CAddTable()
	end
	
	contextMerger:add(mergeModule)
	
	return contextMerger
end

function BiEncoder:_buildStateMerger()
	local zipTable = nn.ZipTable()
	
	local paraTable = nn.ParallelTable()
	
	-- for each layer, we merge the cell and the hidden state of the LSTM from fwd and bwd
	for i = 1, self.args.numEffectiveLayers do
		
		local mergeModule
		if self.args.merge == 'concat' then
			mergeModule = nn.JoinTable(2, 2)
		else
			mergeModule = nn.CAddTable()
		end
		
		paraTable:add(mergeModule)
	end
	
	local stateMerger = nn.Sequential()
	stateMerger:add(zipTable)
	stateMerger:add(paraTable)
	
	return stateMerger
end

function BiEncoder:_buildBridge()
	
	local bridge
	
	self.args.bridge = self.args.bridge or 'nil'
	
	if self.args.bridge == 'copy' then
		_G.logger:info(" * Identity bridge between encoder and decoder hidden states")
		bridge = nn.MapTable(nn.Identity())
	elseif self.args.bridge == 'affine' then
		_G.logger:info(" * Non-linear affine transformation between encoder and decoder hidden states")
		bridge = nn.Sequential()
			:add(nn.JoinTable(2))
		bridge:add(nn.Linear(self.args.hiddenSize * self.args.numEffectiveLayers, self.args.hiddenSize * self.args.nDecLayers, false))
		bridge:add(nn.Tanh())
		bridge:add(nn.View(-1, self.args.nDecLayers, self.args.hiddenSize))
		bridge:add(nn.SplitTable(2))
	elseif self.args.bridge == 'nil' then
		_G.logger:info(" * No bridge between encoder and decoder hidden states")
		bridge = nn.NilModule() -- mapping output to nil, and gradInput to nil
	else
		error("Bridge type not implemented")
	end
	
	return bridge
end

function BiEncoder:maskPadding()
  self.fwd:maskPadding()
  self.bwd:maskPadding()
end

function BiEncoder:forward(batch)
  
	-- First, run forward pass for the two directional recurrent encoders
  local fwdStates, fwdContext = self.fwd:forward(batch)
  reverseInput(batch)
  local bwdStates, bwdContext = self.bwd:forward(batch)
  reverseInput(batch)
  
  -- Second, merge them using the mergers
  local contextMergerInput = {fwdContext, bwdContext}
  local stateMergerInput = {fwdStates, bwdStates}
  
  local context = self.contextMerger:forward(contextMergerInput)
  local encOutStates = self.stateMerger:forward(stateMergerInput)
  
  local states = self.bridge:forward(encOutStates)
  
  -- Store input during training for the backward pass
  if self.train then
		self.contextMergerInput = contextMergerInput
		self.stateMergerInput = stateMergerInput
		self.bridgeInput = encOutStates
	end
  
  return states, context
end

function BiEncoder:backward(batch, gradBridgeStatesOutput, gradContextOutput)

  local gradContextOutputFwd
  local gradContextOutputBwd

  local gradStatesOutputFwd
  local gradStatesOutputBwd
  
  --backward pass from the bridge
  local gradStatesOutput = self.bridge:backward(self.bridgeInput, gradBridgeStatesOutput)
  
  --backward pass for the context merger 
  local gradContext = self.contextMerger:backward(self.contextMergerInput, gradContextOutput)
  gradContextOutputFwd = gradContext[1]
  gradContextOutputBwd = gradContext[2]
  
  -- backward pass from the state merger
  local gradStates
  if gradStatesOutput then -- it can be nil if use nil bridge
		gradStates = self.stateMerger:backward(self.stateMergerInput, gradStatesOutput)
		gradStatesOutputFwd = gradStates[1]
		gradStatesOutputBwd = gradStates[2]
  end
  
  
	-- backward pass for the forward encoder
  local gradInputFwd = self.fwd:backward(batch, gradStatesOutputFwd, gradContextOutputFwd)
	
	-- backward pass for the backward encoder
  local gradInputBwd = self.bwd:backward(batch, gradStatesOutputBwd, gradContextOutputBwd)

  for t = 1, batch.sourceLength do
    onmt.utils.Tensor.recursiveAdd(gradInputFwd[t], gradInputBwd[batch.sourceLength - t + 1])
  end
  
  self.contextMergerInput = nil

  return gradInputFwd
end
