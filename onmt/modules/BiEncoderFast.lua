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
local BiEncoder, parent = torch.class('onmt.BiEncoderFast', 'nn.Container')

--[[ Create a bi-encoder.

Parameters:

  * `input` - input neural network.
  * `rnn` - recurrent template module.
  * `merge` - fwd/bwd merge operation {"concat", "sum"}
]]
function BiEncoder:__init(input, rnn, merge)
  parent.__init(self)

  self.fwd = onmt.Encoder.new(input, rnn)
  self.bwd = onmt.Encoder.new(input:clone('weight', 'bias', 'gradWeight', 'gradBias'), rnn:clone())
  self.wordEmb = input:clone('weight', 'bias', 'gradWeight', 'gradBias')

  self.args = {}
  self.args.merge = merge

  self.args.rnnSize = rnn.outputSize
  self.args.numEffectiveLayers = rnn.numEffectiveLayers
  self.args.layers = rnn.layers

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

	self:add(self.contextMerger)
  self:resetPreallocation()
end

--[[ Return a new BiEncoder using the serialized data `pretrained`. ]]
function BiEncoder.load(pretrained)
  local self = torch.factory('onmt.BiEncoderFast')()

  parent.__init(self)

  self.fwd = onmt.Encoder.load(pretrained.modules[1])
  self.bwd = onmt.Encoder.load(pretrained.modules[2])
  self.contextMerger = pretrained.modules[3]
  self.stateMerger = pretrained.modules[3]
  self.args = pretrained.args

  self:add(self.fwd)
  self:add(self.bwd)
  self:add(self.contextMerger)
  self:add(self.stateMerger)

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
    name = 'BiEncoderFast',
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

	_G.logger:info(' * Using a nn module to merge the forward and backward encoder contexts')
	local paraTable = nn.ParallelTable()
	paraTable:add(nn.Identity()) -- for the fwdContext
	
	paraTable:add(nn.Reverse(2, true)) -- reverse the backward context, so the states are 'aligned'
	
	local contextMerger = nn.Sequential():add(paraTable)
	local mergeModule
	if self.args.merge == 'concat' then
		mergeModule = nn.JoinTable(2, 2)
	else
		mergeModule = nn.CAddTable()
	end
	
	contextMerger:add(mergeModule)
	
	return contextMerger
end

function BiEncoder:_buildStateMerger()
	
	_G.logger:info(' * Using a nn module to merge the forward and backward encoder states')
	
	local zipTable = nn.ZipTable()
	
	local paraTable = nn.ParallelTable()
	
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

function BiEncoder:maskPadding()
  self.fwd:maskPadding()
  self.bwd:maskPadding()
end

function BiEncoder:forward(batch)
  if self.statesProto == nil then
    self.statesProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                         self.stateProto,
                                                         { batch.size, self.args.hiddenSize })
  end
  
  self.buffers = {}

  local states = onmt.utils.Tensor.reuseTensorTable(self.statesProto, { batch.size, self.args.hiddenSize })

  local fwdStates, fwdContext = self.fwd:forward(batch)
  reverseInput(batch)
  local bwdStates, bwdContext = self.bwd:forward(batch)
  reverseInput(batch)

  --~ if self.args.merge == 'concat' then
    --~ for i = 1, #fwdStates do
      --~ states[i]:narrow(2, 1, self.args.rnnSize):copy(fwdStates[i])
      --~ states[i]:narrow(2, self.args.rnnSize + 1, self.args.rnnSize):copy(bwdStates[i])
    --~ end
    --~ 
  --~ elseif self.args.merge == 'sum' then
    --~ for i = 1, #states do
      --~ states[i]:copy(fwdStates[i])
      --~ states[i]:add(bwdStates[i])
    --~ end
    --~ 
  --~ end
  
  local contextMergerInput = {fwdContext, bwdContext}
  local stateMergerInput = {fwdStates, bwdStates}
  
  if self.train then
		self.contextMergerInput = contextMergerInput
		self.stateMergerInput = stateMergerInput
	end
  
  local context = self.contextMerger:forward(contextMergerInput)
  local states = self.stateMerger:forward(stateMergerInput)
  
  return states, context
end

function BiEncoder:backward(batch, gradStatesOutput, gradContextOutput)
  gradStatesOutput = gradStatesOutput
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.hiddenSize })

  local gradContextOutputFwd
  local gradContextOutputBwd

  local gradStatesOutputFwd
  local gradStatesOutputBwd

  --~ if self.args.merge == 'concat' then
--~ 
    --~ for i = 1, #gradStatesOutput do
      --~ local statesSplit = gradStatesOutput[i]:chunk(2, 2)
      --~ table.insert(gradStatesOutputFwd, statesSplit[1])
      --~ table.insert(gradStatesOutputBwd, statesSplit[2])
    --~ end
  --~ elseif self.args.merge == 'sum' then
--~ 
    --~ gradStatesOutputFwd = gradStatesOutput
    --~ gradStatesOutputBwd = gradStatesOutput
  --~ end
  
  --backward pass for the merger 
  local gradContext = self.contextMerger:backward(self.contextMergerInput, gradContextOutput)
  gradContextOutputFwd = gradContext[1]
  gradContextOutputBwd = gradContext[2]
  
  local gradStates = self.stateMerger:backward(self.stateMergerInput, gradStatesOutput)
  gradStatesOutputFwd = gradStates[1]
  gradStatesOutputBwd = gradStates[2]
  
	-- backward pass for the forward encoder
  local gradInputFwd = self.fwd:backward(batch, gradStatesOutputFwd, gradContextOutputFwd)
  --~ 
	
	-- backward pass for the backward encoder
  local gradInputBwd = self.bwd:backward(batch, gradStatesOutputBwd, gradContextOutputBwd)

  for t = 1, batch.sourceLength do
    onmt.utils.Tensor.recursiveAdd(gradInputFwd[t], gradInputBwd[batch.sourceLength - t + 1])
  end
  
  self.contextMergerInput = nil

  return gradInputFwd
end
