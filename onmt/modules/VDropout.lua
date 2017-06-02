local VDropout, Parent = torch.class('onmt.VDropout', 'nn.Module')

local function debugInput(Tensor)
	if Tensor:norm() ~= Tensor:norm() then
		error('Input is Nan !!!!')
	elseif Tensor:nElement() == 0 then
		error('Tensor is Empty')
	end
end

local function debugOutput(Tensor)
	if Tensor:norm() ~= Tensor:norm() then
		error('Output is Nan !!!!')
	end
end

function VDropout:__init(p, layer_size)
  Parent.__init(self)
  self.p = p or 0.5
  self.layer_size = layer_size
  self.train = true
  if self.p >= 1 or self.p < 0 then
    error('<Dropout> illegal percentage, must be 0 <= p < 1')
  end
  self.noiseInit = torch.Tensor(1):zero()
  local max_batch_size = max_batch_size or 128
  -- need to dimension it, otherwise won't keep its sharedness through resizing
  self.noise = torch.Tensor()
end

-- Cloning this module
-- Set the noise of the new module the same as the original
function VDropout:clone(...)
	
	local clone = parent.clone(...)
	
	clone.noise:set(self.noise)
	
	return clone
end

-- Noise initialization using Bernoulli distribution
function VDropout:initNoise(bsz)
	
	local batchSize = self.batchSize or bsz -- only update batchSize if self.batchSize is nil (default)
	self.noise:resize(batchSize, self.layer_size):zero()
	self.noise:bernoulli(1-self.p)
	self.noise:div(1-self.p)
end

function VDropout:setBatchSize(batchSize)
	self.batchSize = batchSize
end

-- During forward pass, the noise will be generated MANUALLY using the init noise function
-- The sequencer container will have to call the function for every forward pass
function VDropout:updateOutput(input)
	self.output:resizeAs(input):copy(input)
  if self.p > 0 and self.train then
   debugInput(self.noise)
   self.output:cmul(self.noise)
  end
  return self.output
end

function VDropout:updateGradInput(_, gradOutput)
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  if self.p > 0 and self.train then
    self.gradInput:cmul(self.noise) -- simply mask the gradients with the shared noise vector
  end
  return self.gradInput
end

function VDropout:setp(p)
  self.p = p
end

function VDropout:__tostring__()
  return string.format('%s(%f,%i)', torch.type(self), self.p, self.layer_size)
end



function VDropout:clearState()
  self.noise:set()
  return Parent.clearState(self)
end

function VDropout.dropoutWords(p, batch)
   local vocabMask = torch.Tensor()
   for i = 1, batch.sourceInput:size(1) do
      local vocab = {}
      local vocabMap = {}
      for j = 1, batch.sourceInput:size(2) do
        local x = batch.sourceInput[i][j]
        if x > onmt.Constants.EOS and not vocab[x] then
          table.insert(vocabMap, x)
          vocab[x]=#vocabMap
        end
      end
      vocabMask:resize(#vocabMap)
      vocabMask:bernoulli(1-p)
      for j = 1, batch.sourceInput:size(2) do
        local x = batch.sourceInput[i][j]
        if x > onmt.Constants.EOS and vocabMask[vocab[x]] == 0 then
          batch.sourceInput[i][j] = onmt.Constants.PAD
        end
      end
   end
end
