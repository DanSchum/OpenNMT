require('nngraph')

--[[
Implementation of a single stacked-RHN step as
an nn unit.

      h^L_{t-1} --- h^L_t
      c^L_{t-1} --- c^L_t
                 |


                 .
                 |
             [dropout]
                 |
      h^1_{t-1} --- h^1_t
      c^1_{t-1} --- c^1_t
                 |
                 |
                x_t

Computes $$(c_{t-1}, h_{t-1}, x_t) => (c_{t}, h_{t})$$.

--]]
local RHN, parent = torch.class('onmt.RHN', 'onmt.Network')

--[[
Parameters:

  * `layers` - Number of RHN layers, L. Normally only 1
  * `inputSize` - Size of input layer
  * `hiddenSize` - Size of the hidden layers.
  * `dropout` - Dropout rate to use. Possibly only for input layer
--]]
function RHN:__init(layers, inputSize, hiddenSize, dropout)
  dropout = dropout or 0

  self.dropout = dropout
  self.layers = layers
  self.numEffectiveLayers = layers * 1
  self.outputSize = hiddenSize
  self.inputSize = inputSize
	
	self.recurrenceDepth = 8
	self.initialBias = -4

  parent.__init(self, self:_buildModel(layers, inputSize, hiddenSize, dropout, residual, dropout_input))
end

local function Dropout(input, noise)
	return nn.CMulTable()({input, noise})
end

--[[ Stack the LSTM units. ]]
function RHN:_buildModel(layers, inputSize, hiddenSize, dropout, residual, dropout_input, ln)
  local inputs = {}
  local outputs = {}

  for _ = 1, layers do
    table.insert(inputs, nn.Identity()()) -- h0: batchSize x hiddenSize
  end
  
  table.insert(inputs, nn.Identity()()) -- recurrentMask: batchSize x layers x hiddenSize
	local recurrentMasks = inputs[#inputs]
	
	recurrentMasks = nn.SplitTable(2)(recurrentMasks)

  table.insert(inputs, nn.Identity()()) -- x: batchSize x inputSize
  local x = inputs[#inputs]

  local prevInput
  local nextH

  for L = 1, layers do
    local input
    local inputDim

    if L == 1 then
      -- First layer input is x.
      input = x
      inputDim = inputSize
    else
      inputDim = hiddenSize
      input = nextH
      
      -- Apply input dropout to input (only from the second layer)
    end
    
    local prevH = inputs[L]
    local recurrentMask_L = nn.SelectTable(L)(recurrentMasks)

    nextH = self:_buildLayer(inputDim, hiddenSize, self.recurrenceDepth, self.initialBias)({prevH, recurrentMask_L, input})

    table.insert(outputs, nextH)
  end

  return nn.gModule(inputs, outputs)
end

--[[ Build a single RHN unit layer. ]]
function RHN:_buildLayer(inputSize, hiddenSize, depth, initialBias)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- prevH
  table.insert(inputs, nn.Identity()()) -- mask
  table.insert(inputs, nn.Identity()()) -- x

  local prevH = inputs[1]
  local hiddenMask = inputs[2]
  local x = inputs[3]
  
  local states = {[0] = prevH}

  for d = 1, depth do
		local i2h = {}
		local h2h = {}
		local tGate, transformGate, cGate
		
		local droppedHidden = Dropout(states[d-1], hiddenMask)
		if d == 1 then -- First depth layer: input is x
			-- we can group these computations to make it faster (later)
			
			local groupI2H = nn.Linear(inputSize, 2 * hiddenSize)(x)
			local groupH2H = nn.Linear(hiddenSize, 2 * hiddenSize)(droppedHidden)
			local allInputSum = nn.CAddTable()({groupI2H, groupH2H})
			
			local reshaped = nn.Reshape(2, hiddenSize)(allInputSum)
			local n1, n2 = nn.SplitTable(2)(reshaped):split(2)
			
			tGate = nn.Sigmoid()(nn.AddConstant(initialBias, false)(n1))
			transformGate = nn.Tanh()(n2)
		else
			
			local allInputSum = nn.Linear(hiddenSize, 2 * hiddenSize)(droppedHidden) 
			local reshaped = nn.Reshape(2, hiddenSize)(allInputSum)
			local n1, n2 = nn.SplitTable(2)(reshaped):split(2)
			
			tGate = nn.Sigmoid()(nn.AddConstant(initialBias, false)(n1))
			transformGate = nn.Tanh()(n2)
					
		end
		
		cGate = nn.AddConstant(1,false)(nn.MulConstant(-1, false)(tGate))
		-- Compute the state of current depth
		states[d] = nn.CAddTable()({
        nn.CMulTable()({cGate, states[d - 1]}),
        nn.CMulTable()({tGate, transformGate})
		})
  
  end
    
  local nextH = states[depth]

  return nn.gModule(inputs, {nextH})
end
