local LSTM, parent = torch.class('onmt.LSTMCell', 'onmt.Network')

--]]
function LSTM:__init(inputSize, hiddenSize, dropout)
	dropout = dropout or 0
  parent.__init(self, self:_buildModel(inputSize, hiddenSize, dropout))
end



function LSTM:_buildModel(inputSize, hiddenSize, dropout)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local prevC = inputs[1]
  local prevH = inputs[2]
  local x = inputs[3]
  
  -- Apply dropout to input
  
  if dropout > 0 then
		x = nn.Dropout(dropout)(x)
  end

  -- Evaluate the input sums at once for efficiency.
  local i2h = nn.Linear(inputSize, 4 * hiddenSize)(x)
  local h2h = nn.Linear(hiddenSize, 4 * hiddenSize)(prevH)
  local allInputSums = nn.CAddTable()({i2h, h2h})

  local reshaped = nn.Reshape(4, hiddenSize)(allInputSums)
  local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)

  -- Decode the gates.
  local inGate = nn.Sigmoid()(n1)
  local forgetGate = nn.Sigmoid()(n2)
  local outGate = nn.Sigmoid()(n3)

  -- Decode the write inputs.
  local inTransform = nn.Tanh()(n4)

  -- Perform the LSTM update.
  local nextC = nn.CAddTable()({
    nn.CMulTable()({forgetGate, prevC}),
    nn.CMulTable()({inGate, inTransform})
  })
  
  -- Gated cells form the output.
  local nextH = nn.CMulTable()({outGate, nn.Tanh()(nextC)})

  return nn.gModule(inputs, {nextC, nextH})
end

return LSTM
